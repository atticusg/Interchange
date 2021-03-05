from copy import copy
import json

from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz
import torch
import torch.nn.functional as F
import torch.nn as nn

class IntegratedGradientsBase:
    def __init__(self, model, data, classes):
        self.model = model
        self.data = data  # for decoding
        self.classes = classes

    def predict_with_ig(self, input_tuples):
        attrs, deltas = self.ig.attribute(
            inputs=input_tuples,
            baselines=None,
            target=input_tuples[-1],
            return_convergence_delta=True)

        # Z-score normalize the scores per example:
        scores = self.ig_summarize_attributions(attrs)

        # Input ids for metadata:
        input_ids = input_tuples[0]

        # Model predictions to keep as metadata:
        probs = self.predict_proba(input_tuples)
        preds = self.predict(input_tuples, probs=probs)
        probs = [dict(zip(self.classes, p)) for p in probs]

        # True class for metadata and for IG:
        labels = [self.classes[i] for i in input_tuples[-1]]

        # Accumulate all the results from this batch into a list of dicts
        # that is suitable for our work and easily modified (by `self.visualize`)
        # to use the captum visualization tools
        data = []
        iterator = zip(input_ids, labels, scores, deltas, attrs, preds, probs)
        for inputs, label, score, delta, attr, pred, prob in iterator:
            tokens = self.ids_to_tokens(inputs)
            d = {
                'input_ids': inputs,
                'raw_input': tokens,
                'true_class': label,
                'pred_class': pred,
                'pred_probs': prob,
                'attr_score': attr.sum().item(),
                'attr_class': None,
                'word_attributions': score,
                'convergence_score': delta.item()
            }
            data.append(d)
        return data

    @staticmethod
    def visualize(ig_data):
        recs = []
        for d in ig_data:
            # Minor adjustments so that `d` can be the params argument to
            # viz.VisualizationDataRecord
            params = copy(d)
            del params['input_ids']
            params['pred_prob'] = params['pred_probs'][params['pred_class']]
            del params['pred_probs']
            score_vis = viz.VisualizationDataRecord(**params)
            recs.append(score_vis)
        viz.visualize_text(recs)

    @staticmethod
    def ig_summarize_attributions(attrs):
        attrs = attrs.sum(dim=-1).squeeze(0)
        attrs = ((attrs.T - torch.mean(attrs, dim=-1)) / torch.norm(attrs, dim=-1)).T
        return attrs.detach().cpu().numpy()

    @staticmethod
    def to_json(ig_data, output_filename):
        for d in ig_data:
            d['input_ids'] = d['input_ids'].cpu().detach().tolist()
            d['word_attributions'] = list(d['word_attributions'])
            for k, v in d['pred_probs'].items():
                d['pred_probs'][k] = float(v)
            d['attr_score'] = float(d['attr_score'])
            d['convergence_score'] = float(d['convergence_score'])
        with open(output_filename, "wt") as f:
            json.dump(ig_data, f, sort_keys=True, indent=4)

    def predict_proba(self, input_tuples):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_tuples)
            preds = F.softmax(logits, dim=1)
            return preds.cpu().numpy()

    def predict(self, input_tuples, probs=None):
        """Class predictions; `probs` can be the output of
        `predict_proba` to avoid having to do that work twice
        where we want both kinds of prediction.
        """
        if probs is None:
            preds = self.predict_proba(input_tuples)
        else:
            preds = probs
        return [self.classes[i] for i in preds.argmax(1)]



class IntegratedGradientsBERT(IntegratedGradientsBase):
    def __init__(self, model, data=None, classes=('neutral', 'entailment', 'contradiction'), layer=None):
        super().__init__(model, data, classes)
        self.tokenizer = self.model.tokenizer
        if layer is None:
            self.layer = self.model.bert.embeddings
        else:
            self.layer = layer
        self.ig = LayerIntegratedGradients(
            self.ig_forward,
            self.layer)

    def ig_forward(self, input_ids, token_type_ids, attention_mask, original_input, label):
        input_tuple = (input_ids, token_type_ids, attention_mask, original_input, label)
        return self.model(input_tuple)

    def ids_to_tokens(self, inputs):
        return self.tokenizer.convert_ids_to_tokens(inputs)



class IgLSTMEmbeddingModule(nn.Module):
    # wrapper module around the original embedding module
    def __init__(self, embedding):
        super(IgLSTMEmbeddingModule, self).__init__()
        self.embedding = embedding

    def forward(self, input_ids):
        return self.embedding(input_ids)  # [batch_size, sentence_len]

class IgLSTMRNNModule(nn.Module):
    # wrapper module around the torch.nn.LSTM so that Captum works because it
    # does not accept forward functions to return nested tuples. The original
    # nn.LSTM.forward() returns (output, (h_n, c_n))
    def __init__(self, lstm_layer):
        super(IgLSTMRNNModule, self).__init__()
        self.lstm_layer = lstm_layer

    def forward(self, hidden):
        output, _ = self.lstm_layer(hidden)
        return output


# new LSTM IG class adopted for LSTM trained using BERT tokenization.
class IntegratedGradientsLSTM(IntegratedGradientsBase):
    def __init__(self, model, data=None, classes=('neutral', 'entailment', 'contradiction'),
                 layer: int=None):
        """ Provide an index (0 or 1) to indicate the lstm layer """
        super().__init__(model, data, classes)
        self.tokenizer = self.model.tokenizer # bert tokenizer
        self.embedding = IgLSTMEmbeddingModule(self.model.embedding.embedding)
        self.lstm_layers = [IgLSTMRNNModule(layer) for layer in self.model.lstm_layers]

        if layer is None:
            self.layer = self.embedding
        else:
            self.layer = self.lstm_layers[layer]
        self.ig = LayerIntegratedGradients(self.ig_forward, self.layer)

    def ig_forward(self, input_ids, token_type_ids, attention_mask, original_input, label):
        # Arguments correspond to elements in the input_tuple of the bert dataset
        # re-implement LSTM forward function using wrapper modules for IG
        emb_x = self.embedding(input_ids)
        hidden = emb_x
        for lstm_layer in self.lstm_layers:
            hidden = lstm_layer(hidden)
        hidden = hidden.transpose(0, 1)
        hidden_dim = hidden.shape[-1] // 2
        forward_out = hidden[-1, :, :hidden_dim]
        backward_out = hidden[0, :, hidden_dim:]
        repr = torch.cat((forward_out, backward_out), dim=1)

        repr = self.model.dropout0(repr)
        output = self.model.feed_forward1(repr)
        output = self.model.activation1(output)

        output = self.model.feed_forward2(output)
        output = self.model.activation2(output)
        output = self.model.logits(output)
        return output

    def ids_to_tokens(self, inputs):
        # use bert tokenizer to decode
        return self.tokenizer.convert_ids_to_tokens(inputs)


# Old version of LSTM code
# class IntegratedGradientsLSTM(IntegratedGradientsBase):
#     def __init__(self, model, data=None, classes=('neutral', 'entailment', 'contradiction'),
#                  layer: int=None):
#         """ Provide an index (0 or 1) to indicate the lstm layer """
#         super().__init__(model, data, classes)
#         self.embedding = IgLSTMEmbeddingModule(self.model.embedding.embedding)
#         self.lstm_layers = [IgLSTMRNNModule(layer) for layer in self.model.lstm_layers]
#
#         if layer is None:
#             self.layer = self.embedding
#         else:
#             self.layer = self.lstm_layers[layer]
#         self.ig = LayerIntegratedGradients(
#             self.ig_forward,
#             self.layer)
#
#     def ig_forward(self, input_ids, label):
#         emb_x = self.embedding(input_ids)
#         hidden = emb_x
#         for lstm_layer in self.lstm_layers:
#             hidden = lstm_layer(hidden)
#         hidden = hidden.transpose(0, 1)
#         hidden_dim = hidden.shape[-1] // 2
#         forward_out = hidden[-1, :, :hidden_dim]
#         backward_out = hidden[0, :, hidden_dim:]
#         repr =  torch.cat((forward_out, backward_out), dim=1)
#
#         repr = self.model.dropout0(repr)
#         output = self.model.feed_forward1(repr)
#         output = self.model.activation1(output)
#
#         output = self.model.feed_forward2(output)
#         output = self.model.activation2(output)
#         output = self.model.logits(output)
#         return output
#
#     def ids_to_tokens(self, inputs):
#         if self.data is None:
#             raise ValueError("Cannot decode")
#         return self.data.decode(inputs)