import torch

from intervention import ComputationGraph
from intervention import GraphNode


def generate_lstm_fxn(lstm_layer):
    def _lstm_layer_fxn(x):
        res, _ = lstm_layer(x)
        return res
    return _lstm_layer_fxn

class MQNLI_LSTM_CompGraph(ComputationGraph):
    def __init__(self, lstm_model):
        if lstm_model.task != "mqnli":
            raise ValueError("The LSTM model must be for MQNLI!")
        self.model = lstm_model

        @GraphNode()
        def input(x):
            # (18,) or (18, batch_size)
            if len(x.shape) == 1:
                x = x.unsqueeze(1)
            assert len(x.shape) == 2 and x.shape[0] == 18, f"x.shape is {x.shape}"
            return self.model.embedding((x,))

        @GraphNode(input)
        def premise_emb(x):
            # (18, emb_size) or (18, batch_size, emb_size)
            assert len(x.shape) == 3 and x.shape[0] == 18 and x.shape[2] == self.model.embed_dim, \
                f"x.shape is {x.shape}"
            return self.model.premise_emb(x)

        @GraphNode(input)
        def hypothesis_emb(x):
            # (18, emb_size) or (18, batch_size, emb_size)
            assert len(x.shape) == 3 and x.shape[0] == 18 and x.shape[2] == self.model.embed_dim, \
                f"x.shape is {x.shape}"
            return self.model.hypothesis_emb(x)

        prem_node = premise_emb
        for i in range(len(self.model.lstm_layers)):
            lstm_forward_fxn = generate_lstm_fxn(self.model.lstm_layers[i])
            prem_node = GraphNode(prem_node, name=f"premise_lstm_{i}", forward=lstm_forward_fxn)

        hyp_node = hypothesis_emb
        for i, lstm_layer in enumerate(self.model.lstm_layers):
            lstm_forward_fxn = generate_lstm_fxn(self.model.lstm_layers[i])
            hyp_node = GraphNode(hyp_node, name=f"hypothesis_lstm_{i}", forward=lstm_forward_fxn)

        @GraphNode(prem_node, hyp_node)
        def concat_final_state(prem_h, hyp_h):
            # (9, batch_size, h_dim)
            h_dim = self.model.lstm_hidden_dim * (2 if self.model.bidirectional else 1)
            assert len(prem_h.shape) == 3 and prem_h.shape[0] == 9 and prem_h.shape[2] == h_dim, \
                f"hypothesis_lstm_{i} x.shape is {prem_h.shape}"
            assert len(hyp_h.shape) == 3 and hyp_h.shape[0] == 9 and hyp_h.shape[2] == h_dim, \
                f"hypothesis_lstm_{i} x.shape is {hyp_h.shape}"

            return self.model.concat_final_state(prem_h, hyp_h)

        @GraphNode(concat_final_state)
        def ffnn1(x):
            # (batch_size, h_dim)
            h_dim = self.model.lstm_hidden_dim * (4 if self.model.bidirectional else 2)
            assert (len(x.shape) == 2 and x.shape[1] == h_dim)

            return self.model.feed_forward1(x)

        @GraphNode(ffnn1)
        def activation1(x):
            # (batch_size, h_dim)
            h_dim = self.model.lstm_hidden_dim * (4 if self.model.bidirectional else 2)
            assert (len(x.shape) == 2 and x.shape[1] == h_dim)

            return self.model.activation1(x)

        @GraphNode(activation1)
        def ffnn2(x):
            # (batch_size, h_dim)
            h_dim = self.model.lstm_hidden_dim * (4 if self.model.bidirectional else 2)
            assert (len(x.shape) == 2 and x.shape[1] == h_dim)

            return self.model.feed_forward2(x)

        @GraphNode(ffnn2)
        def activation2(x):
            h_dim = self.model.lstm_hidden_dim * (4 if self.model.bidirectional else 2)
            assert (len(x.shape) == 2 and x.shape[1] == h_dim)

            return self.model.activation2(x)

        @GraphNode(activation2)
        def logits(x):
            h_dim = self.model.lstm_hidden_dim * (4 if self.model.bidirectional else 2)
            assert (len(x.shape) == 2 and x.shape[1] == h_dim)

            return self.model.logits(x)

        @GraphNode(logits)
        def root(x):
            return torch.argmax(x, dim=1)


        super(MQNLI_LSTM_CompGraph, self).__init__(root)


