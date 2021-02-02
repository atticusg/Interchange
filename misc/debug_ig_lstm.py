from datasets.mqnli import get_collate_fxn
from modeling.pretrained_bert import PretrainedBertModule
from modeling.lstm import LSTMModule
import os
from modeling.utils import load_model
import torch
from torch.utils.data import DataLoader

from feature_importance import IntegratedGradientsBERT, IntegratedGradientsLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ig_load_model(src_basename, src_dirname="mqnli_models"):
    path = os.path.join(src_dirname, src_basename)
    if 'lstm' in src_basename:
        model_class = LSTMModule
    else:
        model_class = PretrainedBertModule

    model, _ = load_model(model_class, path, device=device)
    return model

def ig_load_data(src_basename, src_dirname="mqnli_data"):
    path = os.path.join(src_dirname, src_basename)
    data = torch.load(path)
    return data

def analyze_sample(model, data, examples, n=8, batch_size=4, output_filename=None, layer=None, shuffle=True):
    n_batches = int(n / batch_size)
    if 'LSTM' in model.__class__.__name__:
        ig_class = IntegratedGradientsLSTM
        # collate_fn = get_collate_fxn(examples, batch_first=False)
    else:
        ig_class = IntegratedGradientsBERT
        # collate_fn = None
    ig = ig_class(model, data, layer=layer)
    dataloader = DataLoader(examples, batch_size=batch_size, shuffle=shuffle, collate_fn=None)
    data = []
    for i, input_tuple in enumerate(dataloader, start=1):
        if i % 100 == 0:
            print(f"Batch {i} of {n_batches}")
        # input_tuple = (input_tuple[0].to(device), input_tuple[-1].to(device))
        # print("input_tuple[0].shape", input_tuple[0].shape)
        # print("input_tuple[-1].shape", input_tuple[-1].shape)
        # res = ig.ig_forward(*input_tuple)
        # print(res)
        input_tuple = tuple([x.to(device) for x in input_tuple])
        ig.model.train()
        res = ig.predict_with_ig(input_tuple)
        print(res)
        data += res
        if i == n_batches:
            break
    if output_filename:
        ig.to_json(data, output_filename)
    return data

def main():
    print("loading model")
    lstm_model_easy = ig_load_model("lstm-easy-best.pt")
    print("lstm_model_easy tokenizer", lstm_model_easy.tokenizer)
    print("loading data")
    lstm_data = ig_load_data("mqnli-bert-default.pt")
    lstm_res = analyze_sample(lstm_model_easy, lstm_data, lstm_data.dev,
                              layer=0, shuffle=False)
    print(lstm_res)

if __name__ == "__main__":
    main()