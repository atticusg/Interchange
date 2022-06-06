import torch

from modeling.trainer import evaluate_and_predict
from modeling.utils import load_model
from modeling import PretrainedBertModule

device = torch.device("cuda")
data = torch.load("data/mqnli/preprocessed/bert-hard.pt")
bert_model, _ = load_model(PretrainedBertModule, "data/models/bert-hard_abl-best.pt",
                           opts={"tokenizer_vocab_path": "data/tokenization/bert-vocab.txt"},
                           device=device)
# lstm_model, _ = load_model(LSTMModule, "data/models/lstm-hard-best.pt", device=device)
# test_set = data.test
# print(f"Got {len(test_set)} examples for test set")

# bert_corr, bert_total = evaluate_and_predict(test_set, bert_model, eval_batch_size=256, device=device)
# print(f"BERT's test set accuracy {bert_corr / bert_total : .2%}") #88.50 hard, 54.51 abl

# lstm_corr, lstm_total = evaluate_and_predict(test_set, lstm_model, eval_batch_size=256, device=device)
# print(f"LSTM's test set accuracy {lstm_corr / lstm_total : .2%}") #46.32 hard


train_set = data.train
bert_corr, bert_total = evaluate_and_predict(train_set, bert_model, eval_batch_size=256, device=device)
print(f"BERT's test set accuracy {bert_corr / bert_total : .2%}")
