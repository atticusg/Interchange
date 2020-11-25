import pytest

from trainer import *
from datasets.sentiment import SentimentData
from datasets.mqnli import MQNLIData
from modeling.lstm import LSTMModule, LSTMSelfAttnModule
from modeling.transformer import TransformerModule
from modeling.cbow import CBOWModule
from modeling.utils import modularize

@pytest.fixture
def sentiment_data():
    return SentimentData("sentiment_data/senti.train.tsv",
                         "sentiment_data/senti.dev.tsv",
                         "sentiment_data/senti.test.tsv")

@pytest.fixture
def sentiment_transformer_data():
    return SentimentData("sentiment_data/senti.train.tsv",
                         "sentiment_data/senti.dev.tsv",
                         "sentiment_data/senti.test.tsv",
                         for_transformer=True)

@pytest.fixture
def mqnli_mini_data():
    return MQNLIData("mqnli_data/mini.train.txt",
                     "mqnli_data/mini.dev.txt",
                     "mqnli_data/mini.test.txt")

@pytest.fixture
def mqnli_data():
    return MQNLIData("mqnli_data/mqnli.train.txt",
                     "mqnli_data/mqnli.dev.txt",
                     "mqnli_data/mqnli.test.txt")

save_path_1 = "sentiment_models/test_lstm.pt"
save_path_2 = "sentiment_models/test_train_lstm.pt"
save_path_3 = "sentiment_models/test_train_lstm_attn.pt"
save_path_4 = "sentiment_models/test_train_transformer.pt"
save_path_5 = "sentiment_models/test_train_ffnn.pt"
save_path_6 = "mqnli_models/test_mini_ffnn.pt"
save_path_7 = "mqnli_models/full_ffnn.pt"




def test_load_and_eval(sentiment_data):
    model, _ = load_model(LSTMModule, save_path_1)
    corr, total, _, _ = evaluate_and_predict(sentiment_data.dev, model)
    acc = corr / total
    print("Test Accuracy: %.4f" % acc)


def test_modularize():
    f = lambda x, y: x * y
    f_module = modularize(f)