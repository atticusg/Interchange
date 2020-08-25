import pytest
import torch


from train import *
from datasets import SentimentData
from lstm_model import LSTMModule, LSTMSelfAttnModule


@pytest.fixture
def sentiment_data():
    return SentimentData("sentiment_data/senti.train.tsv",
                         "sentiment_data/senti.dev.tsv",
                         "sentiment_data/senti.test.tsv")

save_path_1 = "sentiment_models/test_lstm.pt"
save_path_2 = "sentiment_models/test_train_lstm.pt"
save_path_3 = "sentiment_models/test_train_lstm_attn.pt"

def test_train_lstm(sentiment_data):
    train_config = {
        'batch_size': 2048,
        'max_epochs': 100,
        'evals_per_epoch': 5,
        'patient_epochs': 20,
        'lr': 0.0003,
        'model_save_path': save_path_1
    }
    model_config = {
        'embed_dim': 100,
        'vocab_size': sentiment_data.vocab_size,
        'bidirectional': True,
        'lstm_hidden_dim': 25,
        'embed_init_scaling': 0.1,
        'device': torch.device("cuda")
    }
    model = LSTMModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(sentiment_data, model, **train_config)
    trainer.train()

def test_train_lstm_self_attn(sentiment_data):
    train_config = {
        'batch_size': 2000,
        'run_steps': 20,
        'max_epochs': 100,
        'evals_per_epoch': 5,
        'patient_epochs': 20,
        'lr': 0.0003,
        'model_save_path': "sentiment_models/test_lstm.pt"
    }
    model_config = {
        'embed_dim': 100,
        'vocab_size': sentiment_data.vocab_size,
        'bidirectional': True,
        'lstm_hidden_dim': 25,
        'attn_query_dim': 10,
        'device': torch.device("cuda")
    }

    model = LSTMSelfAttnModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(sentiment_data, model, **train_config)
    trainer.train()

def test_load_and_eval(sentiment_data):
    model, _ = load_model(LSTMModule, save_path_1)
    corr, total, _, _ = evaluate_and_predict(sentiment_data.test, model)
    acc = corr / total
    print("Test Accuracy: %.4f" % acc)