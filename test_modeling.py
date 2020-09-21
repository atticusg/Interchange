import pytest

from train import *
from datasets.sentiment import SentimentData
from datasets.mqnli import MQNLIData
from modeling.lstm import LSTMModule, LSTMSelfAttnModule
from modeling.transformer import TransformerModule
from modeling.ffnn import CBOWModule
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
def mqnli_data():
    return MQNLIData("mqnli_data/mini.train.txt",
                     "mqnli_data/mini.dev.txt",
                     "mqnli_data/mini.test.txt")

save_path_1 = "sentiment_models/test_lstm.pt"
save_path_2 = "sentiment_models/test_train_lstm.pt"
save_path_3 = "sentiment_models/test_train_lstm_attn.pt"
save_path_4 = "sentiment_models/test_train_transformer.pt"
save_path_5 = "sentiment_models/test_train_ffnn.pt"
save_path_6 = "mqnli_models/test_mini_ffnn.pt"


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
        'max_epochs': 100,
        'evals_per_epoch': 5,
        'patient_epochs': 20,
        'lr': 0.0003,
        'model_save_path': save_path_3
    }
    model_config = {
        'embed_dim': 128,
        'vocab_size': sentiment_data.vocab_size,
        'bidirectional': True,
        'lstm_hidden_dim': 64,
        'attn_query_dim': 10,
        'device': torch.device("cuda")
    }

    model = LSTMSelfAttnModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(sentiment_data, model, **train_config)
    trainer.train()

def test_load_and_eval(sentiment_data):
    model, _ = load_model(LSTMModule, save_path_1)
    corr, total, _, _ = evaluate_and_predict(sentiment_data.dev, model)
    acc = corr / total
    print("Test Accuracy: %.4f" % acc)

def test_run_transformer(sentiment_transformer_data):
    model_config = {
        'hidden_dim': 128,
        'vocab_size': sentiment_transformer_data.vocab_size,
        'output_classes': sentiment_transformer_data.output_classes,
        'num_transformer_heads': 2,
        'num_transformer_layers': 2,
        'embed_init_scaling': 0.1,
        'dropout': 0.1,
        'device': torch.device("cuda")
    }
    train_config = {
        'batch_size': 2048,
        'max_epochs': 100,
        'evals_per_epoch': 5,
        'patient_epochs': 20,
        'lr': 0.0003,
        'batch_first': False, # Transformers only accept length in first dimension and batch in second
        'model_save_path': save_path_4
    }
    model = TransformerModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(sentiment_transformer_data, model, **train_config)
    trainer.train()

def test_run_ffnn(sentiment_transformer_data):
    model_config = {
        'hidden_dim': 128,
        'vocab_size': sentiment_transformer_data.vocab_size,
        'output_classes': sentiment_transformer_data.output_classes,
        'activation_type': 'relu',
        'dropout': 0.1,
        'embed_init_scaling': 0.1,
        'batch_first': False,
        'device': torch.device("cuda")
    }
    train_config = {
        'batch_size': 2048,
        'max_epochs': 100,
        'evals_per_epoch': 5,
        'patient_epochs': 20,
        'lr': 0.0003,
        'batch_first': False,
        # Only accept length in first dimension and batch in second
        'model_save_path': save_path_4
    }
    model = CBOWModule(**model_config).to(torch.device('cuda'))
    trainer = Trainer(sentiment_transformer_data, model, **train_config)
    trainer.train()


def test_train_ffnn_mqnli(mqnli_data):
    model_config = {
        'task': 'mqnli',
        'output_classes': mqnli_data.output_classes,
        'vocab_size': mqnli_data.vocab_size,
        'hidden_dim': 128,
        'activation_type': 'relu',
        'dropout': 0.1,
        'embed_init_scaling': 0.1,
        'batch_first': False,
        'device': torch.device("cuda")
    }

    train_config = {
        'batch_size': 1000,
        'max_epochs': 1000,
        'evals_per_epoch': 5,
        'patient_epochs': 1000,
        'lr': 0.001,
        'batch_first': False,
        'model_save_path': save_path_6
    }

    model = CBOWModule(**model_config).to(torch.device('cuda'))
    trainer = Trainer(mqnli_data, model, **train_config)
    trainer.train()


def test_modularize():
    f = lambda x, y: x * y
    f_module = modularize(f)