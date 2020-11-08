import pytest
import torch

from train import *
from datasets.sentiment import SentimentData
from datasets.mqnli import MQNLIData, MQNLIBertData
from modeling.pretrained_bert import PretrainedBertModule
from modeling.lstm import LSTMModule, LSTMSelfAttnModule
from modeling.transformer import TransformerModule
from modeling.cbow import CBOWModule

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
def mqnli_mini_sep_data():
    return MQNLIData("mqnli_data/mini.train.txt",
                     "mqnli_data/mini.dev.txt",
                     "mqnli_data/mini.test.txt",
                     use_separator=True)

@pytest.fixture
def mqnli_data():
    return MQNLIData("mqnli_data/mqnli.train.txt",
                     "mqnli_data/mqnli.dev.txt",
                     "mqnli_data/mqnli.test.txt",)

@pytest.fixture
def mqnli_bert_data():
    return MQNLIBertData("mqnli_data/mqnli.train.txt",
                         "mqnli_data/mqnli.dev.txt",
                         "mqnli_data/mqnli.test.txt",
                         "mqnli_data/bert-remapping.txt")

@pytest.fixture
def mqnli_bert_mini_data():
    return MQNLIBertData("mqnli_data/mini.train.txt",
                         "mqnli_data/mini.dev.txt",
                         "mqnli_data/mini.test.txt",
                         "mqnli_data/bert-remapping.txt")

save_path_1 = "sentiment_models/test_lstm.pt"
save_path_2 = "sentiment_models/test_train_lstm.pt"
save_path_3 = "sentiment_models/test_train_lstm_attn.pt"
save_path_4 = "sentiment_models/test_train_transformer.pt"
save_path_5 = "sentiment_models/test_train_ffnn.pt"
save_path_6 = "mqnli_models/test_mini_ffnn.pt"
save_path_7 = "mqnli_models/full_ffnn.pt"


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


def test_train_ffnn_mqnli_mini(mqnli_mini_data):
    model_config = {
        'task': 'mqnli',
        'output_classes': mqnli_mini_data.output_classes,
        'vocab_size': mqnli_mini_data.vocab_size,
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
        'run_steps': 3,
        'evals_per_epoch': 5,
        'patient_epochs': 1000,
        'lr': 0.001,
        'batch_first': False,
        'model_save_path': save_path_6
    }

    model = CBOWModule(**model_config).to(torch.device('cuda'))
    trainer = Trainer(mqnli_mini_data, model, **train_config)
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
        'max_epochs': 200,
        'evals_per_epoch': 5,
        'patient_epochs': 20,
        'lr': 0.001,
        'batch_first': False,
        'model_save_path': save_path_7
    }

    model = CBOWModule(**model_config).to(torch.device('cuda'))
    trainer = Trainer(mqnli_data, model, **train_config)
    trainer.train()
    # Dev Accuracy 94.9%


def test_train_ffnn_mqnli_emb_fixed(mqnli_data):
    model_config = {
        'task': 'mqnli',
        'output_classes': mqnli_data.output_classes,
        'vocab_size': mqnli_data.vocab_size,
        'hidden_dim': 128,
        'activation_type': 'relu',
        'dropout': 0.1,
        'embed_init_scaling': 0.1,
        'fix_embeddings': True,
        'batch_first': False,
        'device': torch.device("cuda")
    }

    train_config = {
        'batch_size': 2048,
        'max_epochs': 200,
        'evals_per_epoch': 5,
        'patient_epochs': 20,
        'lr': 0.001,
        'batch_first': False,
        'model_save_path': save_path_7
    }

    model = CBOWModule(**model_config).to(torch.device('cuda'))
    trainer = Trainer(mqnli_data, model, **train_config)
    trainer.train()

    # Dev Accuracy 85.53%

def test_train_lstm_mqnli(mqnli_data):
    model_config = {
        'task': 'mqnli',
        'output_classes': mqnli_data.output_classes,
        'vocab_size': mqnli_data.vocab_size,

        'embed_dim': 256,
        'lstm_hidden_dim': 128,
        'bidirectional': True,
        'num_lstm_layers': 2,
        'dropout': 0.1,
        'embed_init_scaling': 0.1,
        'batch_first': False,
        'device': torch.device("cuda")
    }
    train_config = {
        'batch_first': False,
        'batch_size': 512,
        'max_epochs': 400,
        'evals_per_epoch': 5,
        'patient_epochs': 40,
        'lr': 0.0004,
        'weight_norm': 0.001,
        'model_save_path': "mqnli_models/lstm.pt"
    }

    model = LSTMModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(mqnli_data, model, **train_config)
    trainer.train()

def test_run_transformer_mqnli_mini(mqnli_mini_data):
    model_config = {
        'hidden_dim': 128,
        'vocab_size': mqnli_mini_data.vocab_size,
        'output_classes': mqnli_mini_data.output_classes,
        'num_transformer_heads': 4,
        'num_transformer_layers': 4,
        'embed_init_scaling': 0.1,
        'dropout': 0,
        'device': torch.device("cuda")
    }
    train_config = {
        'batch_size': 1000,
        'max_epochs': 500,
        'evals_per_epoch': 5,
        'patient_epochs': 500,
        'lr': 0.01,
        'batch_first': False, # Transformers only accept length in first dimension and batch in second
        'model_save_path': "mqnli_models/transformer.pt"
    }
    model = TransformerModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(mqnli_mini_data, model, **train_config)
    trainer.train()

def test_run_transformer_mqnli(mqnli_data):
    model_config = {
        'hidden_dim': 128,
        'vocab_size': mqnli_data.vocab_size,
        'output_classes': mqnli_data.output_classes,
        'num_transformer_heads': 4,
        'num_transformer_layers': 4,
        'embed_init_scaling': 0.1,
        'dropout': 0.1,
        'device': torch.device("cuda")
    }
    train_config = {
        'batch_size': 2048,
        'max_epochs': 200,
        'evals_per_epoch': 3,
        'patient_epochs': 40,
        'lr': 0.003,
        'batch_first': False, # Transformers only accept length in first dimension and batch in second
        'model_save_path': "mqnli_models/transformer.pt"
    }
    model = TransformerModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(mqnli_data, model, **train_config)
    trainer.train()

def test_train_lstm_mqnli_mini(mqnli_mini_data):
    model_config = {
        'task': 'mqnli',
        'output_classes': mqnli_mini_data.output_classes,
        'vocab_size': mqnli_mini_data.vocab_size,

        'embed_dim': 256,
        'lstm_hidden_dim': 128,
        'bidirectional': True,
        'num_lstm_layers': 4,
        'dropout': 0,
        'embed_init_scaling': 0.1,
        'batch_first': False,
        'device': torch.device("cuda")
    }
    train_config = {
        'batch_first': False,
        'run_steps': 5,
        'batch_size': 500,
        'max_epochs': 400,
        'evals_per_epoch': 2,
        'patient_epochs': 400,
        'lr': 0.003,
        'weight_norm': 0,
        'model_save_path': "mqnli_models/lstm_sep_test.pt"
    }

    model = LSTMModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(mqnli_mini_data, model, **train_config)
    trainer.train()


def test_train_lstm_mqnli_mini_sep(mqnli_mini_sep_data):
    model_config = {
        'task': 'mqnli',
        'output_classes': mqnli_mini_sep_data.output_classes,
        'vocab_size': mqnli_mini_sep_data.vocab_size,

        'embed_dim': 256,
        'lstm_hidden_dim': 128,
        'bidirectional': True,
        'num_lstm_layers': 4,
        'dropout': 0,
        'embed_init_scaling': 0.1,
        'batch_first': False,
        'p_h_separator': 1,
        'device': torch.device("cuda")
    }
    train_config = {
        'batch_first': False,
        'batch_size': 15,
        'run_steps': 3,
        'max_epochs': 100,
        'evals_per_epoch': 2,
        'patient_epochs': 400,
        'lr': 0.003,
        'weight_norm': 0,
        'model_save_path': "mqnli_models/lstm_sep_test.pt"
    }

    model = LSTMModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(mqnli_mini_sep_data, model, **train_config)
    trainer.train()

def test_train_transformer_mqnli(mqnli_bert_mini_data):
    model_config = {
        "tokenizer_vocab_path": "mqnli_data/bert-vocab.txt",
        'device': torch.device("cuda")
    }
    train_config = {
        'batch_size': 8,
        'eval_batch_size': 64,
        'use_collate': False,

        'optimizer_type': 'adamw',
        'lr': 3e-5,
        'lr_scheduler_type': 'linear',
        'lr_warmup_ratio': 0.5,
        'weight_norm': 0.01,

        'max_epochs': 5,
        'evals_per_epoch': 5,
        'patient_epochs': 400,

        'model_save_path': "mqnli_models/bert_test.pt"
    }

    model = PretrainedBertModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(mqnli_bert_mini_data, model, **train_config)
    trainer.train()

def test_load_bert():
    save_path = "mqnli_models/bert/retrained_ber_1106_150804.pt"
    model = load_model(PretrainedBertModule, save_path)
