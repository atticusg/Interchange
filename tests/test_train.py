import pytest
import torch

from trainer import *
from modeling.lstm import LSTMModule, LSTMSelfAttnModule
from modeling.transformer import TransformerModule
from modeling.cbow import CBOWModule

@pytest.fixture
def mqnli_data():
    return torch.load("data/mqnli/preprocessed/lstm-easy.pt")

@pytest.fixture
def mqnli_bert_data():
    return torch.load("data/mqnli/preprocessed/bert-easy.pt")


save_path_6 = "data/training/test/test_mini_ffnn.pt"
save_path_7 = "data/training/test/full_ffnn.pt"



def test_train_ffnn_mqnli(mqnli_data):
    # depreciated, data object has changed
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
    # deprecated, mqnli_data object has changed
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
    # deprecated, LSTM models now use BERT tokenization
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
        'model_save_path': "data/training/test/lstm.pt"
    }

    model = LSTMModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(mqnli_data, model, **train_config)
    trainer.train()


def test_run_transformer_mqnli(mqnli_data):
    # deprecated
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
        'model_save_path': "data/training/test/transformer.pt"
    }
    model = TransformerModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(mqnli_data, model, **train_config)
    trainer.train()


    model = LSTMModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(mqnli_mini_data, model, **train_config)
    trainer.train()


def test_train_lstm_mqnli_mini_sep(mqnli_mini_sep_data):
    # deprecated
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
        'model_save_path': "data/training/test/lstm_sep_test.pt"
    }

    model = LSTMModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(mqnli_mini_sep_data, model, **train_config)
    trainer.train()


def test_train_lstm_mqnli_hard():
    data = torch.load("data/mqnli/preprocessed/lstm-hard.pt")
    print("output classes", data.output_classes)
    model_config = {
        'task': 'mqnli',
        'output_classes': data.output_classes,
        'vocab_size': data.vocab_size,

        'embed_dim': 256,
        'lstm_hidden_dim': 128,
        'bidirectional': True,
        'num_lstm_layers': 4,
        'dropout': 0,
        'embed_init_scaling': 0.1,
        'batch_first': False
    }
    train_config = {
        'batch_first': False,
        'batch_size': 8,
        'max_epochs': 100,
        'evals_per_epoch': 2,
        'patient_epochs': 400,
        'lr': 0.0001,
        'lr_scheduler_type': 'step',
        'lr_step_epochs': 20,
        'lr_step_decay_rate': 1. / 3.,
        'weight_norm': 0,
        'model_save_path': "data/training/test/lstm_sep_test.pt"
    }

    model = LSTMModule(**model_config).to(torch.device("cuda"))
    trainer = Trainer(data, model, **train_config)
    trainer.train()
