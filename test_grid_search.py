import pytest
import torch

from grid_search import GridSearch
from modeling.lstm import LSTMModule
from datasets.mqnli import MQNLIData

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


@pytest.fixture
def base_train_config():
    return {
        'batch_first': False,
        'batch_size': 512,
        'max_epochs': 100,
        'evals_per_epoch': 4,
        'patient_epochs': 20,
        'lr': 0.001,
        'weight_norm': 0.001,
        'model_save_path': "mqnli_models/lstm"
    }

@pytest.fixture
def base_lstm_model_config(mqnli_mini_data):
    return {
        'task': 'mqnli',
        'output_classes': mqnli_mini_data.output_classes,
        'vocab_size': mqnli_mini_data.vocab_size,

        'embed_dim': 256,
        'lstm_hidden_dim': 128,
        'bidirectional': True,
        'num_lstm_layers': 2,
        'dropout': 0.1,
        'embed_init_scaling': 0.1,
        'batch_first': False,
        'device': torch.device("cuda")
    }


def test_grid_search_basic(mqnli_mini_data, base_lstm_model_config, base_train_config):
    grid_dict = {"lr": [1., 0.1, 0.01, 0.001],
                 "weight_norm": [0, 0.001, 0.01, 0.1]}
    gs = GridSearch(LSTMModule, mqnli_mini_data, base_lstm_model_config, base_train_config, "mqnli_models/test_db1.db")
    gs.execute(grid_dict)

def test_grid_search_2epochs(mqnli_mini_data, base_lstm_model_config, base_train_config):
    base_train_config["max_epochs"] = 2
    grid_dict = {"lr": [1., 0.1, 0.01, 0.001],
                 "weight_norm": [0, 0.01]}
    gs = GridSearch(LSTMModule, mqnli_mini_data, base_lstm_model_config, base_train_config, "mqnli_models/test_db1.db")
    gs.execute(grid_dict)

def test_grid_search_5epochs(mqnli_mini_data, base_lstm_model_config, base_train_config):
    base_train_config["max_epochs"] = 5
    grid_dict = {"bidirectional": [True, False],
                 "num_lstm_layers": [1,2,4],
                 "lr": [1., 0.3, 0.1, 0.03, 0.01, 0.003],
                 "weight_norm": [0, 0.01]}
    gs = GridSearch(LSTMModule, mqnli_mini_data, base_lstm_model_config, base_train_config, "mqnli_models/test_db1.db")
    gs.execute(grid_dict)