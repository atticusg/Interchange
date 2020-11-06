import torch
import argparse
from grid_search import GridSearch
from modeling.lstm import LSTMModule
from datasets.mqnli import MQNLIData


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("layers", type=int)
    args = parser.parse_args()

    mqnli_data = MQNLIData("mqnli_data/mqnli.train.txt",
                           "mqnli_data/mqnli.dev.txt",
                           "mqnli_data/mqnli.test.txt",
                           use_separator=True)
    base_lstm_model_config =  {
        'task': 'mqnli',
        'output_classes': mqnli_data.output_classes,
        'vocab_size': mqnli_data.vocab_size,

        'embed_dim': 256, # fix
        'lstm_hidden_dim': 128, # fix
        'bidirectional': True, # fix
        'dropout': 0.1,
        'p_h_separator': 1,
        'num_lstm_layers': 1,

        'embed_init_scaling': 0.1,
        'batch_first': False,
        'device': torch.device("cuda")
    }

    base_train_config = {
        'batch_first': False,
        'batch_size': 512,
        'max_epochs': 400,
        'evals_per_epoch': 5,
        'patient_epochs': 20,
        'lr': 0.001,
        'weight_norm': 0,
        'model_save_path': "mqnli_models/lstm_sep"
    }

    gs = GridSearch(LSTMModule, mqnli_data, base_lstm_model_config,
                    base_train_config, "experiment_data/bert.db")
    grid_dict = {"num_lstm_layers": [args.layers],
                 "lr": [0.001, 0.0003, 0.0001],
                 "dropout": [0, 0.1, 0.3],}
    gs.execute(grid_dict)


if __name__ == "__main__":
    main()