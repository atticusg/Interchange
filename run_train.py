from datasets.mqnli import MQNLIData
from modeling.cbow import CBOWModule
from train import Trainer

import torch

def main():
    mqnli_data = MQNLIData("mqnli_data/mqnli.train.txt",
                     "mqnli_data/mqnli.dev.txt",
                     "mqnli_data/mqnli.test.txt")

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
        'batch_size': 1024,
        'max_epochs': 200,
        'evals_per_epoch': 5,
        'patient_epochs': 20,
        'lr': 0.001,
        'batch_first': False,
        'model_save_path': "mqnli_models/cbow.pt"
    }

    # Accuracy % 94.71

    model = CBOWModule(**model_config).to(torch.device('cuda'))
    trainer = Trainer(mqnli_data, model, **train_config)
    trainer.train()


if __name__ == "__main__":
    main()