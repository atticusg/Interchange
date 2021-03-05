import torch
import argparse

from train_manager import DEFAULT_LSTM_OPTS
from modeling.lstm import LSTMModule
from trainer import Trainer
import experiment.manager

from typing import Dict


class TrainLSTMExperiment(experiment.Experiment):
    def experiment(self, opts: Dict):
        data = torch.load(opts["data_path"])
        model = LSTMModule(**opts)
        model = model.to(torch.device("cuda"))
        trainer = Trainer(data, model, **opts)
        ckpt, model_save_path = trainer.train()

        return {
            "model_save_path": model_save_path,
            "epoch": ckpt["epoch"],
            "step": ckpt["step"],
            "duration": ckpt["duration"],
            "loss": ckpt["loss"],
            "best_dev_acc": ckpt["best_dev_acc"]
        }

def main():
    parser = argparse.ArgumentParser()
    for arg_name, default_val in DEFAULT_LSTM_OPTS.items():
        arg_type = type(default_val)
        arg_type = int if arg_type == bool else arg_type
        parser.add_argument(f"--{arg_name}", type=arg_type, default=default_val)

    parser.add_argument("--id", type=int)
    parser.add_argument("--db_path", type=str)

    args = parser.parse_args()
    e = TrainLSTMExperiment()
    args = vars(args)
    experiment.manager.recover_boolean_args(args, DEFAULT_LSTM_OPTS)
    e.run(args)


if __name__ == "__main__":
    main()