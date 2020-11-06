import torch
import argparse

from train_bert import DEFAULT_OPTS
from modeling.pretrained_bert import PretrainedBertModule
from train import Trainer
from experiment import Experiment

from typing import Dict


class TrainBertExperiment(Experiment):
    def experiment(self, opts: Dict):
        data = torch.load(opts["data_path"])
        model = PretrainedBertModule(
            tokenizer_vocab_path=opts["tokenizer_vocab_path"],
            device=opts["device"]
        )
        model = model.to(model.device)
        trainer = Trainer(data, model, opts=opts)
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
    for arg_name, default_val in DEFAULT_OPTS.items():
        arg_type = type(default_val)
        arg_type = int if arg_type == bool else arg_type
        parser.add_argument(f"--{arg_name}", type=arg_type, default=default_val)

    parser.add_argument("--id", type=int)
    parser.add_argument("--db_path", type=str)

    args = parser.parse_args()
    e = TrainBertExperiment()
    opts = vars(args)
    e.run(vars(args))


if __name__ == "__main__":
    main()