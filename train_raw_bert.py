import torch
import argparse

from train_manager import DEFAULT_RAW_BERT_OPTS
from modeling.raw_bert import RawBertModule
# from modeling.pretrained_bert import PretrainedBertModule
from modeling.trainer import Trainer
import experiment

from typing import Dict


class TrainRawBertExperiment(experiment.Experiment):
    def experiment(self, opts: Dict):
        data = torch.load(opts["data_path"])
        model = RawBertModule(
            tokenizer_vocab_path=opts["tokenizer_vocab_path"],
            num_hidden_layers=opts["num_hidden_layers"],
            hidden_size=opts["hidden_size"],
            num_attention_heads=opts["num_attention_heads"],
            output_classes=opts["output_classes"]
        )
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
    opts = experiment.parse_args(parser, DEFAULT_RAW_BERT_OPTS)
    e = TrainRawBertExperiment()
    e.run(opts)

if __name__ == "__main__":
    main()
