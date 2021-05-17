import torch
from torch.utils.data import DataLoader
import argparse
import json
from typing import Dict
from dataclasses import asdict

from antra.location import parse_str

from compgraphs.mqnli_bert import Full_MQNLI_Bert_CompGraph
from compgraphs.mqnli_logic import Full_MQNLI_Logic_CompGraph

from counterfactual import CounterfactualTrainingConfig
from counterfactual.multidataloader import MultiTaskDataLoader
from counterfactual.dataset import MQNLIRandomIterableCfDataset, MQNLIBertGraphInputDataset
from counterfactual.trainer import CounterfactualTrainer
from counterfactual.scheduling import LinearCfTrainingSchedule, FixedRatioSchedule

import experiment
from modeling.pretrained_bert import PretrainedBertModule


class CounterfactualTrainExperiment(experiment.Experiment):
    def experiment(self, opts: Dict):
        print("Loading datasets...")
        base_data = torch.load(opts["data_path"])
        mapping = json.loads(opts["mapping"])
        for high_node, low_node_to_loc in mapping.items():
            for low_node, low_loc in low_node_to_loc.items():
                true_loc = parse_str(low_loc)
                low_node_to_loc[low_node] = true_loc
        opts["mapping"] = mapping
        conf = CounterfactualTrainingConfig(**opts)

        # setup models
        print("Loading models...")
        high_model = Full_MQNLI_Logic_CompGraph(base_data)
        low_base_model = PretrainedBertModule(
            tokenizer_vocab_path=conf.tokenizer_vocab_path,
            output_classes=conf.output_classes
        )
        low_base_model = low_base_model.to(torch.device("cuda"))
        low_model = Full_MQNLI_Bert_CompGraph(low_base_model)

        # setup training dataset and dataloaders
        train_base_dataset = MQNLIBertGraphInputDataset(base_data.train)
        train_base_dataloader = train_base_dataset.get_dataloader(
            shuffle=True, batch_size=conf.train_batch_size)
        train_cf_dataset = MQNLIRandomIterableCfDataset(
            base_data.train, high_model, mapping,
            num_random_bases=conf.cf_train_num_random_bases,
            num_random_ivn_srcs=conf.cf_train_num_random_ivn_srcs,
            fix_examples=False
        )
        train_cf_dataloader = train_cf_dataset.get_dataloader(batch_size=conf.train_batch_size)
        train_schedule = LinearCfTrainingSchedule(
            base_dataset=train_base_dataset,
            batch_size=conf.train_batch_size,
            num_subepochs=conf.num_subepochs_per_epoch,
            warmup_subepochs=conf.scheduler_warmup_subepochs,
            ratio_step_size=conf.scheduler_warmup_step_size
        )
        # train_schedule = lambda x: [10, 10]
        train_dataloader = MultiTaskDataLoader(
            tasks=[train_base_dataloader, train_cf_dataloader],
            task_names=["base", "cf_random"],
            return_task_name=True,
            schedule_fn=train_schedule
        )

        # setup eval dataset and dataloaders
        eval_base_dataset = MQNLIBertGraphInputDataset(base_data.dev)
        eval_base_dataloader = train_base_dataset.get_dataloader(
            shuffle=False, batch_size=conf.eval_batch_size)
        eval_cf_dataset = MQNLIRandomIterableCfDataset(
            base_data.dev, high_model, mapping,
            num_random_bases=conf.cf_eval_num_random_bases,
            num_random_ivn_srcs=conf.cf_eval_num_random_ivn_srcs,
            fix_examples=True
        )
        num_cf_examples = conf.cf_eval_num_random_bases * conf.cf_eval_num_random_ivn_srcs
        eval_cf_dataloader = eval_cf_dataset.get_dataloader(batch_size=conf.eval_batch_size)
        eval_schedule = FixedRatioSchedule(
            dataset_sizes=[len(eval_base_dataset), num_cf_examples],
            batch_size=conf.eval_batch_size
        )
        eval_dataloader = MultiTaskDataLoader(
            tasks=[eval_base_dataloader, eval_cf_dataloader],
            task_names=["base", "cf_random"],
            return_task_name=True,
            schedule_fn=eval_schedule
        )

        trainer = CounterfactualTrainer(
            train_dataloader,
            eval_dataloader,
            low_model=low_model,
            high_model=high_model,
            configs=conf
        )
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
    opts = experiment.parse_args(parser, CounterfactualTrainingConfig())
    e = CounterfactualTrainExperiment()
    e.run(opts)

if __name__ == "__main__":
    main()