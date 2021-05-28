import torch
import time
from torch.utils.data import DataLoader
import argparse
import json
from typing import Dict
from pprint import pprint

from antra import LOC
from antra.location import parse_str

from compgraphs.mqnli_bert import Full_MQNLI_Bert_CompGraph, \
    Abstr_MQNLI_Bert_CompGraph, MQNLI_Bert_CompGraph
from compgraphs.mqnli_logic import Full_MQNLI_Logic_CompGraph, \
    Full_Abstr_MQNLI_Logic_CompGraph

from counterfactual import CounterfactualTrainingConfig
from counterfactual.multidataloader import MultiTaskDataLoader
from counterfactual.dataset import MQNLIRandomCfDataset, MQNLIBertGraphInputDataset
from counterfactual.augmented import MQNLIRandomAugmentedDataset
from counterfactual.trainer import CounterfactualTrainer
from counterfactual.scheduling import LinearCfTrainingSchedule, FixedRatioSchedule

from causal_abstraction.interchange import find_abstractions_batch
from causal_abstraction.success_rates import analyze_counts

import experiment
from modeling.pretrained_bert import PretrainedBertModule
from modeling.utils import get_model_locs


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

        if conf.cf_type == "random_only":
            cf_task_name = "cf_random"
            train_cf_dataset = MQNLIRandomCfDataset(
                base_data.train, high_model, mapping,
                num_random_bases=conf.cf_train_num_random_bases,
                num_random_ivn_srcs=conf.cf_train_num_random_ivn_srcs,
                fix_examples=False
            )
        elif conf.cf_type == "augmented":
            cf_task_name = "augmented"
            train_cf_dataset = MQNLIRandomAugmentedDataset(
                base_data.train, high_model, mapping,
                num_random_bases=conf.cf_train_num_random_bases,
                num_random_ivn_srcs=conf.cf_train_num_random_ivn_srcs,
                fix_examples=False
            )
        else:
            raise ValueError(f"Invalid cf_type {conf.cf_type}")
        train_cf_dataloader = train_cf_dataset.get_dataloader(batch_size=conf.train_batch_size)

        if conf.train_multitask_scheduler_type == "fixed":
            num_train_cf_examples = conf.cf_train_num_random_bases * conf.cf_train_num_random_ivn_srcs
            train_schedule = FixedRatioSchedule(
                dataset_sizes=[len(train_base_dataset), num_train_cf_examples],
                batch_size=conf.train_batch_size,
                num_subepochs=conf.num_subepochs_per_epoch,
                ratio=conf.base_to_cf_ratio
            )
        elif conf.train_multitask_scheduler_type == "linear":
            train_schedule = LinearCfTrainingSchedule(
                base_dataset=train_base_dataset,
                batch_size=conf.train_batch_size,
                num_subepochs=conf.num_subepochs_per_epoch,
                warmup_subepochs=conf.scheduler_warmup_subepochs,
                ratio_step_size=conf.scheduler_warmup_step_size,
                final_ratio=conf.base_to_cf_ratio
            )
        else:
            raise ValueError(f"Invalid multitask_scheduler_type {conf.train_multitask_scheduler_type}")

        # train_schedule = lambda x: [10, 10]
        train_dataloader = MultiTaskDataLoader(
            tasks=[train_base_dataloader, train_cf_dataloader],
            task_names=["base", cf_task_name],
            return_task_name=True,
            schedule_fn=train_schedule
        )

        # setup eval dataset and dataloaders
        eval_base_dataset = MQNLIBertGraphInputDataset(base_data.dev)
        eval_base_dataloader = eval_base_dataset.get_dataloader(
            shuffle=False, batch_size=conf.eval_batch_size)

        if conf.cf_type == "random_only":
            eval_cf_dataset = MQNLIRandomCfDataset(
                base_data.dev, high_model, mapping,
                num_random_bases=conf.cf_eval_num_random_bases,
                num_random_ivn_srcs=conf.cf_eval_num_random_ivn_srcs,
                fix_examples=True
            )

        elif conf.cf_type == "augmented":
            eval_cf_dataset = MQNLIRandomAugmentedDataset(
                base_data.dev, high_model, mapping,
                num_random_bases=conf.cf_eval_num_random_bases,
                num_random_ivn_srcs=conf.cf_eval_num_random_ivn_srcs,
                fix_examples=False
            )
        else:
            raise ValueError(f"Invalid cf_type {conf.cf_type}")

        num_eval_cf_examples = conf.cf_eval_num_random_bases * conf.cf_eval_num_random_ivn_srcs
        eval_cf_dataloader = eval_cf_dataset.get_dataloader(batch_size=conf.eval_batch_size)
        eval_schedule = FixedRatioSchedule(
            dataset_sizes=[len(eval_base_dataset), num_eval_cf_examples],
            batch_size=conf.eval_batch_size
        )
        eval_dataloader = MultiTaskDataLoader(
            tasks=[eval_base_dataloader, eval_cf_dataloader],
            task_names=["base", cf_task_name],
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
        if not conf.eval_only:
            train_start_time = time.time()
            best_ckpt, model_save_path = trainer.train()
            train_duration = time.time() - train_start_time
            res_dict = {
                "model_save_path": model_save_path,
                "epoch": best_ckpt["epoch"],
                "avg_train_loss": best_ckpt["avg_train_loss"],
                "best_dev_base_acc": best_ckpt["best_dev_acc"],
                "best_dev_total_acc": best_ckpt["best_dev_total_acc"],
                "train_duration": train_duration
            }
        else:
            eval_res = trainer.evaluate_and_predict()
            dev_base_acc = eval_res["base_correct_cnt"] / eval_res["base_cnt"]
            dev_total_acc = eval_res["total_correct_cnt"]  / eval_res["total_cnt"]
            res_dict = {
                "best_dev_base_acc": dev_base_acc,
                "best_dev_total_acc": dev_total_acc
            }

        print("======= Finished Training =======")

        if not conf.interx_after_train:
            pprint(res_dict)
            return res_dict

        print("======= Running interchange experiments =======")

        # Conduct interchange experiments from here on
        assert not conf.eval_only
        module = PretrainedBertModule(
            tokenizer_vocab_path=conf.tokenizer_vocab_path,
            output_classes=conf.output_classes
        )
        module.load_state_dict(best_ckpt["model_state_dict"])
        module = module.to(torch.device("cuda"))
        module.eval()


        # load data
        data = base_data

        # assume only one high intermediate node and low node
        high_intermediate_node = None
        low_intermediate_node = None
        for high_node, low_dict in mapping.items():
            high_intermediate_node = high_node
            for low_node, low_loc in low_dict.items():
                low_intermediate_node = low_node
        low_intermediate_nodes = [low_intermediate_node]
        high_intermediate_nodes = [high_intermediate_node]

        low_locs = get_model_locs(high_intermediate_node, "bert")

        print(f"high node: {high_intermediate_node}, low_locs: {low_locs}")

        # set up models
        # here we use old bert compgraph class because it has argmax as root node
        low_base_compgraph = MQNLI_Bert_CompGraph(module)
        low_abstr_compgraph = Abstr_MQNLI_Bert_CompGraph(low_base_compgraph, low_intermediate_nodes)
        low_abstr_compgraph.set_cache_device(torch.device("cpu"))

        high_abstr_compgraph = Full_Abstr_MQNLI_Logic_CompGraph(
            high_model, high_intermediate_nodes)

        low_nodes_to_indices = {
            low_node: [LOC[:,x,:] for x in low_locs]
            for low_node in low_intermediate_nodes
        }

        fixed_assignments = {x: {x: None} for x in ["root", "input"]}

        # ------ Batched causal_abstraction experiment ------ #
        start_time = time.time()
        batch_results = find_abstractions_batch(
            low_model=low_abstr_compgraph,
            high_model=high_abstr_compgraph,
            low_model_type="bert",
            low_nodes_to_indices=low_nodes_to_indices,
            dataset=data.dev,
            num_inputs=conf.interx_num_inputs,
            batch_size=conf.interx_batch_size,
            fixed_assignments=fixed_assignments,
            unwanted_low_nodes={"root", "input"}
        )
        interx_duration = time.time() - start_time
        print(f"Interchange experiment took {interx_duration:.2f}s")

        counts_analysis = analyze_counts(batch_results)

        res_dict["interx_duration"] = interx_duration
        res_dict.update(counts_analysis)
        res_dict["interx_mappings"] = res_dict["mappings"]
        del res_dict["mappings"]

        pprint(res_dict)
        return res_dict



def main():
    parser = argparse.ArgumentParser()
    opts = experiment.parse_args(parser, CounterfactualTrainingConfig())
    e = CounterfactualTrainExperiment()
    e.run(opts)

if __name__ == "__main__":
    main()
