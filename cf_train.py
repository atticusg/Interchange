import os

import numpy as np
import torch
import time
from torch.utils.data import DataLoader, Dataset
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
from counterfactual.dataset import MQNLIRandomCfDataset, \
    MQNLIBertGraphInputDataset, MQNLIImpactfulCFDataset
from counterfactual.augmented import MQNLIRandomAugmentedDataset
from counterfactual.trainer import CounterfactualTrainer
from counterfactual.scheduling import LinearCfTrainingSchedule, FixedRatioSchedule

from causal_abstraction.interchange import find_abstractions_batch
from causal_abstraction.success_rates import analyze_counts

import experiment
from modeling.pretrained_bert import PretrainedBertModule
from modeling.utils import get_model_locs


def setup_dataloader(conf: CounterfactualTrainingConfig,
                     dataset, high_model, mapping, is_eval: bool):
    batch_size = conf.eval_batch_size if is_eval else conf.train_batch_size
    base_dataset = MQNLIBertGraphInputDataset(dataset)
    base_dataloader = base_dataset.get_dataloader(shuffle=True, batch_size=batch_size)

    dataset_args = {
        "base_dataset": dataset,
        "high_model": high_model,
        "mapping": mapping,
        "num_random_bases": conf.cf_eval_num_random_bases if is_eval else conf.cf_train_num_random_bases,
        "num_random_ivn_srcs": conf.cf_eval_num_random_ivn_srcs if is_eval else conf.cf_train_num_random_ivn_srcs,
        "fix_examples": is_eval
    }

    if conf.cf_type == "random_only":
        cf_task_name = "cf_random"
        cf_dataset = MQNLIRandomCfDataset(**dataset_args)
    elif conf.cf_type == "augmented":
        cf_task_name = "augmented"
        cf_dataset = MQNLIRandomAugmentedDataset(**dataset_args)
    elif conf.cf_type == "impactful":
        cf_task_name = "cf_impactful"
        cf_dataset = MQNLIImpactfulCFDataset(
            **dataset_args, impactful_ratio=conf.cf_impactful_ratio
        )
    else:
        raise ValueError(f"Invalid cf_type {conf.cf_type}")
    cf_dataloader = cf_dataset.get_dataloader(batch_size=batch_size)

    num_cf_examples = conf.cf_eval_num_random_bases * conf.cf_eval_num_random_ivn_srcs if is_eval \
            else conf.cf_train_num_random_bases * conf.cf_train_num_random_ivn_srcs

    if is_eval:
        schedule = FixedRatioSchedule(
            dataset_sizes=[len(base_dataset), num_cf_examples],
            batch_size=batch_size
        )
    else:
        if conf.train_multitask_scheduler_type == "fixed":
            schedule = FixedRatioSchedule(
                dataset_sizes=[len(base_dataset), num_cf_examples],
                batch_size=batch_size,
                num_subepochs=conf.num_subepochs_per_epoch,
                ratio=conf.base_to_cf_ratio
            )
        elif conf.train_multitask_scheduler_type == "linear":
            schedule = LinearCfTrainingSchedule(
                base_dataset=base_dataset,
                batch_size=batch_size,
                num_subepochs=conf.num_subepochs_per_epoch,
                warmup_subepochs=conf.scheduler_warmup_subepochs,
                ratio_step_size=conf.scheduler_warmup_step_size,
                final_ratio=conf.base_to_cf_ratio
            )
        else:
            raise ValueError(f"Invalid multitask_scheduler_type "
                             f"{conf.train_multitask_scheduler_type}")

    multitask_dataloader = MultiTaskDataLoader(
        tasks=[base_dataloader, cf_dataloader],
        task_names=["base", cf_task_name],
        return_task_name=True,
        schedule_fn=schedule
    )

    return multitask_dataloader


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

        # set seed
        torch.manual_seed(conf.seed)
        np.random.seed(conf.seed)

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
        train_dataloader = setup_dataloader(conf, base_data.train, high_model, mapping, is_eval=False)
        eval_dataloader = setup_dataloader(conf, base_data.dev, high_model, mapping, is_eval=True)

        # check validity of eval metric name
        valid_metric_names = {f"eval_{task}_acc" for task in eval_dataloader.task_names}
        valid_metric_names.add("eval_avg_acc")
        if conf.primary_metric not in valid_metric_names:
            raise ValueError(f"Invalid metric name {conf.primary_metric}!")

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
                "train_duration": train_duration
            }
            res_dict.update(best_ckpt["metrics"])
        else:
            return trainer.evaluate_and_predict()

        print("======= Finished Training =======")

        if not conf.interx_after_train:
            pprint(res_dict)
            return res_dict

        print("======= Running interchange experiments =======")
        # Conduct interchange experiments from here on
        module = PretrainedBertModule(
            tokenizer_vocab_path=conf.tokenizer_vocab_path,
            output_classes=conf.output_classes
        )
        module.load_state_dict(best_ckpt["model_state_dict"])
        module = module.to(torch.device("cuda"))
        module.eval()

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
        interx_results = find_abstractions_batch(
            low_model=low_abstr_compgraph,
            high_model=high_abstr_compgraph,
            low_model_type="bert",
            low_nodes_to_indices=low_nodes_to_indices,
            dataset=base_data.dev,
            num_inputs=conf.interx_num_inputs,
            batch_size=conf.interx_batch_size,
            fixed_assignments=fixed_assignments,
            unwanted_low_nodes={"root", "input"}
        )
        interx_duration = time.time() - start_time
        print(f"Interchange experiment took {interx_duration:.2f}s")

        if conf.interx_save_results:
            save_path = os.path.join(conf.res_save_dir, "interx-res.pt")
            torch.save(interx_results, save_path)
            print(f"Saved interx results to {save_path}")

        counts_analysis = analyze_counts(interx_results)

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
