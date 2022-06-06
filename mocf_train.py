import os

import numpy as np
import torch
import time
import argparse
import json
from typing import Dict, Tuple, Callable
from pprint import pprint

from antra import LOC
from antra.interchange.mapping import AbstractionMapping
from antra.location import parse_str

from compgraphs.mqnli_bert import Full_MQNLI_Bert_CompGraph, \
    Abstr_MQNLI_Bert_CompGraph, MQNLI_Bert_CompGraph
from compgraphs.mqnli_logic import Full_MQNLI_Logic_CompGraph, \
    Full_Abstr_MQNLI_Logic_CompGraph

from datasets.mqnli import MQNLIBertDataset

from counterfactual.multi_objective.typings import MultiObjectiveTrainingConfig
from counterfactual.multi_objective.trainer import MultiObjectiveTrainer
from counterfactual.multi_objective.dataset import MQNLIMultiObjectiveDataset
from counterfactual.multi_objective.dataloader import MultiObjectiveDataLoader

from causal_abstraction.interchange import find_abstractions_batch
from causal_abstraction.success_rates import analyze_counts

import experiment
from modeling.pretrained_bert import PretrainedBertModule
from modeling.utils import get_model_locs

import probing.modules
import probing.utils

def get_mo_weight_function(conf: MultiObjectiveTrainingConfig) -> Tuple[Callable[[int], Dict[str, float]], bool]:
    if conf.mo_weight_type == 'fixed':
        def _weight_fn(epoch: int):
            return {
                'base': conf.mo_base_weight,
                'cf': conf.mo_cf_weight,
                'aug': conf.mo_aug_weight,
                'probe': conf.mo_probe_weight
            }
        return _weight_fn, True
    else:
        raise NotImplementedError

def setup_dataloader(conf: MultiObjectiveTrainingConfig,
                     dataset: MQNLIBertDataset, high_model, mapping, is_eval: bool):
    batch_size = conf.eval_batch_size if is_eval else conf.train_batch_size

    mo_dataset = MQNLIMultiObjectiveDataset(
        base_dataset=dataset,
        high_model=high_model,
        mapping=mapping,
        num_random_bases=conf.cf_eval_num_random_bases if is_eval else conf.cf_train_num_random_bases,
        num_random_ivn_srcs=conf.cf_eval_num_random_ivn_srcs if is_eval else conf.cf_train_num_random_ivn_srcs,
        impactful_ratio=conf.cf_impactful_ratio,
        fix_examples=True
    )
    mo_base_dataloader = mo_dataset.get_dataloader(batch_size=batch_size)

    weight_fn, weight_per_epoch = get_mo_weight_function(conf)
    return MultiObjectiveDataLoader(
        dataset=mo_dataset,
        dataloader=mo_base_dataloader,
        weight_fn=weight_fn,
        weight_per_epoch=weight_per_epoch,
    )

def setup_probe(conf: MultiObjectiveTrainingConfig, mapping: AbstractionMapping, low_base_model: PretrainedBertModule):
    intervened_nodes = {n for n in mapping.keys() if n not in {"input", "root"}}
    # assume only one node to intervene on for now
    high_node = list(intervened_nodes)[0]
    low_node_to_loc = mapping[high_node]
    if len(low_node_to_loc) > 1 or len(intervened_nodes) > 1:
        raise NotImplementedError
    low_node = list(low_node_to_loc.keys())[0]
    low_loc = low_node_to_loc[low_node]
    probe_output_classes = probing.utils.get_num_classes(high_node)
    probe_input_dim = probing.utils.get_low_hidden_dim("bert", low_base_model)
    return probing.modules.Probe(
        high_node=high_node,
        low_node=low_node,
        low_loc=low_loc,
        is_control=False,
        probe_output_classes=probe_output_classes,
        probe_input_dim=probe_input_dim,
        probe_max_rank=conf.probe_max_rank,
        probe_dropout=conf.probe_dropout
    )

class MultiObjectiveTrainExperiment(experiment.Experiment):
    def experiment(self, opts: Dict):
        print("Loading datasets...")
        base_data = torch.load(opts["data_path"])
        mapping = json.loads(opts["mapping"])
        for high_node, low_node_to_loc in mapping.items():
            for low_node, low_loc in low_node_to_loc.items():
                true_loc = parse_str(low_loc)
                low_node_to_loc[low_node] = true_loc
        opts["mapping"] = mapping
        conf = MultiObjectiveTrainingConfig(**opts)

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

        probe = setup_probe(conf, mapping, low_base_model).to(torch.device('cuda'))

        trainer = MultiObjectiveTrainer(
            train_dataloader,
            eval_dataloader,
            low_model=low_model,
            high_model=high_model,
            probe_model=probe,
            configs=conf,
            store_cf_pairs=conf.interx_num_cf_training_pairs
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

        low_locs = get_model_locs(high_intermediate_node, "bert_mocf_vp_simple")

        print(f"high node: {high_intermediate_node}, low_locs: {low_locs}")

        # set up models
        if conf.interx_num_cf_training_pairs == 0:
            # use old setup for interx experiment
            # here we use old bert compgraph class because it has argmax as root node
            low_base_compgraph = MQNLI_Bert_CompGraph(module)
            low_compgraph = Abstr_MQNLI_Bert_CompGraph(low_base_compgraph, low_intermediate_nodes)
            low_compgraph.set_cache_device(torch.device("cpu"))
            intervention_pairs = None
            interx_base_dataset = base_data.dev
        else:
            # In this case we are doing interventions on pairs encountered during training
            # we use new bert compgraph class (Full_MQNLI_Bert... vs MQNLI_Bert...)
            # because the new abstraction interface requires it
            # low_base_compgraph =
            low_compgraph = Full_MQNLI_Bert_CompGraph(module, output="argmax")
            low_compgraph.set_cache_device(torch.device("cpu"))
            # assert len(trainer.cf_training_pairs) == trainer.store_cf_pairs
            intervention_pairs = trainer.cf_training_pairs
            interx_base_dataset = base_data.train

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
            low_model=low_compgraph,
            high_model=high_abstr_compgraph,
            low_model_type="bert",
            low_nodes_to_indices=low_nodes_to_indices,
            dataset=interx_base_dataset,
            num_inputs=conf.interx_num_inputs,
            batch_size=conf.interx_batch_size,
            fixed_assignments=fixed_assignments,
            unwanted_low_nodes={"root", "input"},
            intervention_pairs=intervention_pairs
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
    opts = experiment.parse_args(parser, MultiObjectiveTrainingConfig())
    e = MultiObjectiveTrainExperiment()
    e.run(opts)

if __name__ == "__main__":
    main()
