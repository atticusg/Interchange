import os
import torch
import argparse
import experiment
import csv
from datetime import datetime
from trainer import load_model

import compgraphs
import modeling
from modeling.utils import get_target_loc_dict

from probing.modules import Probe
from probing.dataset import ProbingData
from probing.trainer import ProbeTrainer
import probing.utils

from probe import DEFAULT_PROBING_OPTS

opts = {
    "model_path": "",
    "data_path": "",
    "model_type": "",

    "is_control": False,
    "probe_max_rank": 24,
    "probe_dropout": 0.1,
    "probe_train_num_examples": 64000,
    "probe_train_num_dev_examples": 3200,
    "probe_correct_examples_only": True,

    "probe_train_batch_size": 128,
    "probe_train_eval_batch_size": 256,
    "probe_train_weight_norm": 0.,
    "probe_train_max_epochs": 80,
    "probe_train_lr": 0.001,
    "probe_train_lr_patience_epochs": 4,
    "probe_train_lr_anneal_factor": 0.5,
    "res_save_dir": "",
    "res_save_path": "",
    "log_path": ""
}

def main():
    model_type = "bert"
    model_class = modeling.get_module_class_by_name(model_type)
    module, _ = load_model(model_class, "../data/models/bert-hard-best.pt")
    device = torch.device("cuda")
    module = module.to(device)
    module.eval()

    print("=== Loading data")
    data = torch.load("data/mqnli/preprocessed/bert-easy.pt")

    print("=== Constructing compgraph")
    lo_base_compgraph_class = compgraphs.get_compgraph_class_by_name(model_type)
    lo_base_compgraph = lo_base_compgraph_class(module)
    lo_abstr_compgraph_class = compgraphs.get_abstr_compgraph_class_by_name("bert")
    hi_compgraph = compgraphs.MQNLI_Logic_CompGraph(data)

    probe_input_dim = probing.utils.get_low_hidden_dim("bert", module)

    for low_node in probing.utils.get_low_nodes("bert"):
        print(f"\n=== Getting hidden vectors for low node {low_node}")
        lo_abstr_compgraph = lo_abstr_compgraph_class(lo_base_compgraph,
                                                      [low_node])
        probe_data = ProbingData(data, hi_compgraph, lo_abstr_compgraph,
                                 low_node, "bert", **opts)

        del probe_data
        del lo_abstr_compgraph
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()