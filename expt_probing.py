import torch
import argparse
import experiment
from trainer import load_model

import compgraphs
import modeling
from modeling.utils import get_target_loc_dict

from probing.modules import Probe
from probing.dataset import ProbingData
from probing.trainer import ProbeTrainer
from probing.utils import get_num_classes, get_low_nodes

DEFAULT_PROBING_OPTS = {
    "model_path": "",
    "data_path": "",
    "model_type": "",

    "is_control": False,
    "probe_max_rank": 24,
    "probe_dropout": 0.1,
    "probe_train_num_examples": 6400,
    "probe_train_num_dev_examples": 1600,
    "probe_correct_examples_only": True,

    "probe_train_batch_size": 128,
    "probe_train_eval_batch_size": 256,
    "probe_train_weight_norm": 0.,
    "probe_train_max_epochs": 40,
    "probe_train_lr": 0.001,
    "probe_train_lr_patience_epochs": 4,
    "probe_train_lr_anneal_factor": 0.5,
    "probe_save_dir": ""
}


class ProbingExperiment(experiment.Experiment):
    def experiment(self, opts):
        print("loading model")
        model_type = opts.get("model_type", "bert")
        model_class = modeling.get_module_class_by_name(model_type)
        module, _ = load_model(model_class, opts["model_path"])
        device = torch.device("cuda")
        module = module.to(device)
        module.eval()

        print("loading data")
        data = torch.load(opts["data_path"])

        print("constructing compgraph")
        lo_base_compgraph_class = compgraphs.get_compgraph_class_by_name(model_type)
        lo_base_compgraph = lo_base_compgraph_class(module)
        hi_compgraph = compgraphs.MQNLI_Logic_CompGraph(data)

        probes = {}
        probe_input_dim = 768 if opts["model_type"] == "bert" else 256  # TODO: possibly change this
        # do probing by low node
        for low_node in get_low_nodes(opts["model_type"]):
            lo_abstr_compgraph_class = compgraphs.get_abstr_compgraph_class_by_name(opts["model_name"])
            lo_abstr_compgraph = lo_abstr_compgraph_class(lo_base_compgraph)
            probe_data = ProbingData(data, hi_compgraph, lo_abstr_compgraph,
                                     low_node, **opts)
            loc_dict = get_target_loc_dict(opts["model_type"])
            for hi_node in loc_dict.keys():
                probe_output_classes = get_num_classes(hi_node)
                for low_loc in loc_dict[hi_node]:
                    probe = Probe(hi_node, low_node, low_loc, opts["is_control"],
                              probe_output_classes, probe_input_dim,
                              opts["probe_max_rank"], opts["probe_dropout"])
                    trainer = ProbeTrainer(probe_data, probe, device=device, **opts)
                    best_probe_checkpoint = trainer.train()
                    probes[(hi_node, low_node, low_loc)] = probe
                    dev_acc = best_probe_checkpoint["dev_acc"]
                    probe_save_path = best_probe_checkpoint["model_save_path"]

                    # TODO: store results into a pandas dataframe

    def init_probes(self, opts):
        loc_dict = get_target_loc_dict(opts["model_type"])
        probe_input_dim = 768 if opts["model_type"] == "bert" else 256 # TODO: possibly change this
        probes = {}
        for hi_node in loc_dict.keys():
            for low_loc in loc_dict[hi_node]:
                for low_node in get_low_nodes(opts["model_type"]):
                    probe_output_classes = get_num_classes(hi_node)
                    p = Probe(hi_node, low_node, low_loc, opts["is_control"],
                              probe_output_classes, probe_input_dim,
                              opts["probe_max_rank"], opts["probe_dropout"])
                    probes[(hi_node, low_node, low_loc)] = p
        return probes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opts = experiment.parse_args(parser, DEFAULT_PROBING_OPTS)
    e = ProbingExperiment()
    e.run(opts)
