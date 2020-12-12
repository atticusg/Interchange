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


DEFAULT_PROBING_OPTS = {
    "model_path": "",
    "data_path": "",
    "model_type": "",

    "is_control": False,
    "probe_max_rank": 24,
    "probe_dropout": 0.1,
    "probe_train_num_examples": 6400,
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
    "res_save_path": ""
}


class ProbingExperiment(experiment.Experiment):
    def experiment(self, opts):
        print("=== Loading model")
        model_type = opts.get("model_type", "bert")
        model_class = modeling.get_module_class_by_name(model_type)
        module, _ = load_model(model_class, opts["model_path"])
        device = torch.device("cuda")
        module = module.to(device)
        module.eval()

        print("=== Loading data")
        data = torch.load(opts["data_path"])

        print("=== Constructing compgraph")
        lo_base_compgraph_class = compgraphs.get_compgraph_class_by_name(model_type)
        lo_base_compgraph = lo_base_compgraph_class(module)
        lo_abstr_compgraph_class = compgraphs.get_abstr_compgraph_class_by_name(opts["model_type"])
        hi_compgraph = compgraphs.MQNLI_Logic_CompGraph(data)
        # hi_compgraph.set_cache_device(torch.device("cpu"))

        if not os.path.exists(opts["res_save_dir"]): os.mkdir(opts["res_save_dir"])
        time_str = datetime.now().strftime('%m%d-%H%M%S')
        res_save_path = os.path.join(opts["res_save_dir"], f"results-{time_str}.csv")
        csv_f = open(res_save_path, "w")
        fieldnames = ["high_node", "low_node", "low_loc", "dev_acc", "dev_loss", "save_path"]
        writer = csv.DictWriter(csv_f, fieldnames)
        writer.writeheader()

        probe_input_dim = probing.utils.get_low_hidden_dim(opts["model_type"], module)

        # do probing by low node
        for low_node in probing.utils.get_low_nodes(opts["model_type"]):
            print(f"\n=== Getting hidden vectors for low node {low_node}")
            lo_abstr_compgraph = lo_abstr_compgraph_class(lo_base_compgraph, [low_node])
            # lo_abstr_compgraph.set_cache_device(torch.device("cpu"))
            probe_data = ProbingData(data, hi_compgraph, lo_abstr_compgraph,
                                     low_node, **opts)
            loc_dict = get_target_loc_dict(opts["model_type"])
            print(f"=== Training probes")
            for high_node in loc_dict.keys():
                probe_output_classes = probing.utils.get_num_classes(high_node)
                for low_loc in loc_dict[high_node]:
                    probe = Probe(high_node, low_node, low_loc, opts["is_control"],
                              probe_output_classes, probe_input_dim,
                              opts["probe_max_rank"], opts["probe_dropout"])
                    trainer = ProbeTrainer(probe_data, probe, device=device, **opts)
                    best_probe_checkpoint = trainer.train()
                    dev_acc = best_probe_checkpoint["dev_acc"]
                    dev_loss = best_probe_checkpoint["dev_loss"]
                    save_path = best_probe_checkpoint["model_save_path"]
                    writer.writerow({"high_node": high_node,
                                     "low_node": low_node,
                                     "low_loc": low_loc,
                                     "dev_acc": dev_acc,
                                     "dev_loss": dev_loss,
                                     "save_path": save_path})
                break # for testing purposes
            break # for testing

        csv_f.close()
        return {
            "res_save_path": res_save_path
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opts = experiment.parse_args(parser, DEFAULT_PROBING_OPTS)
    e = ProbingExperiment()
    e.run(opts)
