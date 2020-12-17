import os
import torch
import argparse
import experiment
import csv
import time
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

        if not os.path.exists(opts["res_save_dir"]): os.mkdir(opts["res_save_dir"])
        time_str = datetime.now().strftime('%m%d-%H%M%S')
        res_save_path = os.path.join(opts["res_save_dir"], f"results-{time_str}.csv")
        csv_f = open(res_save_path, "w")
        fieldnames = ["high_node", "low_node", "low_loc", "is_control", "train_acc", "dev_acc", "dev_loss", "save_path"]
        writer = csv.DictWriter(csv_f, fieldnames)
        writer.writeheader()

        probe_input_dim = probing.utils.get_low_hidden_dim(opts["model_type"], module)

        # do probing by low node
        for low_node in probing.utils.get_low_nodes(opts["model_type"]):
            start_time = time.time()
            print(f"\n=== Getting hidden vectors for low node {low_node}")
            lo_abstr_compgraph = lo_abstr_compgraph_class(lo_base_compgraph, [low_node])
            # lo_abstr_compgraph.set_cache_device(torch.device("cpu"))
            probe_data = ProbingData(data, hi_compgraph, lo_abstr_compgraph,
                                     low_node, opts["model_type"], **opts)
            loc_dict = get_target_loc_dict(opts["model_type"])
            print(f"=== Training probes")
            for high_node in loc_dict.keys():
                probe_output_classes = probing.utils.get_num_classes(high_node)
                for low_loc in loc_dict[high_node]:
                    for is_control in [False, True]:
                        probe = Probe(
                            high_node, low_node, low_loc, is_control,
                            probe_output_classes, probe_input_dim,
                            opts["probe_max_rank"], opts["probe_dropout"]
                        )
                        trainer = ProbeTrainer(
                            probe_data, probe, device=device, **opts
                        )
                        best_probe_checkpoint = trainer.train()
                        train_acc = best_probe_checkpoint["train_acc"]
                        dev_acc = best_probe_checkpoint["dev_acc"]
                        dev_loss = best_probe_checkpoint["dev_loss"]
                        save_path = best_probe_checkpoint["model_save_path"]
                        writer.writerow({"high_node": high_node,
                                         "low_node": low_node,
                                         "low_loc": low_loc,
                                         "is_control": is_control,
                                         "train_acc": train_acc,
                                         "dev_acc": dev_acc,
                                         "dev_loss": dev_loss,
                                         "save_path": save_path})
                        del probe
                        del trainer
                    # break  # for testing
            low_node_duration = time.time() - start_time
            print(f"=== Probing {low_node} took {low_node_duration:.1f}s")
            lo_abstr_compgraph.clear_caches() # important!
            del probe_data
            del lo_abstr_compgraph
            torch.cuda.empty_cache()
            # break  # for testing

        csv_f.close()
        return {
            "res_save_path": res_save_path
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    opts = experiment.parse_args(parser, DEFAULT_PROBING_OPTS)
    e = ProbingExperiment()
    e.run(opts)
