import os
import argparse
import pickle
import json
from datetime import datetime
import time

import torch
from torch.utils.data import DataLoader

import intervention
import interchange_manager
import datasets.mqnli
import modeling
from trainer import load_model
import experiment
from interchange_analysis import Analysis

import compgraphs
from compgraphs import mqnli_logic as logic
from compgraphs.mqnli_logic import Abstr_MQNLI_Logic_CompGraph
from modeling.utils import get_target_locs

from typing import List, Dict

SAMPLE_RES_DICT = {
    'runtime': 0.,
    'save_path': "",
}



class InterchangeExperiment(experiment.Experiment):
    def experiment(self, opts: Dict) -> Dict:
        res, duration, high_model, low_model, hi2lo_dict = self.run_interchanges(opts)
        save_path = self.save_interchange_results(opts, res)
        res_dict = {"runtime": duration, "save_path": save_path}
        analysis_res = self.analyze_interchanges(res, high_model, low_model,
                                                 hi2lo_dict, opts)
        res_dict.update(analysis_res)
        print("final res_dict", res_dict)
        return res_dict

    def run_interchanges(self, opts: Dict):
        # load low level model
        model_type = opts.get("model_type", "lstm")
        model_class = modeling.get_module_class_by_name(opts.get("model_type", "lstm"))
        device = torch.device("cuda")
        module, _ = load_model(model_class, opts["model_path"], device=device)
        module.eval()

        # load data
        data = torch.load(opts["data_path"])

        # get intervention information based on model type and data variant
        abstraction = json.loads(opts["abstraction"])
        high_intermediate_node = abstraction[0]
        low_intermediate_nodes = abstraction[1]
        high_intermediate_nodes = [high_intermediate_node]

        data_variant = datasets.mqnli.get_data_variant(data)
        print("data_variant", data_variant)
        loc_mapping_type = opts.get("loc_mapping_type", "")
        loc_mapping_type = loc_mapping_type if loc_mapping_type else data_variant
        interv_info = {
            "target_locs": get_target_locs(high_intermediate_node, loc_mapping_type)
        }
        print(f"high node: {high_intermediate_node}, low_locs: {interv_info['target_locs']}")

        # set up models
        low_compgraph_class = compgraphs.get_compgraph_class_by_name(model_type)
        low_abstr_compgraph_class = compgraphs.get_abstr_compgraph_class_by_name(model_type)
        low_base_compgraph = low_compgraph_class(module)
        low_model = low_abstr_compgraph_class(low_base_compgraph,
                                              low_intermediate_nodes,
                                              interv_info=interv_info,
                                              root_output_device=torch.device("cpu"))
        low_model.set_cache_device(torch.device("cpu"))
        high_model = Abstr_MQNLI_Logic_CompGraph(data, high_intermediate_nodes)


        # set up to get examples from dataset
        if opts.get("interchange_batch_size", 0):
            # ------ Batched interchange experiment ------ #
            start_time = time.time()
            batch_results = intervention.find_abstractions_batch(
                low_model=low_model,
                high_model=high_model,
                low_model_type=model_type,
                dataset=data.dev,
                num_inputs=opts["num_inputs"],
                batch_size=opts["interchange_batch_size"],
                fixed_assignments={x: {x: intervention.LOC[:]} for x in
                                   ["root", "input"]},
                unwanted_low_nodes={"root", "input"}
            )
            duration = time.time() - start_time
            print(f"Interchange experiment took {duration:.2f}s")
            return batch_results, duration, high_model, low_model, None
        else:
            # ------ Original interchange experiment ------ #
            low_inputs, high_interventions, hi2lo_dict = \
                self.prepare_inputs(data=data,
                                    num_inputs=opts["num_inputs"],
                                    model_type=model_type,
                                    high_intermediate_node=high_intermediate_node,
                                    device=device)

            # setup for find_abstractions
            fixed_assignments = {x: {x: intervention.LOC[:]} for x in ["root", "input"]}
            if model_type == "lstm":
                input_mapping = lambda x: x  # identity function
            elif model_type == "bert":
                input_mapping = lambda x: hi2lo_dict[intervention.utils.serialize(x)]
            unwanted_nodes = {"root", "input"}

            # find abstractions
            start_time = time.time()
            with torch.no_grad():
                print("Finding abstractions")
                res = intervention.find_abstractions_torch(
                    low_model=low_model,
                    high_model=high_model,
                    high_inputs=low_inputs,
                    total_high_interventions=high_interventions,
                    fixed_assignments=fixed_assignments,
                    input_mapping=input_mapping,
                    unwanted_low_nodes=unwanted_nodes
                )

            duration = time.time() - start_time
            print(f"Finished finding abstractions, took {duration:.2f} s")
            return res, duration, high_model, low_model, hi2lo_dict

    def prepare_inputs(self, data, num_inputs, model_type, high_intermediate_node, device):
        # set up to get examples from dataset
        hi2lo_dict = {}
        if model_type == "lstm":
            collate_fn = lambda batch: datasets.utils.lstm_collate(batch,
                                                                   batch_first=False)
            dataloader = DataLoader(data.dev, batch_size=1, shuffle=False,
                                    collate_fn=collate_fn)
        elif model_type == "bert":
            dataloader = DataLoader(data.dev, batch_size=1, shuffle=False)

        # get examples from dataset
        low_inputs = []
        high_interventions = []
        print("Preparing inputs")
        for i, input_tuple in enumerate(dataloader):
            if i == num_inputs: break
            orig_input = input_tuple[-2]
            input_tuple = [x.to(device) for x in input_tuple]
            if model_type == "bert":
                orig_input = orig_input.T
                hi2lo_dict[intervention.utils.serialize(orig_input)] = input_tuple

            low_inputs.append(
                intervention.Intervention({"input": orig_input}, {}))

            high_interventions += self.get_interventions(high_intermediate_node,
                                                         orig_input)
        return low_inputs, high_interventions, hi2lo_dict

    def get_interventions(self, high_node: str, base_input: torch.Tensor) \
            -> List[intervention.Intervention]:
        if high_node in {"subj_adj", "v_adv", "obj_adj"}:
            intervention_values = logic.intersective_projections
        elif high_node in {"subj_noun", "v_verb", "obj_noun"}:
            intervention_values = torch.tensor([logic.INDEP, logic.EQUIV],
                                               dtype=torch.long)
        elif high_node in {"sentence_q", "vp_q"}:
            intervention_values = logic.quantifier_signatures
        elif high_node == "neg":
            intervention_values = logic.negation_signatures
        elif high_node in {"subj", "v_bar", "obj"}:
            intervention_values = torch.tensor(
                [logic.INDEP, logic.EQUIV, logic.ENTAIL, logic.REV_ENTAIL],
                dtype=torch.long
            )
        elif high_node in {"vp", "negp"}:
            intervention_values = torch.tensor(
                [logic.INDEP, logic.EQUIV, logic.ENTAIL, logic.REV_ENTAIL,
                 logic.CONTRADICT, logic.ALTER, logic.COVER],
                dtype=torch.long
            )
        else:
            raise ValueError(f"Invalid high-level node: {high_node}")

        res = [intervention.Intervention({"input": base_input},
                                         {high_node: value.unsqueeze(0)})
               for value in intervention_values]

        return res


    def save_interchange_results(self, opts: Dict, res: List) -> str:
        save_dir = opts.get("res_save_dir", "")

        if save_dir and opts["save_intermediate_results"]:
            abstraction = json.loads(opts["abstraction"])
            high_intermediate_node = abstraction[0]

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            time_str = datetime.now().strftime("%m%d-%H%M%S")
            res_file_name = f"res-id{opts['id']}-{high_intermediate_node}-{time_str}"

            if opts.get("interchange_batch_size", 0):
                res_file_name += ".pt"
                save_path = os.path.join(save_dir, res_file_name)
                torch.save(res, save_path)
            else:
                res_file_name += ".pkl"
                save_path = os.path.join(save_dir, res_file_name)
                with open(save_path, "wb") as f:
                    pickle.dump(res, f)
            return save_path
        else:
            return ""

    def analyze_interchanges(self, res: List, high_model, low_model, hi2lo_dict,
                             opts: Dict) -> Dict:
        a = Analysis(res, high_model, low_model, hi2lo_dict=hi2lo_dict, **opts)
        return a.analyze()

def main():
    parser = argparse.ArgumentParser()
    opts = experiment.parse_args(parser, interchange_manager.INTERCHANGE_DEFAULT_OPTS)
    e = InterchangeExperiment()
    e.run(opts)

if __name__ == "__main__":
    main()
