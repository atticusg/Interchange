import time
import torch
import pickle
import os
import json
from datetime import datetime
import argparse

from intervention.abstraction_torch import find_abstractions
from intervention.utils import serialize
from intervention import Intervention, LOC

from torch.utils.data import DataLoader
from datasets.utils import my_collate
from datasets.mqnli import get_data_variant

from modeling import get_module_class_by_name
from train import load_model

from compgraphs import get_compgraph_class_by_name
from compgraphs import get_abstr_compgraph_class_by_name
from compgraphs import mqnli_logic as logic
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph

from experiment import Experiment
from expt_interchange_analysis import Analysis

from typing import List, Dict, Optional

SAMPLE_RES_DICT = {
    'runtime': 0.,
    'save_path': "",
}

def get_target_locs(high_node_name: str, data_variant: str="lstm",
                    lstm_p_h_continuous: bool=True):
    if "lstm" in data_variant:
        # mapping for lstm model
        d = {"sentence_q": [logic.IDX_Q_S],
             "subj_adj": [logic.IDX_A_S],
             "subj_noun": [logic.IDX_N_S],
             "neg": [logic.IDX_NEG],
             "v_adv": [logic.IDX_ADV],
             "v_verb": [logic.IDX_V],
             "vp_q": [logic.IDX_Q_O],
             "obj_adj": [logic.IDX_A_O],
             "obj_noun": [logic.IDX_N_O],
             "obj": [logic.IDX_A_O, logic.IDX_N_O],
             "vp": [logic.IDX_Q_O],
             "v_bar": [logic.IDX_ADV, logic.IDX_V],
             "negp":[logic.IDX_NEG],
             "subj": [logic.IDX_A_S, logic.IDX_N_S]}

        for k, idxs in d.items():
            locs = []
            for idx in idxs:
                locs.append(LOC[idx])
                if lstm_p_h_continuous:
                    locs.append(LOC[10 + idx])
            d[k] = locs
        return d[high_node_name]

    if "bert" in data_variant:
        # mapping for bert model
        # [ <CLS> | not | every | bad | singer | does | not | badly | sings | <e> | every | good | song ]
        #  0        1     2       3     4        5      6     7       8       9     10      11     12

        d = {"sentence_q": [1, 2, 14, 15],
             "subj_adj": [3, 16],
             "subj_noun": [4, 17],
             "neg": [5, 6, 18, 19],
             "v_adv": [7, 20],
             "v_verb": [8, 21],
             "vp_q": [9, 10, 22, 23],
             "obj_adj": [11, 24],
             "obj_noun": [12, 25],
             "obj": [11, 12, 24, 25],
             "vp": [8, 9, 10, 21, 22, 23],
             "v_bar": [7, 8, 20, 21],
             "negp": [5, 6, 18, 19],
             "subj": [3, 4, 16, 17]}
        return d[high_node_name]

class InterchangeExperiment(Experiment):
    def experiment(self, opts: Dict) -> Dict:
        res, duration, high_model, low_model, hi2lo_dict = self.get_results(opts)
        # pickle file
        save_path = self.save_results(opts, res)
        res_dict = {"runtime": duration, "save_path": save_path}
        analysis_res = self.analyze_results(res, high_model, low_model,
                                            hi2lo_dict, opts)
        res_dict.update(analysis_res)
        print("res_dict", res_dict)
        return res_dict

    def get_results(self, opts: Dict):
        model_type = opts.get("model_type", "lstm")
        model_class = get_module_class_by_name(opts.get("model_type", "lstm"))
        device = torch.device("cuda")
        module, _ = load_model(model_class, opts["model_path"], device=device)
        module.eval()

        data = torch.load(opts["data_path"])
        data_variant = get_data_variant(data)

        abstraction = json.loads(opts["abstraction"])
        high_intermediate_node = abstraction[0]
        low_intermediate_nodes = abstraction[1]
        high_intermediate_nodes = [high_intermediate_node]

        p_h_continuous = False
        if model_type == "lstm":
            p_h_continuous = hasattr(module, "p_h_separator") \
                             and module.p_h_separator is not None

        interv_info = {
            "target_locs": get_target_locs(high_intermediate_node, data_variant,
                                           p_h_continuous)
        }

        low_compgraph_class = get_compgraph_class_by_name(model_type)
        low_abstr_compgraph_class = get_abstr_compgraph_class_by_name(model_type)
        low_base_compgraph = low_compgraph_class(module)
        low_model = low_abstr_compgraph_class(low_base_compgraph,
                                              low_intermediate_nodes,
                                              interv_info=interv_info,
                                              root_output_device=torch.device("cpu"))
        high_model = MQNLI_Logic_CompGraph(data, high_intermediate_nodes)

        hi2lo_dict = {}
        if model_type == "lstm":
            collate_fn = lambda batch: my_collate(batch, batch_first=False)
            dataloader = DataLoader(data.dev, batch_size=1, shuffle=False,
                                    collate_fn=collate_fn)
        elif model_type == "bert":
            dataloader = DataLoader(data.dev, batch_size=1, shuffle=False)

        low_inputs = []
        high_interventions = []
        num_inputs = opts["num_inputs"]
        print("Preparing inputs")
        for i, input_tuple in enumerate(dataloader):
            if i == num_inputs: break
            orig_input = input_tuple[-2]
            input_tuple = [x.to(device) for x in input_tuple]
            if model_type == "bert":
                orig_input = orig_input.T
                hi2lo_dict[serialize(orig_input)] = input_tuple

            low_inputs.append(Intervention({"input": orig_input}, {}))

            high_interventions += self.get_interventions(high_intermediate_node,
                                                         orig_input)

        fixed_assignments = {x: {x: LOC[:]} for x in ["root", "input"]}

        if model_type == "lstm":
            input_mapping = lambda x: x  # identity function
        elif model_type == "bert":
            input_mapping = lambda x: hi2lo_dict[serialize(x)]

        unwanted_nodes = {"root", "input"}

        start_time = time.time()
        with torch.no_grad():
            print("Finding abstractions")
            res = find_abstractions(
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

    def save_results(self, opts: Dict, res: List) -> str:
        abstraction = json.loads(opts["abstraction"])
        high_intermediate_node = abstraction[0]
        save_dir = opts["res_save_dir"]

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        time_str = datetime.now().strftime("%m%d-%H%M%S")
        res_file_name = f"res-id{opts['id']}-{high_intermediate_node}-{time_str}.pkl"
        save_path = os.path.join(save_dir, res_file_name)

        with open(save_path, "wb") as f:
            pickle.dump(res, f)

        return save_path

    def analyze_results(self, res: List, high_model, low_model, hi2lo_dict,
                        opts: Dict) -> Dict:
        a = Analysis(res, opts["abstraction"], high_model, low_model,
                     opts["graph_alpha"], opts["res_save_dir"],
                     low_model_type=opts["model_type"],
                     hi2lo_dict=hi2lo_dict,
                     expt_id=opts["id"])
        return a.analyze()
        # return {}

    def get_interventions(self, high_node: str, base_input: torch.Tensor) \
            -> List[Intervention]:
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

        res = [Intervention({"input": base_input},
                            {high_node: value.unsqueeze(0)})
               for value in intervention_values]

        return res




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--res_save_dir", required=True)
    parser.add_argument("--graph_alpha", type=int, default=100)
    parser.add_argument("--abstraction", type=str, default='["subj_adj",["bert_layer_0"]]')
    parser.add_argument("--num_inputs", type=int, default=50)
    parser.add_argument("--model_type", type=str, default="lstm")

    parser.add_argument("--id", type=int)
    parser.add_argument("--db_path", type=str)

    args = parser.parse_args()
    e = InterchangeExperiment()
    e.run(vars(args))

if __name__ == "__main__":
    main()
