import time
import torch
import pickle
import os
import json
from datetime import datetime
import argparse

from intervention.abstraction_torch import find_abstractions

from intervention import Intervention, LOC
from datasets.utils import my_collate
from modeling.lstm import LSTMModule
from train import load_model

from compgraphs import mqnli_logic
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph
from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph

from experiment import Experiment

from torch.utils.data import DataLoader



SAMPLE_RES_DICT = {
    'runtime': 0.,
    'save_path': "",
}

def get_loc_mapping(p_h_continuous=True):
    d = {"sentence_q": [mqnli_logic.IDX_Q_S],
         "subj_adj": [mqnli_logic.IDX_A_S],
         "subj_noun": [mqnli_logic.IDX_N_S],
         "neg": [mqnli_logic.IDX_NEG],
         "v_adv": [mqnli_logic.IDX_ADV],
         "v_verb": [mqnli_logic.IDX_V],
         "vp_q": [mqnli_logic.IDX_Q_O],
         "obj_adj": [mqnli_logic.IDX_A_O],
         "obj_noun": [mqnli_logic.IDX_N_O],
         "obj": [mqnli_logic.IDX_A_O, mqnli_logic.IDX_N_O],
         "vp": [mqnli_logic.IDX_Q_O],
         "v_bar": [mqnli_logic.IDX_ADV, mqnli_logic.IDX_V],
         "negp":[mqnli_logic.IDX_NEG],
         "subj": [mqnli_logic.IDX_A_S, mqnli_logic.IDX_N_S]}

    for k, idxs in d.items():
        locs = []
        for idx in idxs:
            locs.append(LOC[idx])
            if p_h_continuous:
                locs.append(LOC[10 + idx])
        d[k] = locs
    return d


class InterchangeExperiment(Experiment):
    def get_results(self, opts):
        module, _ = load_model(LSTMModule, opts["model_path"],
                               device=torch.device("cpu"))
        module.eval()
        data = torch.load(opts["data_path"], map_location=torch.device('cpu'))
        print("abstraction:", opts["abstraction"])
        abstraction = json.loads(opts["abstraction"])

        high_intermediate_node = abstraction[0]
        low_intermediate_nodes = abstraction[1]
        num_inputs = opts["num_inputs"]

        high_intermediate_nodes = [high_intermediate_node]

        p_h_continuous = hasattr(module, "p_h_separator") and module.p_h_separator is not None
        loc_mapping = get_loc_mapping(p_h_continuous)

        interv_info = {
            "target_locs": loc_mapping[high_intermediate_node]
        }

        base_compgraph = MQNLI_LSTM_CompGraph(module)
        low_model = Abstr_MQNLI_LSTM_CompGraph(base_compgraph,
                                               low_intermediate_nodes,
                                               interv_info=interv_info)
        high_model = MQNLI_Logic_CompGraph(data, high_intermediate_nodes)

        collate_fn = lambda batch: my_collate(batch, batch_first=False)
        dataloader = DataLoader(data.dev, batch_size=1, shuffle=False,
                                collate_fn=collate_fn)

        inputs = []
        high_interventions = []
        for i, input_tuple in enumerate(dataloader):
            if i == num_inputs: break

            input_value = input_tuple[0].to(module.device)
            print("input_value.shape", input_value.shape)
            base_input = Intervention({"input": input_value}, {})
            inputs.append(base_input)
            for proj in mqnli_logic.intersective_projections:
                intervention = Intervention({"input": input_value},
                                            {high_intermediate_node: proj})
                high_interventions.append(intervention)
        fixed_assignments = {x: {x: LOC[:]} for x in ["root", "input"]}
        input_mapping = lambda x: x  # identity function

        unwanted_nodes = {"root", "input"}

        start_time = time.time()
        with torch.no_grad():
            res = find_abstractions(
                low_model=low_model,
                high_model=high_model,
                high_inputs=inputs,
                total_high_interventions=high_interventions,
                fixed_assignments=fixed_assignments,
                input_mapping=input_mapping,
                unwanted_low_nodes=unwanted_nodes
            )
        duration = time.time() - start_time
        print(f"Finished finding abstractions, took {duration:.2f} s")
        return res, duration

    def save_results(self, opts, res):
        abstraction = json.loads(opts["abstraction"])
        high_intermediate_node = abstraction[0]
        num_inputs = opts["num_inputs"]
        save_dir = opts["res_save_dir"]
        time_str = datetime.now().strftime("%b%d-%H%M%S")
        res_file_name = f"res-{num_inputs}-{time_str}-{high_intermediate_node}.pkl"
        save_path = os.path.join(save_dir, res_file_name)

        with open(save_path, "wb") as f:
            pickle.dump(res, f)

        return save_path

    def analyze_results(self, res):
        return {}

    def experiment(self, opts):
        res, duration = self.get_results(opts)
        # pickle file
        save_path = self.save_results(opts, res)
        res_dict = {"runtime": duration, "save_path": save_path}
        analysis_res = self.analyze_results(res)
        res_dict.update(analysis_res)
        return res_dict



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--res_save_dir", required=True)
    parser.add_argument("--abstraction", type=str)
    parser.add_argument("--num_inputs", type=int, default=20)

    parser.add_argument("--id", type=int)
    parser.add_argument("--db_path", type=str)

    args = parser.parse_args()
    e = InterchangeExperiment()
    e.run(vars(args))

if __name__ == "__main__":
    main()
