import time
import torch
import pickle
import os
import json
from datetime import datetime
import argparse

from intervention.abstraction_torch import find_abstractions

from intervention import Intervention, LOC
from datasets.mqnli import MQNLIData
from datasets.utils import my_collate
from modeling.lstm import LSTMModule
from train import load_model

from compgraphs import mqnli_logic
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph
from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph

from torch.utils.data import DataLoader
from grid_search import GridSearch

SAMPLE_RES_DICT = {
    'runtime': 0.,
    'save_path': "",
}

LOC_MAPPING = {
    "obj_adj": LOC[mqnli_logic.IDX_A_O],
    "subj_adj": LOC[mqnli_logic.IDX_A_S],
    "v_adv": LOC[mqnli_logic.IDX_ADV]
}

class InterchangeExperiment:
    def __init__(self, db_path):
        pass

    def get_results(self, opts):
        module, _ = load_model(LSTMModule, opts["model_path"],
                               device=torch.device("cpu"))
        module.eval()
        data = torch.load(opts["data_path"], map_location=torch.device('cpu'))

        abstraction = json.loads(opts["abstraction"])
        high_intermediate_node = abstraction[0]
        low_intermediate_nodes = abstraction[1]
        num_inputs = opts["num_inputs"]

        high_intermediate_nodes = [high_intermediate_node]

        interv_info = {
            "target_loc": LOC_MAPPING[high_intermediate_node]
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
        abstraction = opts["abstraction"]
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
        res_dict.update(res_dict)
        return res_dict



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--db_path", required=True)
    parser.add_argument("--res_save_dir", required=True)
    parser.add_argument("--abstraction", type=str)
    parser.add_argument("--num_inputs", type=int, nargs="+")

    # data_path = "mqnli_data/mqnli.pt"
    # data = MQNLIData("mqnli_data/mqnli.train.txt",
    #                  "mqnli_data/mqnli.dev.txt",
    #                  "mqnli_data/mqnli.test.txt")
    # torch.save(data, data_path)
    # model_path = "mqnli_models/lstm_best.pt"
    # db_path = "experiment_data/runtime.db"

    args = parser.parse_args()

    base_opts = {"abstraction": [],
                 "num_inputs": 100,
                 "res_save_dir": args.res_save_dir}

    grid_dict = {}
    if args.num_inputs:
        grid_dict["num_inputs"] = args.num_inputs



    gs = AbstractionExperiment(args.model_path, args.data_path, base_opts, args.db_path)
    gs.execute(grid_dict)


if __name__ == "__main__":
    main()
