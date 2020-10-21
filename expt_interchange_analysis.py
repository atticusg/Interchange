from experiment import Experiment

from train import load_model
from modeling.lstm import LSTMModule
from analysis import Analysis
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph
from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph

import json
import torch
import os
import argparse

from typing import Dict


class InterchangeAnalysis(Experiment):
    def experiment(self, opts: Dict) -> Dict:
        if "save_path" not in opts \
                or opts["save_path"] is None \
                or not os.path.exists(opts["save_path"]):
            raise ValueError("Cannot find saved data for expt {opts['id']}")

        module, _ = load_model(LSTMModule, opts["model_path"],
                               device=torch.device("cpu"))
        module.eval()
        data = torch.load(opts["data_path"], map_location=torch.device("cpu"))

        abstraction = json.loads(opts["abstraction"])

        high_intermediate_node = abstraction[0]
        low_intermediate_nodes = abstraction[1]

        base_compgraph = MQNLI_LSTM_CompGraph(module)
        low_model = Abstr_MQNLI_LSTM_CompGraph(base_compgraph,
                                               low_intermediate_nodes)
        high_model = MQNLI_Logic_CompGraph(data, [high_intermediate_node])

        a = Analysis(opts["save_path"], opts["abstraction"], high_model,
                     low_model)
        return a.analyze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--abstraction", type=str)
    parser.add_argument("--num_inputs", type=int)

    parser.add_argument("--id", type=int)
    parser.add_argument("--db_path", type=str)

    args = parser.parse_args()

    e = InterchangeAnalysis(finished_status=2)
    e.run(vars(args))

if __name__ == "__main__":
    main()