import os
import json
import torch
import argparse
import pickle

from experiment import Experiment

from intervention import Intervention, GraphInput
from intervention.utils import serialize

from train import load_model
from modeling.lstm import LSTMModule
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph
from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph


from typing import Dict

class Analysis:
    def __init__(self, graph_data, abstraction_str, high_model, low_model):
        if isinstance(graph_data, str):
            with open(graph_data, "rb") as f:
                graph_data = pickle.load(f)
        (experiments, realizations_to_inputs), mapping = graph_data[0]
        self.experiments = experiments
        self.realizations_to_inputs = realizations_to_inputs
        self.mapping = mapping
        abstraction = json.loads(abstraction_str)
        self.high_node = abstraction[0]
        self.low_node = abstraction[1][0]
        self.high_model = high_model
        self.low_model = low_model

    def get_original_input(self, low_interv: Intervention) -> GraphInput:
        interv_tensor = low_interv.intervention[self.low_node]
        k = (serialize(interv_tensor), self.high_node)
        return self.realizations_to_inputs[k].base

    def analyze(self):
        experiments = self.experiments


        #                      LOW      (base_eq)      HIGH
        #                  base_output     =?=     base_output
        #   (low_effect_eq)    =?=                     =?=   (high_effect_eq)
        #                 interv_output    =?=    interv_output
        #                              (interv_eq)

        #   hi_effect_eq?   -true--   C
        #        | false
        #    interv_eq?     --false-- B
        #        | true
        #        A

        interv_count = 0  # A + B + C
        effective_count = 0  # A + B
        interv_eq_count = 0  # A

        for k, v in experiments.items():
            low, high = k

            if len(low.intervention.values) > 0 and len(
                    high.intervention.values) > 0:
                interv_count += 1
                high_base_output, high_interv_output = self.high_model.intervene(
                    high)

                interv_eq = v.item()
                high_effect_eq = (high_base_output == high_interv_output).item()

                if not high_effect_eq:
                    effective_count += 1
                    if interv_eq: interv_eq_count += 1

        effective_ratio = effective_count / interv_count if interv_count != 0 else 0
        interv_eq_rate = interv_eq_count / effective_count if effective_count != 0 else 0

        # source_high_eq_rate = source_high_eq_count / interv_count if interv_count != 0 else 0

        print(f"Percentage of effective high intervs: {effective_count}/{interv_count}={effective_ratio * 100:.3f}%")
        print(f"interv_eq_rate: {interv_eq_count}/{effective_count}={interv_eq_rate * 100:.3f}%")
        return {
            "interv_count": interv_count,
            "effective_ratio": effective_ratio,
            "interv_eq_rate": interv_eq_rate
        }


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