import os
import json
import torch
import argparse
from tqdm import tqdm

from collections import defaultdict

from experiment import Experiment

from intervention import Intervention, GraphInput
from intervention.utils import serialize
from typing import Dict
import pickle

from modeling.utils import load_model
from modeling.lstm import LSTMModule
from compgraphs.mqnli_logic import Abstr_MQNLI_Logic_CompGraph
from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph


class ExploratoryAnalysis:
    def __init__(self, graph_data, abstraction_str, high_model, low_model):
        if isinstance(graph_data, str):
            print("loading data from pickle")
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

        #   hi_effect_eq?   -true--   E
        #        | false
        #    interv_eq?     --false-- D
        #        | true
        #   lo_effect_eq?   --true--  C
        #        | false
        #     base_eq?      --false-- B
        #        | true
        #        A

        interv_count = 0  # A + B + C + D + E = U + V + W + X + Y + Z
        effective_count = 0  # A + B + C + D = X + Y + Z + W
        interv_eq_count = 0  # A + B + C
        strict_success_count = 0  # A
        interv_in_source_count = 0
        interv_eq_in_source_count = 0 # X
        interv_ne_in_source_count = 0 # Z

        count = 0

        input_to_intervention_dict = defaultdict(set)

        for k, v in tqdm(experiments.items()):
            low, high = k

            if len(low.intervention.values) > 0 and len(
                    high.intervention.values) > 0:
                interv_count += 1
                # low_base_output, low_interv_output = self.low_model.intervene(low)
                high_base_output, high_interv_output = self.high_model.intervene(high)

                base_input_str = serialize(low.base["input"])

                source_tensor_input = self.get_original_input(low)["input"]
                source_input_str = serialize(source_tensor_input)
                source_graph_input = GraphInput({"input": source_tensor_input})
                input_to_intervention_dict[base_input_str].add(source_input_str)
                # source_low_output = self.low_model.compute(source_input)

                source_high_node = self.high_model.get_result(self.high_node, source_graph_input)
                interv_high_node = high.intervention[self.high_node]

                # res_interv_outputs_eq = v.item()

                interv_in_source = torch.all(source_high_node == interv_high_node).item()
                if interv_in_source: interv_in_source_count += 1

                # source_high_eq = (source_low_output == high_interv_output).item()
                # if source_high_eq: source_high_eq_count += 1

                # base_eq = (low_base_output == high_base_output).item()
                interv_eq = v.item()
                # interv_eq = (low_interv_output == high_interv_output).item()
                # low_effect_eq = (low_base_output == low_interv_output).item()
                high_effect_eq = (high_base_output == high_interv_output).item()

                if not high_effect_eq:
                    effective_count += 1
                    if interv_eq:
                        interv_eq_count += 1

            count += 1

        effective_ratio = effective_count / interv_count if interv_count != 0 else 0
        interv_eq_rate = interv_eq_count / effective_count if effective_count != 0 else 0
        strict_success_rate = strict_success_count / effective_count if effective_count != 0 else 0
        interv_in_source_rate = interv_in_source_count / interv_count if interv_count != 0 else 0
        # source_high_eq_rate = source_high_eq_count / interv_count if interv_count != 0 else 0
        unique_intervs = {k: len(v) for k, v in input_to_intervention_dict.items()}
        unique_interv_count = sum(v for v in unique_intervs.values())
        print(f"Num of different base inputs: {len(input_to_intervention_dict)}")
        # print(f"Num of diff interv for each input:\n{list(unique_intervs.values())}")
        print(f"Total num of unique intervs for each input = {unique_interv_count}/{interv_count}")
        print(f"Interv in source rate: {interv_in_source_count}/{interv_count}={interv_in_source_rate * 100:.3f}%")
        print(f"Percentage of effective high intervs: {effective_count}/{interv_count}={effective_ratio * 100:.3f}%")
        print(f"interv_eq_rate: {interv_eq_count}/{effective_count}={interv_eq_rate * 100:.3f}%")
        print(f"strict_success_rate: {strict_success_count}/{effective_count}={strict_success_rate * 100:.3f}%")
        # return {
        #     "interv_count": interv_count,
        #     "effective_ratio": effective_ratio,
        #     "interv_eq_rate": interv_eq_rate,
        #     "success_effective_rate": success_effective_rate,
        #     "in_source_given_eq_rate": in_source_given_eq_rate,
        #     "eq_given_in_source_rate": eq_given_in_source_rate
        # }
        return {}

        #
        # |---------------|-------------------------|----------|
        # |               |         base_eq         |          |
        # |               |-------------------------| !base_eq |
        # |               | interv_eq  | !interv_eq |          |
        # |---------------|------------|------------|----------|
        # | !effect_eq    | A          | B          | E        |
        # | effect_eq     | C          | D          |          |
        # |---------------|------------|------------|----------|


        # interv_count = 0             # A + B + C + D + E
        # correct_count = 0            # A + B + C + D
        # success_count = 0            # A + C
        # effective_count = 0          # A + B
        # success_effective_count = 0  # A
        #
        # count = 0
        # for k, v in experiments.items():
        #     low, high = k
        #
        #     if len(low.intervention.values) > 0 and len(high.intervention.values) > 0:
        #         interv_count += 1
        #         low_base_output, low_interv_output = self.low_model.intervene(low)
        #         high_base_output, high_interv_output = self.high_model.intervene(high)
        #
        #         res_interv_outputs_eq = v.item()
        #         base_eq = (low_base_output == high_base_output).item()
        #         interv_eq = (low_interv_output == high_interv_output).item()
        #         low_effect_eq = (low_base_output == low_interv_output).item()
        #         high_effect_eq = (low_base_output == low_interv_output).item()
        #
        #         if not base_eq: continue
        #
        #         assert res_interv_outputs_eq == interv_eq
        #
        #         correct_count += 1
        #         if interv_eq: success_count += 1
        #         if not low_effect_eq: effective_count += 1
        #         if (not low_effect_eq) and interv_eq: success_effective_count += 1
        #
        #     count += 1
        #
        # success_rate = success_count/correct_count if correct_count != 0 else 0
        # success_rate_in_effective = success_effective_count/effective_count if effective_count != 0 else 0
        # effective_rate = effective_count/correct_count if correct_count != 0 else 0
        # print(f"Base accuracy: {correct_count}/{interv_count}={correct_count/interv_count*100:.3f}%")
        # print(f"Success rate: {success_count}/{correct_count}={success_rate*100:.3f}%")
        # print(f"Effective rate: {effective_count}/{correct_count}={effective_rate*100:.3f}%")
        # print(f"Success rate in effective: {success_effective_count}/{effective_count}={success_rate_in_effective*100:.3f}%")
        #
        # return {
        #     "interv_count": interv_count,
        #     "correct_count": correct_count,
        #     "success_count": success_count,
        #     "effective_count": effective_count,
        #     "success_effective_count": success_effective_count,
        #     "effective_rate": effective_rate,
        #     "success_rate": success_rate,
        #     "success_rate_in_effective": success_rate_in_effective
        # }

        # print("check some examples in realizations_to_inputs")
        # for i, (k, v) in enumerate(realizations_to_inputs.items()):
        #     if i == 3: break
        #     # print(k)
        #     arr, name = k
        #     arr = torch.tensor(arr)
        #     print(f"\nName: {name} Arr: {arr.shape}")

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
        high_model = Abstr_MQNLI_Logic_CompGraph(data, [high_intermediate_node])

        a = ExploratoryAnalysis(opts["save_path"], opts["abstraction"], high_model,
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

    e = InterchangeAnalysis(finished_status=None)
    e.run(vars(args))

    """
    pickle_file = "../experiment_data/res-200-Oct11-212216-obj_adj.pkl"
    "experiment_data/res-200-Oct11-212216-obj_adj.pkl"
    found 40200 experiments
    found 200 entries in realizations_to_inputs
    found 30423 successful interchanges
    One example experiment:
    found 40000 non empty intervs
    found 30227 successful intervs
    success rate by type [0.548, 0.6872222222222222, 0.7320769230769231, 0.885]
    encountered types: {((0, 1),), ((0, 3),), ((0, 0),), ((0, 2),)}
    type counts: [1000, 14400, 13000, 11600], sum=40000
    """

"""
old analyze code

def analyze(self):
    experiments = self.experiments
    realizations_to_inputs = self.realizations_to_inputs

    print(f"found {len(experiments)} experiments")
    print(
        f"found {len(realizations_to_inputs)} entries in realizations_to_inputs")
    true_count = sum(1 if v else 0 for v in experiments.values())

    print(f"found {true_count} successful interchanges")
    print("One example experiment:")

    non_empty_interv_count = 0
    success_interv_count = 0

    count = 0

    total_count_by_type = [0] * 4
    success_count_by_type = [0] * 4

    encountered_types = set()

    for k, v in experiments.items():
        low, high = k
        low_int_vals = low.intervention.values
        high_int_vals = high.intervention.values

        if len(low_int_vals) > 0 or len(high_int_vals) > 0:
            non_empty_interv_count += 1
            encountered_types.add(serialize(high_int_vals['obj_adj']))

            interv_type = high_int_vals['obj_adj'][0, 1].item()
            total_count_by_type[interv_type] += 1
            if v:
                success_interv_count += 1
                success_count_by_type[interv_type] += 1
                count += 1

    print(f"found {non_empty_interv_count} non empty intervs")
    print(f"found {success_interv_count} successful intervs")
    success_rate_by_type = [c / total if total != 0 else 0 for c, total in
                            zip(success_count_by_type, total_count_by_type)]
    print(f"success rate by type {success_rate_by_type}")

    print(f"encountered types: {encountered_types}")
    print(
        f"type counts: {total_count_by_type}, sum={sum(total_count_by_type)}")

    count = 0
    for k, v in experiments.items():
        if count >= 3:  break
        low, high = k
        low_int_vals = low.intervention.values
        high_int_vals = high.intervention.values

        if len(low_int_vals) > 0 or len(high_int_vals) > 0:
            interv_input = self.get_original_input(low, "premise_lstm_0", "obj_adj")
            interv_input_tensor = interv_input.base['input']
            print(f"interv_input_Tensor shape {interv_input_tensor.shape}")
            count += 1

    print("check some examples in realizations_to_inputs")
    for i, (k, v) in enumerate(realizations_to_inputs.items()):
        if i == 3: break
        # print(k)
        arr, name = k
        arr = torch.tensor(arr)
        print(f"\nName: {name} Arr: {arr.shape}")



"""


if __name__ == "__main__":
    main()