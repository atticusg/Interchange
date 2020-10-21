import pickle
import torch
import json

from intervention import Intervention
from intervention.utils import serialize

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

    def get_original_input(self, low_interv: Intervention,
                           low_node: str, high_node: str) -> Intervention:
        interv_tensor = low_interv.intervention[low_node]
        k = (serialize(interv_tensor), high_node)
        return self.realizations_to_inputs[k]

    def analyze(self):
        experiments = self.experiments


        #                      LOW      (base_eq)      HIGH
        #                  base_output     =?=     base_output
        #   (low_effect_eq)    =?=                     =?=   (high_effect_eq)
        #                 interv_output    =?=    interv_output
        #                              (interv_eq)
        #
        # |---------------|-------------------------|----------|
        # |               |         base_eq         |          |
        # |               |-------------------------| !base_eq |
        # |               | interv_eq  | !interv_eq |          |
        # |---------------|------------|------------|----------|
        # | !effect_eq    | A          | B          | E        |
        # | effect_eq     | C          | D          |          |
        # |---------------|------------|------------|----------|


        interv_count = 0             # A + B + C + D + E
        correct_count = 0            # A + B + C + D
        success_count = 0            # A + C
        effective_count = 0          # A + B
        success_effective_count = 0  # A

        count = 0
        for k, v in experiments.items():
            low, high = k

            if len(low.intervention.values) > 0 and len(high.intervention.values) > 0:
                interv_count += 1
                low_base_output, low_interv_output = self.low_model.intervene(low)
                high_base_output, high_interv_output = self.high_model.intervene(high)

                res_interv_outputs_eq = v.item()
                base_eq = (low_base_output == high_base_output).item()
                interv_eq = (low_interv_output == high_interv_output).item()
                low_effect_eq = (low_base_output == low_interv_output).item()
                high_effect_eq = (low_base_output == low_interv_output).item()

                if not base_eq: continue

                assert res_interv_outputs_eq == interv_eq
                assert low_effect_eq == high_effect_eq

                correct_count += 1
                if interv_eq: success_count += 1
                if not low_effect_eq: effective_count += 1
                if (not low_effect_eq) and interv_eq: success_effective_count += 1

            count += 1

        success_rate = success_count/correct_count if correct_count != 0 else 0
        success_rate_in_effective = success_effective_count/effective_count if effective_count != 0 else 0
        effective_rate = effective_count/correct_count if correct_count != 0 else 0
        print(f"Base accuracy: {correct_count}/{interv_count}={correct_count/interv_count*100:.3f}%")
        print(f"Success rate: {success_count}/{correct_count}={success_rate*100:.3f}%")
        print(f"Effective rate: {effective_count}/{correct_count}={effective_rate*100:.3f}%")
        print(f"Success rate in effective: {success_effective_count}/{effective_count}={success_rate_in_effective*100:.3f}%")

        return {
            "interv_count": interv_count,
            "correct_count": correct_count,
            "success_count": success_count,
            "effective_count": effective_count,
            "success_effective_count": success_effective_count,
            "effective_rate": effective_rate,
            "success_rate": success_rate,
            "success_rate_in_effective": success_rate_in_effective
        }

        # print("check some examples in realizations_to_inputs")
        # for i, (k, v) in enumerate(realizations_to_inputs.items()):
        #     if i == 3: break
        #     # print(k)
        #     arr, name = k
        #     arr = torch.tensor(arr)
        #     print(f"\nName: {name} Arr: {arr.shape}")



def main():
    pickle_file = "../experiment_data/res-200-Oct11-212216-obj_adj.pkl"
    # analysis = Analysis(pickle_file)
    # analysis.analyze()

    """
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