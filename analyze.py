import pickle
import torch

from intervention import Intervention
from intervention.utils import serialize

class Analysis:
    def __init__(self, pickle_file):
        with open(pickle_file, "rb") as f:
            res = pickle.load(f)
        (experiments, realizations_to_inputs), mapping = res[0]
        self.experiments = experiments
        self.realizations_to_inputs = realizations_to_inputs
        self.mapping = mapping

    def get_original_input(self, low_interv: Intervention,
                           low_node: str, high_node: str) -> Intervention:
        interv_tensor = low_interv.intervention[low_node]
        k = (serialize(interv_tensor), high_node)
        return self.realizations_to_inputs[k]

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


def main():
    pickle_file = "experiment_data/res-200-Oct11-212216-obj_adj.pkl"
    analysis = Analysis(pickle_file)
    analysis.analyze()

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


if __name__ == "__main__":
    main()