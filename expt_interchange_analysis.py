import os
import json
import torch
import argparse
import pickle

from collections import defaultdict

from experiment import Experiment
from expt_graph import analyze_results, save_results

from intervention import Intervention, GraphInput
from intervention.analysis import construct_graph, find_cliques
from intervention.utils import serialize
from intervention.location import Location

from train import load_model
from modeling.lstm import LSTMModule
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph
from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph


from typing import Dict

def stringify_mapping(m):
    res = {}
    for high, low in m.items():
        low_dict = {}
        for low_node, low_loc in low.items():
            if isinstance(low_loc, slice):
                str_low_loc = Location.slice_to_str(low_loc)
            else:
                str_low_loc = str(low_loc)
            low_dict[low_node] = str_low_loc
        res[high] = low_dict
    return res

class Analysis:
    def __init__(self, data, abstraction_str, high_model, low_model,
                 graph_alpha, res_save_dir, expt_id=None):
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = pickle.load(f)
        print(len(data))
        self.data = data
        abstraction = json.loads(abstraction_str)
        self.high_node = abstraction[0]
        self.low_node = abstraction[1][0]
        self.high_model = high_model
        self.low_model = low_model
        self.graph_alpha = graph_alpha
        self.res_save_dir = res_save_dir
        self.expt_id = expt_id

    # def get_original_input(self, low_interv: Intervention) -> GraphInput:
    #     interv_tensor = low_interv.intervention[self.low_node]
    #     k = (serialize(interv_tensor), self.high_node)
    #     return self.realizations_to_inputs[k].base

    def analyze(self):
        res_dict = defaultdict(list)
        for (results, realizations_to_inputs), mapping in self.data:
            res = self.analyze_one_experiment(results, realizations_to_inputs, mapping)
            for key, value in res.items():
                res_dict[key+'s'].append(value)

        for key in res_dict.keys():
            str_list = json.dumps(res_dict[key])
            res_dict[key] = str_list
        return dict(res_dict)

    def analyze_one_experiment(self, results, realizations_to_inputs, mapping):
        print("--Analyzing mapping", stringify_mapping(mapping))
        res_dict = {}
        res_dict.update(self.analyze_counts(results, mapping))
        res_dict.update(self.analyze_graph(results, realizations_to_inputs, mapping))
        return res_dict

    def analyze_counts(self, results, mapping):
        """
                             LOW      (base_eq)      HIGH
                         base_output     =?=     base_output
          (low_effect_eq)    =?=                     =?=   (high_effect_eq)
                        interv_output    =?=    interv_output
                                     (interv_eq)

          hi_effect_eq? --true-- D
               | false
           interv_eq? --false-- C
               | true
            base_eq? --false-- B
               | true
               A
        """
        interv_count = 0  # A + B + C + D
        effective_count = 0  # A + B + C
        interv_eq_count = 0  # A + B
        strict_success_count = 0 # A

        input_set = set(serialize(low.base["input"]) for low, _ in results.keys())

        low_base_outputs = self.get_base_outputs(input_set, self.low_model)

        for count, (k, v) in enumerate(results.items()):
            low, high = k
            if count % 10000 == 0:
                print(f"  processed {count}/{len(results)} examples")

            if len(low.intervention.values) == 0 or len(high.intervention.values) == 0:
                continue

            interv_count += 1
            high_base_output, high_interv_output = self.high_model.intervene(
                high)
            low_base_output = low_base_outputs[serialize(low.base["input"])]

            interv_eq = v.item()
            base_eq = (low_base_output == high_base_output.item())
            high_effect_eq = (high_base_output == high_interv_output).item()

            if not high_effect_eq:
                effective_count += 1
                if interv_eq:
                    interv_eq_count += 1
                    if base_eq:
                        strict_success_count += 1

        effective_ratio = effective_count / interv_count if interv_count != 0 else 0
        interv_eq_rate = interv_eq_count / effective_count if effective_count != 0 else 0
        strict_success_rate = strict_success_count / effective_count if effective_count != 0 else 0
        print(f"  Percentage of effective high intervs: {effective_count}/{interv_count}={effective_ratio * 100:.3f}%")
        print(f"  interv_eq_rate: {interv_eq_count}/{effective_count}={interv_eq_rate * 100:.3f}%")
        print(f"  strict_success_rate: {strict_success_count}/{effective_count}={strict_success_rate * 100:.3f}%")

        res_dict = {
            "interv_count": interv_count,
            "effective_ratio": effective_ratio,
            "interv_eq_rate": interv_eq_rate,
            "strict_success_rate": strict_success_rate,
            "mapping": stringify_mapping(mapping)
        }

        return res_dict

    def get_base_outputs(self, low_inputs, model):
        low_inputs = list(low_inputs)
        batch_size = 32
        low_base_outputs = {}
        for i in range(0, len(low_inputs), batch_size):
            batch = low_inputs[i:i+batch_size]
            tensor_batch = torch.cat(tuple(torch.tensor(t) for t in low_inputs[i:i+batch_size]), dim=1)
            graph_input = GraphInput({"input": tensor_batch})
            output = model.compute(graph_input, store_cache=False)
            for i, o in zip(batch, output):
                low_base_outputs[i] = o.item()
        return low_base_outputs

    def analyze_graph(self, result, realizations_to_inputs, mapping):
        print("  Constructing graph")
        G, causal_edges, input_to_id = construct_graph(
            low_model=self.low_model,
            high_model=self.high_model,
            mapping=mapping,
            result=result,
            realizations_to_inputs=realizations_to_inputs,
            high_node_name=self.high_node,
            high_root_name="root"
        )

        print("  Finding cliques")
        cliques = find_cliques(G, causal_edges, self.graph_alpha)
        graph_save_path = save_results(G, causal_edges, input_to_id, cliques,
                                       self.graph_alpha, self.res_save_dir,
                                       self.expt_id)
        res_dict = {"graph_save_path": graph_save_path}
        res_dict.update(analyze_results(G, causal_edges, input_to_id, cliques))
        return res_dict


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

        a = Analysis(opts["save_path"], opts["abstraction"], high_model, low_model,
                     opts["graph_alpha"], opts["res_save_dir"], opts.get("id", None))
        res_dict = a.analyze()
        print(res_dict)
        return res_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--abstraction", type=str)
    parser.add_argument("--num_inputs", type=int)
    parser.add_argument("--res_save_dir", type=str)
    parser.add_argument("--graph_alpha", type=int, default=100)

    parser.add_argument("--id", type=int)
    parser.add_argument("--db_path", type=str)

    args = parser.parse_args()

    e = InterchangeAnalysis(finished_status=2)
    e.run(vars(args))

if __name__ == "__main__":
    main()
