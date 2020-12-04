import os
import json
import torch
import argparse
import pickle

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from experiment import Experiment
from expt_graph import analyze_graph_results, save_graph_analysis

from intervention import Intervention, GraphInput
from intervention.analysis import construct_graph, construct_graph_batch, find_cliques
from intervention.utils import serialize, deserialize, stringify_mapping

from trainer import load_model
from modeling.lstm import LSTMModule
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph
from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph

from typing import Dict

# for getting base results of bert model
class ListDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, item):
        return self.data[item]
    def __len__(self):
        return len(self.data)

class Analysis:
    def __init__(self, data, high_model, low_model, hi2lo_dict=None, opts=None):
        if isinstance(data, str):
            with open(data, "rb") as f:
                data = pickle.load(f)
        self.data = data
        abstraction = json.loads(opts["abstraction"])
        self.high_node = abstraction[0]
        self.low_node = abstraction[1][0]
        self.high_model = high_model
        self.low_model = low_model
        self.low_model_type = opts.get("model_type", "lstm")
        self.graph_alpha = opts.get("graph_alpha", 100)
        self.res_save_dir = opts.get("res_save_dir", "")
        self.batch_size = opts.get("interchange_batch_size", 0)
        self.low_model_device = getattr(low_model, "device", torch.device("cuda"))
        self.expt_id = opts.get("expt_id", 0)

        if not self.batch_size:
            self.serialize_lo = serialize if self.low_model_type == "lstm" else \
                lambda x : serialize(x[0].squeeze())
            self.hi2lo, self.lo2hi = self.setup_mappings(hi2lo_dict)

    # def get_original_input(self, low_interv: Intervention) -> GraphInput:
    #     interv_tensor = low_interv.intervention[self.low_node]
    #     k = (serialize(interv_tensor), self.high_node)
    #     return self.realizations_to_inputs[k].base

    def setup_mappings(self, hi2lo_dict):
        if self.low_model_type == "lstm":
            return lambda x: x, lambda x: x
        elif self.low_model_type == "bert":
            self._hi2lo_dict = hi2lo_dict
            hi2lo = lambda hi: self._hi2lo_dict[serialize(hi)]
            self._lo2hi_dict = {serialize(lo[0]): deserialize(hi)
                                for hi, lo in hi2lo_dict.items()}
            lo2hi = lambda lo: self._lo2hi_dict[self.serialize_lo(lo)]
            return hi2lo, lo2hi
        else:
            raise NotImplementedError(f"Does not support low model type {self.low_model_type}")


    def analyze(self):
        with torch.no_grad():
            res_dict = defaultdict(list)

            if self.batch_size:
                for i, data_dict in enumerate(self.data):
                    mapping = data_dict["mapping"]
                    print(f"--- Analyzing mapping ({i+1}/{len(self.data)}) {stringify_mapping(mapping)}")
                    res = self.analyze_one_experiment(data_dict, None, mapping)
                    for key, value in res.items():
                        res_dict[key+'s'].append(value)
            else:
                for i, ((results, realizations_to_inputs), mapping) in enumerate(self.data):
                    print(
                        f"--- Analyzing mapping ({i + 1}/{len(self.data)}) {stringify_mapping(mapping)}")
                    res = self.analyze_one_experiment(results, realizations_to_inputs, mapping)
                    for key, value in res.items():
                        res_dict[key+'s'].append(value)

            for key in res_dict.keys():
                str_list = json.dumps(res_dict[key])
                res_dict[key] = str_list
        return dict(res_dict)

    def analyze_one_experiment(self, results, realizations_to_inputs, mapping):
        res_dict = {}
        res_dict.update(self.analyze_counts(results, mapping))
        if self.graph_alpha > 0:
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

        if isinstance(results, dict):
            for base_i, interv_i, low_base_output, high_base_output, low_interv_output, high_interv_output \
                    in zip(results["base_i"], results["interv_i"],
                           results["low_base_res"], results["high_base_res"],
                           results["low_interv_res"], results["high_interv_res"]):
                interv_count += 1

                interv_eq = (low_interv_output == high_interv_output)
                base_eq = (low_base_output == high_base_output)
                high_effect_eq = (high_base_output == high_interv_output)

                if not high_effect_eq:
                    effective_count += 1
                    if interv_eq:
                        interv_eq_count += 1
                        if base_eq:
                            strict_success_count += 1
        else:
            low_base_outputs = self.get_low_base_outputs(results, self.low_model,
                                                         self.low_model_type)
            for i, (k, v) in enumerate(results.items()):
                low, high = k
                if i % 10000 == 0:
                    print(f"    processed {i}/{len(results)} examples")

                if len(low.intervention.values) == 0 or len(high.intervention.values) == 0:
                    continue

                interv_count += 1
                high_base_output, high_interv_output = self.high_model.intervene(
                    high)
                low_base_output = low_base_outputs[self.serialize_lo(low.base["input"])]

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
        print(f"    Percentage of effective high intervs: {effective_count}/{interv_count}={effective_ratio * 100:.3f}%")
        print(f"    interv_eq_rate: {interv_eq_count}/{effective_count}={interv_eq_rate * 100:.3f}%")
        print(f"    strict_success_rate: {strict_success_count}/{effective_count}={strict_success_rate * 100:.3f}%")

        res_dict = {
            "interv_count": interv_count,
            "effective_ratio": effective_ratio,
            "interv_eq_rate": interv_eq_rate,
            "strict_success_rate": strict_success_rate,
            "mapping": stringify_mapping(mapping)
        }

        return res_dict

    def get_low_base_outputs(self, results, model, model_type):
        low_base_outputs = {}
        if model_type == "lstm":
            low_inputs = list(set(serialize(low.base["input"]) for low, _ in results.keys()))
            batch_size = 32
            for i in range(0, len(low_inputs), batch_size):
                batch = low_inputs[i:i+batch_size]
                tensor_batch = torch.cat(tuple(torch.tensor(t) for t in low_inputs[i:i+batch_size]), dim=1)
                graph_input = GraphInput({"input": tensor_batch})
                output = model.compute(graph_input, store_cache=False)
                for i, o in zip(batch, output):
                    low_base_outputs[i] = o.item()

        elif model_type == "bert":
            low_inputs = [tuple(low_interv.base["input"]) for low_interv, _ in results.keys()]
            low_inputs_dataset = ListDataset(low_inputs)
            dataloader = DataLoader(low_inputs_dataset, batch_size=32, shuffle=False)

            for input_tuple in dataloader:
                input_tuple = [x.to(self.low_model_device).view(-1,x.shape[-1]) for x in input_tuple]
                graph_input = GraphInput({"input": input_tuple})
                output = model.compute(graph_input, store_cache=False)
                for i, o in zip(input_tuple[0], output):
                    low_base_outputs[serialize(i)] = o.item()

        return low_base_outputs

    def analyze_graph(self, result, realizations_to_inputs, mapping):
        print("    Constructing graph")
        if isinstance(result, dict) and realizations_to_inputs is None:
            G, causal_edges, input_to_id = construct_graph_batch(result)
        else:
            G, causal_edges, input_to_id = construct_graph(
                low_model=self.low_model,
                high_model=self.high_model,
                mapping=mapping,
                result=result,
                realizations_to_inputs=realizations_to_inputs,
                high_node_name=self.high_node,
                high_root_name="root",
                low_serialize_fxn=self.serialize_lo
            )

        print("    Finding cliques")
        cliques = find_cliques(G, causal_edges, self.graph_alpha)
        graph_save_path = save_graph_analysis(
            G, causal_edges, input_to_id, cliques,
            self.graph_alpha, self.res_save_dir, self.expt_id
        )
        res_dict = {"graph_save_path": graph_save_path}
        res_dict.update(analyze_graph_results(G, causal_edges, input_to_id, cliques))
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

        a = Analysis(opts["save_path"], high_model, low_model, opts=opts)
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

"""
res_dict = {'runtime': 5.930975675582886, 
            'save_path': 'experiment_data/bert/test/res-idNone-subj_adj-1110-123000.pkl', 
            'interv_counts': '[100, 100]', 
            'effective_ratios': '[0.13, 0.13]', 
            'interv_eq_rates': '[0.5384615384615384, 0.6923076923076923]', 
            'strict_success_rates': '[0.5384615384615384, 0.6923076923076923]', 
            'mappings': '[{"root": {"root": "::"}, "input": {"input": "::"}, "subj_adj": {"bert_layer_0": "(slice(None, None, None), 3, slice(None, None, None))"}}, {"root": {"root": "::"}, "input": {"input": "::"}, "subj_adj": {"bert_layer_0": "(slice(None, None, None), 16, slice(None, None, None))"}}]', 
            'graph_save_paths': '["experiment_data/bert/test/graph-1110-123004.pkl", "experiment_data/bert/test/graph-1110-123004.pkl"]', 
            'max_clique_sizes': '[1, 1]', 
            'avg_clique_sizes': '[1.0, 1.0]', 
            'sum_clique_sizes': '[1, 1]', 
            'clique_counts': '[1, 1]'}
"""

# mapping = [{"root": {"root": "::"},
#             "input": {"input": "::"},
#             "subj_adj": {"bert_layer_0": "(slice(None, None, None), 3, slice(None, None, None))"}},
#            {"root": {"root": "::"},
#             "input": {"input": "::"},
#             "subj_adj": {"bert_layer_0": "(slice(None, None, None), 16, slice(None, None, None))"}}]