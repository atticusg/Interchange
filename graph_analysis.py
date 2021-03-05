import pickle
import json
import argparse
import torch

from collections import defaultdict
from experiment import Experiment
from modeling.utils import load_model
from modeling.lstm import LSTMModule

from intervention.utils import stringify_mapping
import causal_abstraction.clique_analysis as clq

from compgraphs.mqnli_logic import Abstr_MQNLI_Logic_CompGraph
from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph


class GraphExperiment(Experiment):
    def experiment(self, opts):
        res_dict = defaultdict(list)
        save_path = opts["save_path"]
        data = torch.load(save_path)
        graphs = []
        for i, data_dict in enumerate(data):
            mapping = data_dict["mapping"]
            print(f"--- Analyzing mapping ({i + 1}/{len(data)}) {stringify_mapping(mapping)}")
            one_expt_res, one_expt_graph = self.analyze_one_experiment(data_dict, opts)
            for key, value in one_expt_res.items():
                res_dict[key + 's'].append(value)
        graph_save_path = clq.save_graph_analysis(graphs, opts["graph_alpha"], opts["res_save_dir"])
        for key in res_dict.keys():
            str_list = json.dumps(res_dict[key])
            res_dict[key] = str_list
        res_dict["graph_save_path"] = graph_save_path
        res = dict(res_dict)
        print(res)
        return res

        # G, causal_edges, input_to_id, cliques = self.get_results(opts)
        # graph_save_path = save_graph_analysis(G, causal_edges, input_to_id, cliques,
        #                                       opts["graph_alpha"],
        #                                       opts["res_save_dir"],
        #                                       opts.get("id", None))
        # res_dict = {"graph_save_path": graph_save_path}
        # res_dict.update(analyze_graph_results(G, causal_edges, input_to_id, cliques))
        # return res_dict

    def analyze_one_experiment(self, data_dict, opts):
        G, causal_edges, input_to_id = clq.construct_graph_batch(data_dict)
        cliques = clq.find_cliques(G, causal_edges, opts["graph_alpha"])
        graph_dict = {
            "graph": G,
            "causal_edges": causal_edges,
            "input_to_id": input_to_id,
            "cliques": cliques
        }
        res_dict = clq.analyze_graph_results(G, causal_edges, input_to_id, cliques)
        return res_dict, graph_dict

    def get_results(self, opts):
        module, _ = load_model(LSTMModule, opts["model_path"],
                               device=torch.device("cpu"))
        module.eval()
        data = torch.load(opts["data_path"], map_location=torch.device('cpu'))
        abstraction = json.loads(opts["abstraction"])

        high_intermediate_node = abstraction[0]
        low_intermediate_nodes = abstraction[1]
        high_intermediate_nodes = [high_intermediate_node]

        print("Loading low level model and data")
        base_compgraph = MQNLI_LSTM_CompGraph(module)
        low_model = Abstr_MQNLI_LSTM_CompGraph(base_compgraph,
                                               low_intermediate_nodes)
        high_model = Abstr_MQNLI_Logic_CompGraph(data, high_intermediate_nodes)

        with open(opts["save_path"], "rb") as f:
            graph_data = pickle.load(f)
            (result, realizations_to_inputs), mapping = graph_data[0]

        print("Mapping", mapping)
        print("Constructing graph")

        # print("Example of key in realizations_to_inputs", list(realizations_to_inputs.keys())[0])
        G, causal_edges, input_to_id = clq.construct_graph(
            low_model=low_model,
            high_model=high_model,
            mapping=mapping,
            result=result,
            realizations_to_inputs=realizations_to_inputs,
            high_node_name=high_intermediate_node,
            high_root_name="root"
        )

        print("Finding cliques")
        cliques = clq.find_cliques(G, causal_edges, opts["graph_alpha"])
        print(len(cliques))
        return G, causal_edges, input_to_id, cliques


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_alpha", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--res_save_dir", type=str, required=True)

    parser.add_argument("--id", type=int)
    parser.add_argument("--db_path", type=str)

    args = parser.parse_args()

    e = GraphExperiment(finished_status=2)
    e.run(vars(args))


if __name__ == "__main__":
    main()