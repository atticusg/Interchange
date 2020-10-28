import os
import pickle
import json
import argparse
import torch

from datetime import datetime

from experiment import Experiment
from train import load_model
from modeling.lstm import LSTMModule

from intervention.analysis import construct_graph, find_cliques
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph
from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph


def analyze_results(G, causal_edges, input_to_id, cliques):
    if len(cliques) == 0:
        res_dict = {"max_clique_size": 0,
                    "avg_clique_size": 0,
                    "sum_clique_size": 0,
                    "clique_count": 0}
        return res_dict
    max_clique_size = max(len(c) for c in cliques)
    avg_clique_size = sum(len(c) for c in cliques) / len(cliques) if len(cliques) > 0 else 0
    num_nodes_in_cliques = sum(len(c) for c in cliques)
    res_dict = {"max_clique_size": max_clique_size,
                "avg_clique_size": avg_clique_size,
                "sum_clique_size": num_nodes_in_cliques,
                "clique_count": len(cliques)}
    return res_dict

def save_results(G, causal_edges, input_to_id, cliques, graph_alpha, res_save_dir, id=None):
    res = {
        "alpha": graph_alpha,
        "graph": G,
        "causal_edges": causal_edges,
        "input_to_id": input_to_id,
        "cliques": cliques
    }
    time_str = datetime.now().strftime("%m%d-%H%M%S")
    if id:
        res_file_name = f"graph-id{id}-{time_str}.pkl"
    else:
        res_file_name = f"graph-{time_str}.pkl"
    graph_save_path = os.path.join(res_save_dir, res_file_name)

    with open(graph_save_path, "wb") as f:
        pickle.dump(res, f)
    print("Saved graph analysis data to", graph_save_path)
    return graph_save_path

class GraphExperiment(Experiment):
    def experiment(self, opts):
        G, causal_edges, input_to_id, cliques = self.get_results(opts)
        graph_save_path = save_results(G, causal_edges, input_to_id, cliques,
                                       opts["graph_alpha"],
                                       opts["res_save_dir"],
                                       opts.get("id", None))
        res_dict = {"graph_save_path": graph_save_path}
        res_dict.update(analyze_results(G, causal_edges, input_to_id, cliques))
        return res_dict

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
        high_model = MQNLI_Logic_CompGraph(data, high_intermediate_nodes)

        with open(opts["save_path"], "rb") as f:
            graph_data = pickle.load(f)
            (result, realizations_to_inputs), mapping = graph_data[0]

        print("Mapping", mapping)
        print("Constructing graph")

        # print("Example of key in realizations_to_inputs", list(realizations_to_inputs.keys())[0])
        G, causal_edges, input_to_id = construct_graph(
            low_model=low_model,
            high_model=high_model,
            mapping=mapping,
            result=result,
            realizations_to_inputs=realizations_to_inputs,
            high_node_name=high_intermediate_node,
            high_root_name="root"
        )

        print("Finding cliques")
        cliques = find_cliques(G, causal_edges, opts["graph_alpha"])
        print(len(cliques))
        return G, causal_edges, input_to_id, cliques




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_alpha", type=int, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--res_save_dir", type=str, required=True)
    parser.add_argument("--abstraction", type=str)

    parser.add_argument("--id", type=int)
    parser.add_argument("--db_path", type=str)

    args = parser.parse_args()

    e = GraphExperiment(finished_status=4)
    e.run(vars(args))


if __name__ == "__main__":
    main()