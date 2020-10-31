import argparse
import torch
import json
import os
import pickle
import csv
from datetime import datetime

from intervention import GraphInput
from experiment import Experiment
from compgraphs import mqnli_logic as logic
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph

int_to_rln = ["indep", "equiv", "entail", "rev_entail", "contradict", "alter", "cover"]
positions = ["p_subj_q", "p_subj_adj", "p_subj_n", "p_neg", "p_adv", "p_v", "p_obj_q", "p_obj_adj", "p_obj_n",
             "h_subj_q", "h_subj_adj", "h_subj_n", "h_neg", "h_adv", "h_v", "h_obj_q", "h_obj_adj", "h_obj_n"]

class VisualizeCliques(Experiment):
    def experiment(self, opts):
        self.data = torch.load(opts["data_path"])
        mappings = json.loads(opts["mappings"])
        graph_save_paths = json.loads(opts["graph_save_paths"])
        abstraction = json.loads(opts["abstraction"])
        self.high_node = abstraction[0]
        self.low_node = abstraction[1][0]

        self.high_model = MQNLI_Logic_CompGraph(self.data, [self.high_node])

        visualize_save_paths = []
        for mapping, graph_save_path in zip(mappings, graph_save_paths):
            visualize_save_path = self.visualize(opts, mapping, graph_save_path)
            visualize_save_paths.append(visualize_save_path)

        print(visualize_save_paths)
        return {"visualize_save_paths", json.dumps(visualize_save_paths)}

    def get_str_from_input(self, x):
        serialized_tensor = x[0][1]
        input_tensor = torch.tensor(serialized_tensor).squeeze()
        input_toks = self.data.decode(input_tensor, return_str=False)
        return input_toks

    def visualize_value(self, value):
        return json.dumps([int_to_rln[x] for x in value])

    def visualize(self, opts, mapping, graph_res_path):
        print("mapping", mapping)
        print("graph_res_path", graph_res_path)
        viz_save_path = self.get_save_path(opts, mapping)

        with open(graph_res_path, "rb") as f:
            graph_res = pickle.load(f)

        graph = graph_res["graph"]
        causal_edges = graph_res["causal_edges"]
        input_to_id = graph_res["input_to_id"]
        cliques = graph_res["cliques"]
        id_to_input = {i: x for x, i in input_to_id.items()}
        id_to_graph_input = {i: GraphInput({"input": torch.tensor(x[0][1])}) for x, i in input_to_id.items()}

        cliques = sorted(cliques, reverse=True, key=lambda c: len(c))

        with open(viz_save_path, "w") as f:
            fieldnames = ["clique", "input", "high_node_value", "root_value"] + positions
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, c in enumerate(cliques):
                for node in c:
                    serialized_input = id_to_input[node]
                    graph_input = id_to_graph_input[node]
                    viz_toks = self.get_str_from_input(serialized_input)
                    root_value = self.high_model.compute(graph_input)
                    viz_root_value = self.visualize_value(root_value)
                    high_node_value = self.high_model.get_result(self.high_node, graph_input)
                    viz_high_node_value = self.visualize_value(high_node_value)
                    d = {"clique": i,
                        "high_node_value": viz_high_node_value,
                        "root_value": viz_root_value}
                    for pos, tok_str in zip(positions, viz_toks):
                        d[pos] = tok_str
                    
                    writer.writerow(d)

        print("Saved visualization data to", viz_save_path)
        return viz_save_path

    def get_save_path(self, opts, mapping):
        high_node = self.high_node
        low_node = self.low_node
        loc = mapping[high_node][low_node]
        time_str = datetime.now().strftime("%m%d-%H%M%S")
        if "id" in opts and opts["id"] is not None:
            res_file_name = f"clqviz-id{opts['id']}-{high_node}-{low_node}-loc{loc}-{time_str}.csv"
        else:
            res_file_name = f"clqviz-{high_node}-{low_node}-loc{loc}-{time_str}.csv"
        return os.path.join(opts["res_save_dir"], res_file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--graph_save_paths", required=True)
    parser.add_argument("--mappings", required=True)
    parser.add_argument("--abstraction", type=str)
    parser.add_argument("--res_save_dir", type=str)

    parser.add_argument("--id", type=int)
    parser.add_argument("--db_path", type=str)

    args = parser.parse_args()

    e = VisualizeCliques(finished_status=3)
    e.run(vars(args))

if __name__ == "__main__":
    main()