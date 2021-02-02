import os
import pickle
from datetime import datetime

import networkx as nx
import copy
import numpy as np
from intervention.utils import serialize

import torch


def get_input(intervention, serialize_fxn=serialize):
    return tuple(sorted((k, serialize_fxn(intervention.base.values[k])) for k in intervention.base.values))


def construct_graph(low_model, high_model, mapping, result, realizations_to_inputs,
                    high_node_name, high_root_name, low_serialize_fxn=serialize):
    G = nx.Graph()
    input_to_id = dict()
    id_to_input = dict()
    inputs_seen = set()
    edges = set()
    total_id = 0
    causal_edges = set()
    count = 0
    for interventions in result:
        low_intervention, high_intervention = interventions

        if len(low_intervention.intervention.values) == 0 or len(
                high_intervention.intervention.values) == 0:
            continue
        count += 1
        input = get_input(low_intervention, low_serialize_fxn)
        if input not in inputs_seen:
            G.add_node(total_id)
            input_to_id[input] = total_id
            id_to_input[total_id] = input
            total_id +=1
            inputs_seen.add(input)
        low_node = None
        for key in mapping[high_node_name]:
            low_node = key
        # index = mapping[high_node_name][low_node]
        # string_array = serialize(low_model.get_result(low_node,low_intervention)[index])
        # print("low intervention", low_intervention.intervention)
        string_array = serialize(low_intervention.intervention[low_node])
        input2 = get_input(realizations_to_inputs[(string_array, high_node_name)], low_serialize_fxn)
        if input2 not in inputs_seen:
            G.add_node(total_id)
            input_to_id[input2] = total_id
            id_to_input[total_id] = input2
            total_id +=1
            inputs_seen.add(input2)
        if result[interventions]:
            edges.add((input_to_id[input], input_to_id[input2]))
            if high_model.get_result(high_root_name,high_intervention)[0] != high_model.get_result(high_root_name,high_intervention.base)[0]:
                causal_edges.add((input_to_id[input], input_to_id[input2]))
    new_causal_edges = set()
    for node in range(total_id):
        G.add_edge(node, node)
        for node2 in range(total_id):
            if (node, node2) in edges and (node2,node) in edges:
                G.add_edge(node, node2)
            if (node, node2) in causal_edges and (node2,node) in causal_edges:
                new_causal_edges.add((node,node2))
    return G, new_causal_edges, input_to_id

def construct_graph_batch(data_dict):
    G = nx.Graph()
    inputs_seen = set()
    edges = set()
    total_id = data_dict["interchange_dataset"].num_examples
    causal_edges = set()
    count = 0

    for base_i, interv_i, low_base_output, high_base_output, low_interv_output, high_interv_output \
                    in zip(data_dict["base_i"], data_dict["interv_i"],
                           data_dict["low_base_res"], data_dict["high_base_res"],
                           data_dict["low_interv_res"], data_dict["high_interv_res"]):
        count += 1
        if base_i not in inputs_seen:
            G.add_node(base_i)
            inputs_seen.add(base_i)

        if interv_i not in inputs_seen:
            G.add_node(interv_i)
            inputs_seen.add(interv_i)

        if low_interv_output == high_interv_output:
            edges.add((base_i, interv_i))
            if high_interv_output != high_base_output:
                causal_edges.add((base_i, interv_i))

    new_causal_edges = set()
    for node in range(total_id):
        G.add_edge(node, node)
        for node2 in range(total_id):
            if (node, node2) in edges and (node2,node) in edges:
                G.add_edge(node, node2)
            if (node, node2) in causal_edges and (node2,node) in causal_edges:
                new_causal_edges.add((node,node2))
    return G, new_causal_edges, None

def find_cliques(G, causal_edges, alpha):
    original_G = G
    cliques = []
    while True:
        G = copy.deepcopy(original_G)
        if len(G.nodes()) ==0:
            break
        while float(len(G.nodes())* (len(G.nodes())+1)*0.5) != float(len(G.edges())):
            edge_dict = {node:set() for node in G.nodes()}
            causal_edge_dict = {node:0 for node in G.nodes()}
            for edge in G.edges():
                edge_dict[edge[0]].add(edge[1])
                edge_dict[edge[1]].add(edge[0])
            for edge in causal_edges:
                if G.has_edge(edge[0], edge[1]) or G.has_edge(edge[1], edge[0]):
                    causal_edge_dict[edge[1]] +=1
                    causal_edge_dict[edge[0]] +=1
            edge_counts = [(k,v) for k, v in sorted(edge_dict.items(), key=lambda item: len(item[1]))]
            causal_edge_counts = [(k, v) for k, v in sorted(causal_edge_dict.items(), key=lambda item: item[1])]
            if causal_edge_counts[-1][1] - causal_edge_counts[0][1]>= alpha:
                G.remove_node(causal_edge_counts[0][0])
            else:
                G.remove_node(edge_counts[0][0])
        new_clique = set()
        for node in G.nodes():
            new_clique.add(node)
            original_G.remove_node(node)
        cliques.append(new_clique)
    final_result = []
    for clique in cliques:
        seen = False
        for node in copy.copy(clique):
            for node2 in clique:
                if ((node, node2) in causal_edges or (node2, node) in causal_edges) and not seen:
                    final_result.append(clique)
                    seen = True
    return final_result


def analyze_graph_results(G, causal_edges, input_to_id, cliques):
    if len(cliques) == 0:
        res_dict = {"max_clique_size": 0,
                    "avg_clique_size": 0,
                    "sum_clique_size": 0,
                    "clique_count": 0}
        return res_dict
    max_clique_size = max(len(c) for c in cliques)
    avg_clique_size = sum(len(c) for c in cliques) / len(cliques) if len(cliques) > 0 else 0
    num_nodes_in_cliques = sum(len(c) for c in cliques)
    # find percentage of causal edges

    res_dict = {"max_clique_size": max_clique_size,
                "avg_clique_size": avg_clique_size,
                "sum_clique_size": num_nodes_in_cliques,
                "clique_count": len(cliques)}
    return res_dict


def save_single_graph_analysis(G, causal_edges, input_to_id, cliques, graph_alpha, res_save_dir, id=None):
    if res_save_dir:
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
    else:
        return ""


def save_graph_analysis(graphs, graph_alpha, res_save_dir, id=None):
    if res_save_dir:
        res = {
            "alpha": graph_alpha,
            "graphs": graphs
        }
        time_str = datetime.now().strftime("%m%d-%H%M%S")
        if id:
            res_file_name = f"graph-id{id}-{time_str}.pkl"
            res["id"] = id
        else:
            res_file_name = f"graph-{time_str}.pkl"
        graph_save_path = os.path.join(res_save_dir, res_file_name)

        with open(graph_save_path, "wb") as f:
            pickle.dump(res, f)
        print("Saved graph analysis data to", graph_save_path)
        return graph_save_path
    else:
        return ""
