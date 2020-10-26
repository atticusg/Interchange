import networkx as nx
import copy
import numpy as np

def get_input(intervention):
    return tuple({(k, tuple(intervention.base.values[k])) for k in intervention.base.values})

def construct_graph(low_model, high_model, mapping, result, realizations_to_inputs, high_node_name, high_root_name):
    G = nx.Graph()
    input_to_id = dict()
    id_to_input = dict()
    inputs_seen = set()
    edges = set()
    total_id = 0
    causal_edges = set()
    count = 0
    for interventions in result:
        count +=1
        low_intervention, high_intervention = interventions
        input = get_input(low_intervention)
        if input not in inputs_seen:
            G.add_node(total_id)
            input_to_id[input] = total_id
            id_to_input[total_id] = input
            total_id +=1
            inputs_seen.add(input)
        low_node = None
        for key in mapping[high_node_name]:
            low_node = key
        index = mapping[high_node_name][low_node]
        string_array = low_model.get_result(low_node,low_intervention)[index].tostring()
        input2 = get_input(realizations_to_inputs[(string_array, high_node_name)])
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
    return G, causal_edges

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
    return cliques
