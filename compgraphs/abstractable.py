from intervention import ComputationGraph
from intervention import GraphNode

from typing import Any, Dict, List, Callable, Set


class AbstractableCompGraph(ComputationGraph):
    def __init__(self, full_graph: Dict[str, List[str]],
                 root_node_name: str,
                 abstract_nodes: List[str],
                 forward_functions: Dict[str, Callable],
                 topological_order: List[str], ):
        """ An abstractable compgraph structure

        :param full_graph:
        :param root_node_name:
        :param abstract_nodes:
        :param forward_functions:
        :param topological_order:
        """
        self.full_graph = full_graph
        self.input_node_names = {k for k, v in full_graph.items() if len(v) == 0}
        self.root_node_name = root_node_name
        self.forward_functions = forward_functions
        self.topological_order = topological_order

        # TODO: extract topological order automatically

        root = self.generate_abstract_graph(abstract_nodes)
        super(AbstractableCompGraph, self).__init__(root)

    def generate_abstract_graph(self, abstract_nodes: List[str]) \
            -> Dict[str, GraphNode]:
        # rearrange nodes in reverse topological order
        relevant_nodes = self.get_node_names(abstract_nodes)
        relevant_node_set = set(relevant_nodes)

        # define input leaf node
        node_dict = {name: self.generate_input_node(name) for name in self.input_node_names}

        for node_name in relevant_nodes:
            if node_name in self.input_node_names:
                continue

            curr_children = self.get_children(node_name, relevant_node_set)
            args = [node_dict[child] for child in curr_children]
            forward = self.generate_forward_function(node_name, curr_children)
            node_dict[node_name] = GraphNode(*args, name=node_name,
                                               forward=forward)

        return node_dict[self.root_node_name]

    def get_node_names(self, abstract_nodes: List[str]) -> List[str]:
        """ Get topologically ordered list of node names in final compgraph,
        given intermediate nodes"""
        abstract_nodes = set(abstract_nodes)
        if self.root_node_name not in abstract_nodes:
            abstract_nodes.add(self.root_node_name)

        for input_node_name in self.input_node_names:
            if input_node_name not in abstract_nodes:
                abstract_nodes.add(input_node_name)

        res = []
        for i in range(len(self.topological_order) - 1, -1, -1):
            if self.topological_order[i] in abstract_nodes:
                res.append(self.topological_order[i])
        return res

    def get_children(self, abstract_node: str, abstract_node_set: Set[str]) \
            -> List[str]:
        """ Get immediate children in abstracted graph given an abstract node """
        res = []
        stack = [abstract_node]
        visited = set()

        while len(stack) > 0:
            curr_node = stack.pop()
            visited.add(curr_node)
            if curr_node != abstract_node and curr_node in abstract_node_set:
                res.append(curr_node)
            else:
                for i in range(len(self.full_graph[curr_node]) - 1, -1, -1):
                    child = self.full_graph[curr_node][i]
                    if child not in visited:
                        stack.append(child)
        return res

    def generate_input_node(self, name: str) -> GraphNode:
        def _input_forward_fxn(x):
            return x
        return GraphNode(name=name, forward=_input_forward_fxn)

    def generate_forward_function(self, abstracted_node: str,
                                  children: List[str]) -> Callable:
        """ Generate forward function to construct an abstract node """
        children_dict = {name: i for i, name in enumerate(children)}

        def _forward(*args):
            if len(args) != len(children):
                raise ValueError(f"Got {len(args)} arguments to forward fxn of "
                                 f"{abstracted_node}, expected {len(children)}")

            def _implicit_call(node_name: str) -> Any:
                if node_name in children_dict:
                    child_idx_in_args = children_dict[node_name]
                    return args[child_idx_in_args]
                else:
                    res = [_implicit_call(child) for child in self.full_graph[node_name]]
                    current_f = self.forward_functions[node_name]
                    return current_f(*res)

            return _implicit_call(abstracted_node)

        return _forward