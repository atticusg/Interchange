import torch

from datasets.mqnli import MQNLIData
from intervention import ComputationGraph
from intervention import GraphNode
from typing import Any, Dict, List, Callable, Set


class AbstractableCompGraph(ComputationGraph):
    def __init__(self, full_graph: Dict[str, List[str]],
                 root_name: str,
                 abstract_nodes: List[str],
                 node_functions: Dict[str, Callable],
                 topological_order: List[str], ):
        """ An abstractable compgraph structure

        :param full_graph:
        :param root_name:
        :param abstract_nodes:
        :param node_functions:
        :param topological_order:
        """
        self.full_graph = full_graph
        self.root_name = root_name
        self.node_functions = node_functions
        self.topological_order = topological_order

        root = self.generate_abstract_graph(abstract_nodes)
        super(AbstractableCompGraph, self).__init__(root)

    def generate_abstract_graph(self, abstract_nodes: List[str]) \
            -> Dict[str, GraphNode]:
        # rearrange nodes in reverse topological order
        relevant_nodes = self.get_node_names(abstract_nodes)
        relevant_node_set = set(relevant_nodes)

        # define input leaf node
        @GraphNode()
        def input(x):
            return x

        node_dict = {"input": input}

        for node_name in relevant_nodes:
            curr_children = self.get_children(node_name, relevant_node_set)
            args = [node_dict[child] for child in curr_children]
            forward = self.generate_forward_function(node_name, curr_children)
            node_dict[node_name] = GraphNode(*args, name=node_name,
                                               forward=forward)

        return node_dict[self.root_name]

    def get_node_names(self, abstract_nodes: List[str]) -> List[str]:
        """ Get topologically ordered list of node names in final compgraph,
        given intermediate nodes"""
        nodes_in_graph = set(abstract_nodes)
        if "sentence" not in nodes_in_graph:
            nodes_in_graph.add("sentence")
        if "input" not in nodes_in_graph:
            nodes_in_graph.add("input")

        res = []
        for i in range(len(self.topological_order) - 1, -1, -1):
            if self.topological_order[i] in nodes_in_graph:
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

    def generate_forward_function(self, abstracted_node: str,
                                  children: List[str]) -> Callable:
        """ Generate forward function to construct an abstract node """
        children_dict = {name: i for i, name in enumerate(children)}

        def _forward(*args):
            if len(args) != children:
                raise ValueError(f"Got {len(args)} arguments to forward fxn of "
                                 f"{abstracted_node}, expected{len(children)}")

            def _implicit_call(node_name: str) -> Any:
                if node_name in children_dict:
                    child_idx_in_args = children_dict[node_name]
                    return args[child_idx_in_args]
                else:
                    res = [_implicit_call(child) for child in self.full_graph]
                    current_f = self.node_functions[node_name]
                    return current_f(*res)

            return _implicit_call(abstracted_node)

        return _forward


class MQNLI_Logic_CompGraph(AbstractableCompGraph):
    def __init__(self, data: MQNLIData, intermediate_nodes: List[str]):
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word

        full_graph = {
            "input": [],
            "get_p": ["input"],
            "get_h": ["input"],
            "obj_noun": ["get_p", "get_h"],
            "obj_adj": ["get_p", "get_h"],
            "obj": ["obj_adj", "obj_noun"],
            "vp_q": ["get_p", "get_h"],
            "v_verb": ["get_p", "get_h"],
            "v_adv": ["get_p", "get_h"],
            "v_bar": ["v_adv", "v_verb"],
            "vp": ["v_bar", "vp_q", "obj"],
            "neg": ["get_p", "get_h"],
            "negp": ["neg", "vp"],
            "subj_noun": ["get_p", "get_h"],
            "subj_adj": ["get_p", "get_h"],
            "subj": ["subj_adj", "subj_noun"],
            "sentence_q": ["get_p", "get_h"],
            "sentence": ["sentence_q", "subj", "negp"]
        }

        topological_order = ["sentence", "sentence_q", "subj", "negp",
                             "subj_adj", "subj_noun", "neg", "vp",
                             "v_bar", "vp_q", "obj", "v_adv", "v_verb",
                             "obj_adj", "obj_noun", "get_p", "get_h",
                             "input"]

        self.keyword_dict = {w: self.word_to_id[w] for w in [
            "emptystring", "no", "some", "every", "notevery", "doesnot"
        ]}

        node_functions = {f.__name__: f for f in [
            self.get_p, self.get_h,
            self.obj_noun, self.obj_adj, self.obj,
            self.vp_q, self.v_verb, self.v_adv, self.v_bar, self.vp,
            self.neg, self.negp,
            self.subj_noun, self.subj_adj, self.subj,
            self.sentence_q, self.sentence
        ]}

        super(MQNLI_Logic_CompGraph, self).__init__(
            full_graph=full_graph,
            root_name="sentence",
            abstract_nodes=intermediate_nodes,
            node_functions=node_functions,
            topological_order=topological_order
        )

    def __getattr__(self, item):
        if item in self._keyword_dict:
            return self._keyword_dict[item]

    ### MQNLI functions
    def get_p(self, input):
        print(f"get_p({input})")
        return "<get_p>"

    def get_h(self, input):
        print(f"get_h({input})")
        return "<get_h>"

    def obj_noun(self, p, h):
        print(f"obj_noun({p}, {h})")
        return "<obj_noun>"

    def obj_adj(self, p, h):
        print(f"obj_adj({p}, {h})")
        return "<obj_adj>"

    def obj(self, a, n):
        print(f"obj({a}, {n})")
        return "<obj>"

    def vp_q(self, p, h):
        print(f"vp_q({p}, {h})")
        return "<vp_q>"

    def v_verb(self, p, h):
        print(f"v_verb({p}, {h})")
        return "<v_verb>"

    def v_adv(self, p, h):
        print(f"v_adv({p}, {h})")
        return "<v_adv>"

    def v_bar(self, a, v):
        print(f"v_bar({a}, {v})")
        return "<v_bar>"

    def vp(self, v, q, o):
        print(f"vp({v}, {q}, {o})")
        return "<vp>"

    def neg(self, p, h):
        print(f"neg({p}, {h})")
        return "<neg>"

    def negp(self, n, v):
        print(f"negp({n}, {v})")
        return "<negp>"

    def subj_noun(self, p, h):
        print(f"subj_noun({p}, {h})")
        return "<subj_noun>"

    def subj_adj(self, p, h):
        print(f"subj_adj({p}, {h})")
        return "<subj_adj>"

    def subj(self, a, n):
        print(f"subj({a}, {n})")
        return "<subj>"

    def sentence_q(self, p, h):
        print(f"sentence_q({p}, {h})")
        return "<sentence_q>"

    def sentence(self, q, s, n):
        print(f"sentence({q}, {s}, {n})")
        return "<sentence>"
