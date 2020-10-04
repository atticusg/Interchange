import torch

from datasets.mqnli import MQNLIData
from compgraphs.abstractable import AbstractableCompGraph
from typing import Any, Dict, List, Callable, Set


class MQNLI_Logic_CompGraph(AbstractableCompGraph):
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

    relations = {
        "INDEP": 0,
        "EQUIV": 1,
        "ENTAIL": 2,
        "REV_ENTAIL": 3,
        "NEGATION": 4,
        "ALTER": 5,
        "COVER": 6
    }

    positions = {
        "Q_S": 0,
        "A_S": 1,
        "N_S": 2,
        "NEG": 3,
        "ADV": 4,
        "V": 5,
        "Q_O": 6,
        "A_O": 7,
        "N_O": 8
    }

    def __init__(self, data: MQNLIData, intermediate_nodes: List[str]):
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word

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
            full_graph=MQNLI_Logic_CompGraph.full_graph,
            root_node_name="sentence",
            abstract_nodes=intermediate_nodes,
            forward_functions=node_functions,
            topological_order=MQNLI_Logic_CompGraph.topological_order
        )

    def __getattr__(self, item):
        """ Used to retrieve keywords, relations or positions"""
        if item in self.keyword_dict:
            return self.keyword_dict[item]
        elif item in MQNLI_Logic_CompGraph.relations:
            return MQNLI_Logic_CompGraph.relations[item]
        elif item in MQNLI_Logic_CompGraph.positions:
            return MQNLI_Logic_CompGraph.positions[item]
        else:
            raise KeyError

    def _intersective_projection(self, p: torch.Tensor, h: torch.Tensor) \
            -> torch.Tensor:
        # (batch_size,), (batch_size,) -> (2, batch_size)
        eq = (p == h)
        p_is_empty = (p == self.emptystring)
        h_is_empty = (h == self.emptystring)
        forward_entail = (~p_is_empty & h_is_empty)
        backward_entail = (p_is_empty & ~h_is_empty)

        res = torch.zeros(p.size(0), 2, dtype=torch.long)
        res[eq] = torch.tensor([self.INDEP, self.EQUIV], dtype=torch.long)
        res[forward_entail] = torch.tensor([self.INDEP, self.ENTAIL],
                                           dtype=torch.long)
        res[backward_entail] = torch.tensor([self.INDEP, self.REV_ENTAIL],
                                            dtype=torch.long)
        return res

    # MQNLI functions
    def get_p(self, input: torch.Tensor) -> torch.Tensor:
        # (18, batch_size) ->  (9, batch_size)
        assert isinstance(input, torch.Tensor) \
               and len(input.shape) == 2 and input.shape[0] == 18, \
               f"invalid input shape: {input.shape}"
        return input[:9, :]

    def get_h(self, input: torch.Tensor) -> torch.Tensor:
        # (18, batch_size) ->  (9, batch_size)
        assert isinstance(input, torch.Tensor) \
               and len(input.shape) == 2 and input.shape[0] == 18, \
            f"invalid input shape: {input.shape}"
        return input[9:, :]

    def obj_noun(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (9, batch_size), (9, batch_size) -> (batch_size,)
        return (p[self.N_O] == h[self.N_O]).type(torch.long)

    def obj_adj(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (9, batch_size), (9, batch_size) -> (batch_size, 2)
        return self._intersective_projection(p[self.A_O], h[self.A_O])

    def obj(self, a: torch.Tensor, n: torch.Tensor):
        # (batch_size, 2), (batch_size,) -> (batch_size,)
        return torch.gather(a, 1, n.unsqueeze(1)).squeeze()

    def vp_q(self, p, h):
        # print(f"vp_q({p}, {h})")
        return f"vp_q({p}, {h})"

    def v_verb(self, p, h):
        # (9, batch_size), (9, batch_size) -> (batch_size,)
        return (p[self.V] == h[self.V]).type(torch.long)

    def v_adv(self, p, h):
        # (9, batch_size), (9, batch_size) -> (batch_size, 2)
        return self._intersective_projection(p[self.ADV], h[self.ADV])

    def v_bar(self, a, v):
        # (batch_size, 2), (batch_size,) -> (batch_size,)
        return torch.gather(a, 1, v.unsqueeze(1)).squeeze()

    def vp(self, v, q, o):
        # print(f"vp({v}, {q}, {o})")
        return f"vp({v}, {q}, {o})"

    def neg(self, p, h):
        # print(f"neg({p}, {h})")
        return f"neg({p}, {h})"

    def negp(self, n, v):
        # print(f"negp({n}, {v})")
        return f"negp({n}, {v})"

    def subj_noun(self, p, h):
        # (9, batch_size), (9, batch_size) -> (batch_size,)
        return (p[self.N_S] == h[self.N_S]).type(torch.long)

    def subj_adj(self, p, h):
        # (9, batch_size), (9, batch_size) -> (2, batch_size)
        return self._intersective_projection(p[self.A_S], h[self.A_S])

    def subj(self, a, n):
        # (batch_size, 2), (batch_size,) -> (batch_size,)
        return torch.gather(a, 1, n.unsqueeze(1)).squeeze()

    def sentence_q(self, p, h):
        # print(f"sentence_q({p}, {h})")
        return f"sentence_q({p}, {h})"

    def sentence(self, q, s, n):
        # print(f"sentence({q}, {s}, {n})")
        return f"sentence({q}, {s}, {n})"
