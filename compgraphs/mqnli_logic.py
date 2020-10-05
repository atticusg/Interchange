import torch

from datasets.mqnli import MQNLIData
from compgraphs.abstractable import AbstractableCompGraph
from typing import Any, Dict, List, Callable, Set

INDEP, EQUIV, ENTAIL, REV_ENTAIL, CONTRADICT, ALTER, COVER = range(7)

IDX_D_S, IDX_A_S, IDX_N_S, IDX_NEG, IDX_ADV, IDX_V, IDX_D_O, IDX_A_O, IDX_N_O = \
    range(9)


def get_relation_composition():
    res = dict()
    rel_list1 = [EQUIV, ENTAIL, REV_ENTAIL,
                 CONTRADICT,
                 COVER, ALTER, INDEP]
    rel_list2 = [EQUIV, ENTAIL, REV_ENTAIL,
                 CONTRADICT,
                 COVER, ALTER, INDEP]
    for r in rel_list1:
        for r2 in rel_list2:
            res[(r, r2)] = INDEP
    for r in rel_list1:
        res[(EQUIV, r)] = r
        res[(r, EQUIV)] = r
    res[(ENTAIL, ENTAIL)] = ENTAIL
    res[(ENTAIL, CONTRADICT)] = ALTER
    res[(ENTAIL, ALTER)] = ALTER
    res[
        (REV_ENTAIL, REV_ENTAIL)] = REV_ENTAIL
    res[(REV_ENTAIL, CONTRADICT)] = COVER
    res[(REV_ENTAIL, COVER)] = COVER
    res[(CONTRADICT, ENTAIL)] = COVER
    res[(CONTRADICT, REV_ENTAIL)] = ALTER
    res[(CONTRADICT, CONTRADICT)] = EQUIV
    res[(CONTRADICT, COVER)] = REV_ENTAIL
    res[(CONTRADICT, ALTER)] = ENTAIL
    res[(ALTER, REV_ENTAIL)] = ALTER
    res[(ALTER, CONTRADICT)] = ENTAIL
    res[(ALTER, COVER)] = ENTAIL
    res[(COVER, ENTAIL)] = COVER
    res[(COVER, CONTRADICT)] = REV_ENTAIL
    res[(COVER, ALTER)] = " reverse entails"
    return res


relation_composition = get_relation_composition()


def strong_composition(signature1, signature2, relation1, relation2):
    # returns the stronger relation of the first relation/signature composed
    # with the second relation signature and vice versa
    composition1 = relation_composition[
        (signature1[relation1], signature2[relation2])]
    composition2 = relation_composition[
        (signature2[relation2], signature1[relation1])]
    if composition1 == INDEP:
        return composition2
    if composition2 != INDEP and composition1 != composition2:
        print("This shouldn't happen", composition1, composition2)
    return composition1


def get_determiner_sigatures():
    det_sigs = dict()
    symmetric_relation = {EQUIV: EQUIV,
                          ENTAIL: REV_ENTAIL,
                          REV_ENTAIL: ENTAIL,
                          CONTRADICT: CONTRADICT, COVER: COVER,
                          ALTER: ALTER,
                          INDEP: INDEP}

    det_sigs[("some", "some")] = (
        {EQUIV: EQUIV, ENTAIL: ENTAIL,
         REV_ENTAIL: REV_ENTAIL, INDEP: INDEP},
        {EQUIV: EQUIV, ENTAIL: ENTAIL,
         REV_ENTAIL: REV_ENTAIL, CONTRADICT: COVER,
         COVER: COVER, ALTER: INDEP,
         INDEP: INDEP}
    )
    det_sigs[("every", "every")] = (
        {EQUIV: EQUIV, ENTAIL: REV_ENTAIL,
         REV_ENTAIL: ENTAIL, INDEP: INDEP},
        {EQUIV: EQUIV, ENTAIL: ENTAIL,
         REV_ENTAIL: REV_ENTAIL, CONTRADICT: ALTER,
         COVER: INDEP, ALTER: ALTER,
         INDEP: INDEP}
    )
    for key in det_sigs:
        signature1, signature2 = det_sigs[key]
        new_signature = dict()
        for key1 in signature1:
            for key2 in signature2:
                new_signature[(key1, key2)] = strong_composition(
                    signature1, signature2, key1, key2)
        det_sigs[key] = new_signature

    new_signature = dict()
    relations = [EQUIV, ENTAIL, REV_ENTAIL, CONTRADICT,
                 COVER, ALTER, INDEP]
    for relation1 in [EQUIV, ENTAIL, REV_ENTAIL,
                      INDEP]:
        for relation2 in relations:
            if (relation2 == EQUIV or relation2 == REV_ENTAIL) and relation1 != INDEP:
                new_signature[(relation1, relation2)] = REV_ENTAIL
            else:
                new_signature[(relation1, relation2)] = INDEP
    det_sigs[("some", "every")] = new_signature
    det_sigs[("some", "every")][
        (ENTAIL, CONTRADICT)] = ALTER
    det_sigs[("some", "every")][
        (ENTAIL, ALTER)] = ALTER
    det_sigs[("some", "every")][
        (EQUIV, ALTER)] = ALTER
    det_sigs[("some", "every")][
        (EQUIV, CONTRADICT)] = CONTRADICT
    det_sigs[("some", "every")][(EQUIV, COVER)] = COVER
    det_sigs[("some", "every")][
        (REV_ENTAIL, COVER)] = COVER
    det_sigs[("some", "every")][
        (REV_ENTAIL, CONTRADICT)] = COVER

    new_signature = dict()
    for key in det_sigs[("some", "every")]:
        new_signature[
            (symmetric_relation[key[0]], symmetric_relation[key[1]])] = \
            symmetric_relation[det_sigs["some", "every"][key]]
    det_sigs[("every", "some")] = new_signature

    return det_sigs

determiner_signatures = get_determiner_sigatures()

mqnli_logic_compgraph = {
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


class MQNLI_Logic_CompGraph(AbstractableCompGraph):
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
            full_graph=mqnli_logic_compgraph,
            root_node_name="sentence",
            abstract_nodes=intermediate_nodes,
            forward_functions=node_functions,
            topological_order=topological_order
        )

    def __getattr__(self, item):
        """ Used to retrieve keywords, relations or positions"""
        if item in self.keyword_dict:
            return self.keyword_dict[item]
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
        res[eq] = torch.tensor([INDEP, EQUIV], dtype=torch.long)
        res[forward_entail] = torch.tensor([INDEP, ENTAIL],
                                           dtype=torch.long)
        res[backward_entail] = torch.tensor([INDEP, REV_ENTAIL],
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
        return (p[IDX_N_O] == h[IDX_N_O]).type(torch.long)

    def obj_adj(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (9, batch_size), (9, batch_size) -> (batch_size, 2)
        return self._intersective_projection(p[IDX_A_O], h[IDX_A_O])

    def obj(self, a: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        # (batch_size, 2), (batch_size,) -> (batch_size,)
        return torch.gather(a, 1, n.unsqueeze(1)).squeeze()

    def vp_q(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (9, batch_size), (9, batch_size) -> (batch_size, 4, 4)
        return None

    def v_verb(self, p, h):
        # (9, batch_size), (9, batch_size) -> (batch_size,)
        return (p[IDX_V] == h[IDX_V]).type(torch.long)

    def v_adv(self, p, h):
        # (9, batch_size), (9, batch_size) -> (batch_size, 2)
        return self._intersective_projection(p[IDX_ADV], h[IDX_ADV])

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
        return (p[IDX_N_S] == h[IDX_N_S]).type(torch.long)

    def subj_adj(self, p, h):
        # (9, batch_size), (9, batch_size) -> (2, batch_size)
        return self._intersective_projection(p[IDX_A_S], h[IDX_A_S])

    def subj(self, a, n):
        # (batch_size, 2), (batch_size,) -> (batch_size,)
        return torch.gather(a, 1, n.unsqueeze(1)).squeeze()

    def sentence_q(self, p, h):
        # print(f"sentence_q({p}, {h})")
        return f"sentence_q({p}, {h})"

    def sentence(self, q, s, n):
        # print(f"sentence({q}, {s}, {n})")
        return f"sentence({q}, {s}, {n})"
