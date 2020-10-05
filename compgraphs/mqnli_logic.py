import torch

from datasets.mqnli import MQNLIData
from compgraphs.abstractable import AbstractableCompGraph
from typing import Any, Dict, List, Callable, Set

INDEP, EQUIV, ENTAIL, REV_ENTAIL, CONTRADICT, ALTER, COVER = range(7)

IDX_D_S, IDX_A_S, IDX_N_S, IDX_NEG, IDX_ADV, IDX_V, IDX_D_O, IDX_A_O, IDX_N_O = range(9)

SOME, EVERY, NO, NOTEVERY = range(4)


def get_relation_composition():
    composition_dict = dict()
    rel_list1 = [EQUIV, ENTAIL, REV_ENTAIL,
                 CONTRADICT,
                 COVER, ALTER, INDEP]
    rel_list2 = [EQUIV, ENTAIL, REV_ENTAIL,
                 CONTRADICT,
                 COVER, ALTER, INDEP]
    for r in rel_list1:
        for r2 in rel_list2:
            composition_dict[(r, r2)] = INDEP
    for r in rel_list1:
        composition_dict[(EQUIV, r)] = r
        composition_dict[(r, EQUIV)] = r
    composition_dict[(ENTAIL, ENTAIL)] = ENTAIL
    composition_dict[(ENTAIL, CONTRADICT)] = ALTER
    composition_dict[(ENTAIL, ALTER)] = ALTER
    composition_dict[(REV_ENTAIL, REV_ENTAIL)] = REV_ENTAIL
    composition_dict[(REV_ENTAIL, CONTRADICT)] = COVER
    composition_dict[(REV_ENTAIL, COVER)] = COVER
    composition_dict[(CONTRADICT, ENTAIL)] = COVER
    composition_dict[(CONTRADICT, REV_ENTAIL)] = ALTER
    composition_dict[(CONTRADICT, CONTRADICT)] = EQUIV
    composition_dict[(CONTRADICT, COVER)] = REV_ENTAIL
    composition_dict[(CONTRADICT, ALTER)] = ENTAIL
    composition_dict[(ALTER, REV_ENTAIL)] = ALTER
    composition_dict[(ALTER, CONTRADICT)] = ENTAIL
    composition_dict[(ALTER, COVER)] = ENTAIL
    composition_dict[(COVER, ENTAIL)] = COVER
    composition_dict[(COVER, CONTRADICT)] = REV_ENTAIL
    composition_dict[(COVER, ALTER)] = REV_ENTAIL

    res = torch.zeros(7,7,dtype=torch.long)
    for (r1, r2), v in composition_dict:
        res[r1,r2] = v
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

# TODO: test this
def get_negation_signatures():
    res = torch.zeros(2,2,7)
    res[0,0] = torch.tensor([INDEP, EQUIV, ENTAIL, REV_ENTAIL, CONTRADICT, ALTER, COVER], dtype=torch.long)

    # {"equivalence":"equivalence",
    # "entails":"reverse entails",
    # "reverse entails":"entails",
    # "contradiction":"contradiction",
    # "cover":"alternation",
    # "alternation":"cover",
    # "independence":"independence"}
    res[1,1] = torch.tensor([INDEP, EQUIV, REV_ENTAIL, ENTAIL, CONTRADICT, COVER, ALTER], dtype=torch.long)

    for rel in range(7):
        res[0,1,rel] = relation_composition[rel,CONTRADICT]

    for rel in range(7):
        res[1,0,rel] = res[0,1,res[1,1,rel]]

    return res


# TODO: test this
def get_determiner_sigatures():
    det_sigs = dict()
    symmetric_relation = {EQUIV: EQUIV,
                          ENTAIL: REV_ENTAIL,
                          REV_ENTAIL: ENTAIL,
                          CONTRADICT: CONTRADICT, COVER: COVER,
                          ALTER: ALTER,
                          INDEP: INDEP}

    det_sigs[(SOME, SOME)] = (
        {EQUIV: EQUIV, ENTAIL: ENTAIL,
         REV_ENTAIL: REV_ENTAIL, INDEP: INDEP},
        {EQUIV: EQUIV, ENTAIL: ENTAIL,
         REV_ENTAIL: REV_ENTAIL, CONTRADICT: COVER,
         COVER: COVER, ALTER: INDEP,
         INDEP: INDEP}
    )
    det_sigs[(EVERY, EVERY)] = (
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
    det_sigs[(SOME, EVERY)] = new_signature
    det_sigs[(SOME, EVERY)][
        (ENTAIL, CONTRADICT)] = ALTER
    det_sigs[(SOME, EVERY)][
        (ENTAIL, ALTER)] = ALTER
    det_sigs[(SOME, EVERY)][
        (EQUIV, ALTER)] = ALTER
    det_sigs[(SOME, EVERY)][
        (EQUIV, CONTRADICT)] = CONTRADICT
    det_sigs[(SOME, EVERY)][(EQUIV, COVER)] = COVER
    det_sigs[(SOME, EVERY)][
        (REV_ENTAIL, COVER)] = COVER
    det_sigs[(SOME, EVERY)][
        (REV_ENTAIL, CONTRADICT)] = COVER

    new_signature = dict()
    for key in det_sigs[(SOME, EVERY)]:
        new_signature[
            (symmetric_relation[key[0]], symmetric_relation[key[1]])] = \
            symmetric_relation[det_sigs[SOME, EVERY][key]]
    det_sigs[(EVERY, SOME)] = new_signature
    
    res = torch.zeros(4,4,4,7, dtype=torch.long)

    for (q1, q2), d in det_sigs.items():
        for (r1, r2), v in d.items():
            res[q1,q2,r1,r2] = v

    return res

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
            forward_functions=node_functions
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
