import torch
import itertools
from datasets.mqnli import MQNLIData
from antra import ComputationGraph, GraphNode, LOC
from antra.abstractable import AbstractableCompGraph
from compgraphs.abstractable import AbstractableCompGraph as OldAbstractableCompGraph
from typing import Any, Dict, List, Callable, Set

INDEP, EQUIV, ENTAIL, REV_ENTAIL, CONTRADICT, ALTER, COVER = range(7)

IDX_Q_S, IDX_A_S, IDX_N_S, IDX_NEG, IDX_ADV, IDX_V, \
IDX_Q_O, IDX_A_O, IDX_N_O = range(9)

SOME, EVERY, NO, NOTEVERY = range(4)


def get_relation_composition():
    # implement dict using exact same logic as
    # MultiplyQuantifiedData/natural_logic_model.py

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

    # construct tensor
    res = torch.zeros(7, 7, dtype=torch.long)
    for (r1, r2), v in composition_dict.items():
        res[r1,r2] = v
    return res


relation_composition = get_relation_composition()


def strong_composition(signature1, signature2, relation1, relation2):
    # helper function copied from MultiplyQuantifiedData/natural_logic_model.py
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


def get_negation_signatures():
    # use same logic as MultiplyQuantifiedData/natural_logic_model.py
    res = torch.zeros(4, 7, dtype=torch.long)
    res[0] = torch.tensor([INDEP, EQUIV, ENTAIL, REV_ENTAIL,
                             CONTRADICT, ALTER, COVER], dtype=torch.long)
    res[3] = torch.tensor([INDEP, EQUIV, REV_ENTAIL, ENTAIL,
                             CONTRADICT, COVER, ALTER], dtype=torch.long)

    for rel in range(7):
        res[1,rel] = relation_composition[rel,CONTRADICT]

    for rel in range(7):
        res[2,rel] = res[1,res[3,rel]]

    return res # [2*2, 7]


negation_signatures = get_negation_signatures()


def get_quantifier_signatures():
    # first construct sigs_dict dict using exact same logic as
    # MultiplyQuantifiedData/natural_logic_model.py

    sigs_dict = dict()
    symmetric_relation = {EQUIV: EQUIV, ENTAIL: REV_ENTAIL, REV_ENTAIL: ENTAIL,
                          CONTRADICT: CONTRADICT, COVER: COVER, ALTER: ALTER,
                          INDEP: INDEP}

    sigs_dict[(SOME, SOME)] = (
        {EQUIV: EQUIV, ENTAIL: ENTAIL, REV_ENTAIL: REV_ENTAIL, INDEP: INDEP},
        {EQUIV: EQUIV, ENTAIL: ENTAIL, REV_ENTAIL: REV_ENTAIL,
         CONTRADICT: COVER, COVER: COVER, ALTER: INDEP, INDEP: INDEP}
    )
    sigs_dict[(EVERY, EVERY)] = (
        {EQUIV: EQUIV, ENTAIL: REV_ENTAIL, REV_ENTAIL: ENTAIL, INDEP: INDEP},
        {EQUIV: EQUIV, ENTAIL: ENTAIL, REV_ENTAIL: REV_ENTAIL,
         CONTRADICT: ALTER, COVER: INDEP, ALTER: ALTER, INDEP: INDEP}
    )
    for key in sigs_dict:
        signature1, signature2 = sigs_dict[key]
        new_signature = dict()
        for key1 in signature1:
            for key2 in signature2:
                new_signature[(key1, key2)] = strong_composition(
                    signature1, signature2, key1, key2)
        sigs_dict[key] = new_signature

    new_signature = dict()
    relations = [EQUIV, ENTAIL, REV_ENTAIL, CONTRADICT,
                 COVER, ALTER, INDEP]
    for relation1 in [EQUIV, ENTAIL, REV_ENTAIL,
                      INDEP]:
        for relation2 in relations:
            if (relation2 == EQUIV or relation2 == REV_ENTAIL) \
                    and relation1 != INDEP:
                new_signature[(relation1, relation2)] = REV_ENTAIL
            else:
                new_signature[(relation1, relation2)] = INDEP
    sigs_dict[(SOME, EVERY)] = new_signature
    sigs_dict[(SOME, EVERY)][(ENTAIL, CONTRADICT)] = ALTER
    sigs_dict[(SOME, EVERY)][(ENTAIL, ALTER)] = ALTER
    sigs_dict[(SOME, EVERY)][(EQUIV, ALTER)] = ALTER
    sigs_dict[(SOME, EVERY)][(EQUIV, CONTRADICT)] = CONTRADICT
    sigs_dict[(SOME, EVERY)][(EQUIV, COVER)] = COVER
    sigs_dict[(SOME, EVERY)][(REV_ENTAIL, COVER)] = COVER
    sigs_dict[(SOME, EVERY)][(REV_ENTAIL, CONTRADICT)] = COVER

    new_signature = dict()
    for key in sigs_dict[(SOME, EVERY)]:
        new_signature[(symmetric_relation[key[0]], symmetric_relation[key[1]])] \
            = symmetric_relation[sigs_dict[SOME, EVERY][key]]
    sigs_dict[(EVERY, SOME)] = new_signature

    # then construct tensor given the above
    res = torch.zeros(4*4, 4*7, dtype=torch.long)

    for neg_1 in [0, 1]:
        for neg_2 in [0, 1]:
            for (q1, q2), d in sigs_dict.items():
                for (r1, r2), v in d.items():
                    neg_sig = negation_signatures[neg_1*2 + neg_2]
                    res[4*(neg_1*2+q1) + (neg_2*2+q2), 7*r1 + r2] = neg_sig[v]
    return res # [4*4, 4*7]

quantifier_signatures = get_quantifier_signatures()

output_remapping = torch.tensor([0, 1, 1, 0, 2, 2, 0], dtype=torch.long)

compgraph_structure = {
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
    "root": ["sentence_q", "subj", "negp"]
}

def get_intersective_projections():
    ps = [[INDEP, INDEP], [INDEP, EQUIV], [INDEP, ENTAIL], [INDEP, REV_ENTAIL]]
    return [torch.tensor(p, dtype=torch.long) for p in ps]

intersective_projections = get_intersective_projections()

"""Deprecated high-level compgraph class. This compgraph does not have nodes
for each individual token in the input, e.g. `v_verb_p` or `obj_adj_h`. It
only has nodes for intermediate nodes in the tree, with the lowest level being
combinations of two words in the input, such as `obj_adj`, composed of
`obj_adj_p` and `obj_adj_h`.
Use `Full_MQNLI_Logic_CompGraph` instead."""
class MQNLI_Logic_CompGraph(ComputationGraph):
    def __init__(self, data: MQNLIData, root_output_device=None):
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.keyword_dict = {w: self.word_to_id[w] for w in [
            "emptystring", "no", "some", "every", "notevery", "doesnot"
        ]}

        @GraphNode()
        def input(x):
            return x

        input.cache_results = False

        @GraphNode(input)
        def get_p(x: torch.Tensor) -> torch.Tensor:
            # (18, batch_size) ->  (9, batch_size)
            return x[:9, :]

        get_p.cache_results = False

        @GraphNode(input)
        def get_h(x: torch.Tensor) -> torch.Tensor:
            # (18, batch_size) ->  (9, batch_size)
            return x[9:, :]

        get_h.cache_results = False

        @GraphNode(get_p, get_h)
        def obj_noun(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (9, batch_size), (9, batch_size) -> (batch_size,)
            return (p[IDX_N_O] == h[IDX_N_O]).type(torch.long)

        @GraphNode(get_p, get_h)
        def obj_adj(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (9, batch_size), (9, batch_size) -> (batch_size, 2)
            return self._intersective_projection(p[IDX_A_O], h[IDX_A_O])

        @GraphNode(obj_adj, obj_noun)
        def obj(a: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            # (batch_size, 2), (batch_size,) -> (batch_size,)
            return torch.gather(a, 1, n.unsqueeze(1)).view(n.shape[0])

        @GraphNode(get_p, get_h)
        def vp_q(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (9, batch_size), (9, batch_size) -> (batch_size, 4*7)
            return self._merge_quantifiers(p[IDX_Q_O], h[IDX_Q_O])

        @GraphNode(get_p, get_h)
        def v_verb(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (9, batch_size), (9, batch_size) -> (batch_size,)
            return (p[IDX_V] == h[IDX_V]).type(torch.long)

        @GraphNode(get_p, get_h)
        def v_adv(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (9, batch_size), (9, batch_size) -> (batch_size, 2)
            return self._intersective_projection(p[IDX_ADV], h[IDX_ADV])

        @GraphNode(v_adv, v_verb)
        def v_bar(a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            # (batch_size, 2), (batch_size,) -> (batch_size,)
            return torch.gather(a, 1, v.unsqueeze(1)).view(v.shape[0])

        @GraphNode(v_bar, vp_q, obj)
        def vp(v: torch.Tensor, q: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
            # (batch_size,), (batch_size, 4*7), (batch_size,) -> (batch_size,)
            idxs = (o * 7 + v).unsqueeze(1)  # note that o is the first argument
            return torch.gather(q, 1, idxs).view(v.shape[0])

        @GraphNode(get_p, get_h)
        def neg(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (9, batch_size), (9, batch_size) -> (batch_size, 7)
            return self._merge_negation(p[IDX_NEG], h[IDX_NEG])

        @GraphNode(neg, vp)
        def negp(n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            # (batch_size, 7), (batch_size,) -> (batch_size,)
            return torch.gather(n, 1, v.unsqueeze(1)).view(v.shape[0])

        @GraphNode(get_p, get_h)
        def subj_noun(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (9, batch_size), (9, batch_size) -> (batch_size,)
            return (p[IDX_N_S] == h[IDX_N_S]).type(torch.long)

        @GraphNode(get_p, get_h)
        def subj_adj(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (9, batch_size), (9, batch_size) -> (batch_size, 2)
            return self._intersective_projection(p[IDX_A_S], h[IDX_A_S])

        @GraphNode(subj_adj, subj_noun)
        def subj(a: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            # (batch_size, 2), (batch_size,) -> (batch_size,)
            return torch.gather(a, 1, n.unsqueeze(1)).view(n.shape[0])

        @GraphNode(get_p, get_h)
        def sentence_q(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (9, batch_size), (9, batch_size) -> (batch_size, 4*7)
            return self._merge_quantifiers(p[IDX_Q_S], h[IDX_Q_S])

        @GraphNode(sentence_q, subj, negp)
        def root(q: torch.Tensor, s: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            # (batch_size, 4*7), (batch_size,), (batch_size,) -> (batch_size,)
            idxs = (s * 7 + n).unsqueeze(1)
            res = torch.gather(q, 1, idxs).view(n.shape[0])
            res = output_remapping[res]
            return res

        super(MQNLI_Logic_CompGraph, self).__init__(root)

    def __getattr__(self, item):
        """ Used to retrieve keywords, relations or positions"""
        if item in self.keyword_dict:
            return self.keyword_dict[item]
        else:
            raise KeyError

    def _intersective_projection(self, p: torch.Tensor, h: torch.Tensor) \
            -> torch.Tensor:
        # (batch_size,), (batch_size,) -> (batch_size, 2)
        eq = (p == h)
        p_is_empty = (p == self.emptystring)
        h_is_empty = (h == self.emptystring)
        forward_entail = (~p_is_empty & h_is_empty)
        backward_entail = (p_is_empty & ~h_is_empty)

        res = torch.zeros(p.size(0), 2, dtype=torch.long)
        res[eq] = torch.tensor([INDEP, EQUIV], dtype=torch.long)
        res[forward_entail] = torch.tensor([INDEP, ENTAIL], dtype=torch.long)
        res[backward_entail] = torch.tensor([INDEP, REV_ENTAIL], dtype=torch.long)
        return res

    def _merge_quantifiers(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (batch_size,), (batch_size,) -> (batch_size, 4*7)
        p_idx_tensor = torch.zeros(p.size(0), dtype=torch.long)
        h_idx_tensor = torch.zeros(h.size(0), dtype=torch.long)
        for q_idx, q_token in enumerate([self.some, self.every, self.no, self.notevery]):
            p_idx_tensor[p == q_token] = q_idx
            h_idx_tensor[h == q_token] = q_idx
        idx_tensor = p_idx_tensor * 4 + h_idx_tensor
        return quantifier_signatures.index_select(0, idx_tensor)

    def _merge_negation(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (batch_size,), (batch_size,) -> (batch_size, 4, 7)
        p_idx_tensor = torch.zeros(p.size(0), dtype=torch.long)
        h_idx_tensor = torch.zeros(h.size(0), dtype=torch.long)
        for q_idx, q_token in enumerate([self.emptystring, self.doesnot]):
            p_idx_tensor[p == q_token] = q_idx
            h_idx_tensor[h == q_token] = q_idx
        idx_tensor = p_idx_tensor * 2 + h_idx_tensor
        return negation_signatures.index_select(0, idx_tensor)

nodes2idxs = {
    "obj_noun": IDX_N_O,
    "obj_adj": IDX_A_O,
    "obj_q": IDX_Q_O,
    "verb": IDX_V,
    "adv": IDX_ADV,
    "neg": IDX_NEG,
    "subj_noun": IDX_N_S,
    "subj_adj": IDX_A_S,
    "subj_q": IDX_Q_S
}

def _generate_leaf_fn(idx):
    def _leaf_fn(x):
        return x[:, idx]
    return _leaf_fn

class Full_MQNLI_Logic_CompGraph(ComputationGraph):
    def __init__(self, data: MQNLIData, device=None):
        self.word_to_id = data.word_to_id
        self.id_to_word = data.id_to_word
        self.keyword_dict = {w: self.word_to_id[w] for w in [
            "emptystring", "no", "some", "every", "notevery", "doesnot"
        ]}
        self.device = torch.device("cpu") if device is None else device

        self.quantifier_signatures = quantifier_signatures.to(self.device)
        self.negation_signatures = negation_signatures.to(self.device)
        self.output_remapping = output_remapping.to(self.device)

        input_node = self.get_input_node()

        leaf_nodes = {}
        for leaf_name, idx in nodes2idxs.items():
            p_forward_fn = _generate_leaf_fn(idx)
            p_leaf_name = "p_" + leaf_name
            p_leaf_node = GraphNode(input_node, name=p_leaf_name, forward=p_forward_fn)
            leaf_nodes[p_leaf_name] = p_leaf_node

            h_forward_fn = _generate_leaf_fn(9 + idx)
            h_leaf_name = "h_" + leaf_name
            h_leaf_node = GraphNode(input_node, name=h_leaf_name, forward=h_forward_fn)
            leaf_nodes[h_leaf_name] = h_leaf_node

        @GraphNode(leaf_nodes["p_obj_noun"], leaf_nodes["h_obj_noun"])
        def obj_noun(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (batch_size,), (batch_size,) -> (batch_size,)
            return (p == h).type(torch.long)

        @GraphNode(leaf_nodes["p_obj_adj"], leaf_nodes["h_obj_adj"])
        def obj_adj(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (batch_size,), (batch_size,)  -> (batch_size, 2)
            return self._intersective_projection(p, h)

        @GraphNode(obj_adj, obj_noun)
        def obj(a: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            # (batch_size, 2), (batch_size,) -> (batch_size,)
            return torch.gather(a, 1, n.unsqueeze(1)).view(n.shape[0])

        @GraphNode(leaf_nodes["p_obj_q"], leaf_nodes["h_obj_q"])
        def vp_q(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (batch_size,), (batch_size,)  -> (batch_size, 4*7)
            return self._merge_quantifiers(p, h)

        @GraphNode(leaf_nodes["p_verb"], leaf_nodes["h_verb"])
        def v_verb(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (batch_size,), (batch_size,) -> (batch_size,)
            return (p == h).type(torch.long)

        @GraphNode(leaf_nodes["p_adv"], leaf_nodes["h_adv"])
        def v_adv(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (batch_size,), (batch_size,) -> (batch_size, 2)
            return self._intersective_projection(p, h)

        @GraphNode(v_adv, v_verb)
        def v_bar(a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            # (batch_size, 2), (batch_size,) -> (batch_size,)
            return torch.gather(a, 1, v.unsqueeze(1)).view(v.shape[0])

        @GraphNode(v_bar, vp_q, obj)
        def vp(v: torch.Tensor, q: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
            # (batch_size,), (batch_size, 4*7), (batch_size,) -> (batch_size,)
            idxs = (o * 7 + v).unsqueeze(1)  # note that o is the first argument
            return torch.gather(q, 1, idxs).view(v.shape[0])

        @GraphNode(leaf_nodes["p_neg"], leaf_nodes["h_neg"])
        def neg(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (batch_size,), (batch_size,) -> (batch_size, 7)
            return self._merge_negation(p, h)

        @GraphNode(neg, vp)
        def negp(n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            # (batch_size, 7), (batch_size,) -> (batch_size,)
            return torch.gather(n, 1, v.unsqueeze(1)).view(v.shape[0])

        @GraphNode(leaf_nodes["p_subj_noun"], leaf_nodes["h_subj_noun"])
        def subj_noun(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (batch_size,), (batch_size,) -> (batch_size,)
            return (p == h).type(torch.long)

        @GraphNode(leaf_nodes["p_subj_adj"], leaf_nodes["h_subj_adj"])
        def subj_adj(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (batch_size,), (batch_size,) -> (batch_size, 2)
            return self._intersective_projection(p, h)

        @GraphNode(subj_adj, subj_noun)
        def subj(a: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            # (batch_size, 2), (batch_size,) -> (batch_size,)
            return torch.gather(a, 1, n.unsqueeze(1)).view(n.shape[0])

        @GraphNode(leaf_nodes["p_subj_q"], leaf_nodes["h_subj_q"])
        def sentence_q(p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
            # (batch_size,), (batch_size,) -> (batch_size, 4*7)
            return self._merge_quantifiers(p, h)

        @GraphNode(sentence_q, subj, negp)
        def root(q: torch.Tensor, s: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            # (batch_size, 4*7), (batch_size,), (batch_size,) -> (batch_size,)
            idxs = (s * 7 + n).unsqueeze(1)
            res = torch.gather(q, 1, idxs).view(n.shape[0])
            res = self.output_remapping[res]
            return res

        super(Full_MQNLI_Logic_CompGraph, self).__init__(root)

    def get_input_node(self):
        return GraphNode.leaf("input")

    @property
    def emptystring(self):
        return self.keyword_dict["emptystring"]

    @property
    def no(self):
        return self.keyword_dict["no"]

    @property
    def some(self):
        return self.keyword_dict["some"]

    @property
    def every(self):
        return self.keyword_dict["every"]

    @property
    def notevery(self):
        return self.keyword_dict["notevery"]

    @property
    def doesnot(self):
        return self.keyword_dict["doesnot"]

    # def __getattr__(self, item):
    #     """ Used to retrieve keywords, relations or positions"""
    #     if item in self.keyword_dict:
    #         return self.keyword_dict[item]
    #     else:
    #         raise KeyError

    def _intersective_projection(self, p: torch.Tensor, h: torch.Tensor) \
            -> torch.Tensor:
        # (batch_size,), (batch_size,) -> (batch_size, 2)
        eq = (p == h)
        p_is_empty = (p == self.emptystring)
        h_is_empty = (h == self.emptystring)
        forward_entail = (~p_is_empty & h_is_empty)
        backward_entail = (p_is_empty & ~h_is_empty)

        res = torch.zeros(p.size(0), 2, dtype=torch.long, device=self.device)
        res[eq] = torch.tensor([INDEP, EQUIV], dtype=torch.long, device=self.device)
        res[forward_entail] = torch.tensor([INDEP, ENTAIL], dtype=torch.long, device=self.device)
        res[backward_entail] = torch.tensor([INDEP, REV_ENTAIL], dtype=torch.long, device=self.device)
        return res

    def _merge_quantifiers(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (batch_size,), (batch_size,) -> (batch_size, 4*7)
        p_idx_tensor = torch.zeros_like(p)
        h_idx_tensor = torch.zeros_like(h)
        for q_idx, q_token in enumerate([self.some, self.every, self.no, self.notevery]):
            p_idx_tensor[p == q_token] = q_idx
            h_idx_tensor[h == q_token] = q_idx
        idx_tensor = p_idx_tensor * 4 + h_idx_tensor
        return self.quantifier_signatures.index_select(0, idx_tensor)

    def _merge_negation(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (batch_size,), (batch_size,) -> (batch_size, 4, 7)
        p_idx_tensor = torch.zeros_like(p)
        h_idx_tensor = torch.zeros_like(h)
        for q_idx, q_token in enumerate([self.emptystring, self.doesnot]):
            p_idx_tensor[p == q_token] = q_idx
            h_idx_tensor[h == q_token] = q_idx
        idx_tensor = p_idx_tensor * 2 + h_idx_tensor
        return self.negation_signatures.index_select(0, idx_tensor)

class Full_Abstr_MQNLI_Logic_CompGraph(AbstractableCompGraph):
    def __init__(self, graph: Full_MQNLI_Logic_CompGraph,
                 intermediate_nodes: List[str]):
        super(Full_Abstr_MQNLI_Logic_CompGraph, self).__init__(
            graph=graph, abstract_nodes=intermediate_nodes
        )


"""Abstraction of deprecated `MQNLI_Logic_Compgraph`. This compgraph does not
have nodes for each individual token in the input. Use
`Full_MQNLI_Logic_CompGraph`  instead."""
class Abstr_MQNLI_Logic_CompGraph(OldAbstractableCompGraph):
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
            self.sentence_q, self.root
        ]}

        super(Abstr_MQNLI_Logic_CompGraph, self).__init__(
            full_graph=compgraph_structure,
            root_node_name="root",
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
        # (batch_size,), (batch_size,) -> (batch_size, 2)
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

    def _merge_quantifiers(self, p: torch.Tensor, h: torch.Tensor) \
            -> torch.Tensor:
        # (batch_size,), (batch_size,) -> (batch_size, 4*7)
        p_idx_tensor = torch.zeros(p.size(0), dtype=torch.long)
        h_idx_tensor = torch.zeros(h.size(0), dtype=torch.long)
        for q_idx, q_token in enumerate([self.some, self.every, self.no, self.notevery]):
            p_idx_tensor[p == q_token] = q_idx
            h_idx_tensor[h == q_token] = q_idx
        idx_tensor = p_idx_tensor * 4 + h_idx_tensor
        return quantifier_signatures.index_select(0, idx_tensor)

    def _merge_negation(self, p: torch.Tensor, h: torch.Tensor) \
            -> torch.Tensor:
        # (batch_size,), (batch_size,) -> (batch_size, 4, 7)
        p_idx_tensor = torch.zeros(p.size(0), dtype=torch.long)
        h_idx_tensor = torch.zeros(h.size(0), dtype=torch.long)
        for q_idx, q_token in enumerate([self.emptystring, self.doesnot]):
            p_idx_tensor[p == q_token] = q_idx
            h_idx_tensor[h == q_token] = q_idx
        idx_tensor = p_idx_tensor * 2 + h_idx_tensor
        return negation_signatures.index_select(0, idx_tensor)


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
        return torch.gather(a, 1, n.unsqueeze(1)).view(n.shape[0])

    def vp_q(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (9, batch_size), (9, batch_size) -> (batch_size, 4*7)
        return self._merge_quantifiers(p[IDX_Q_O], h[IDX_Q_O])

    def v_verb(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (9, batch_size), (9, batch_size) -> (batch_size,)
        return (p[IDX_V] == h[IDX_V]).type(torch.long)

    def v_adv(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (9, batch_size), (9, batch_size) -> (batch_size, 2)
        return self._intersective_projection(p[IDX_ADV], h[IDX_ADV])

    def v_bar(self, a: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # (batch_size, 2), (batch_size,) -> (batch_size,)
        return torch.gather(a, 1, v.unsqueeze(1)).view(v.shape[0])

    def vp(self, v: torch.Tensor, q: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
        # (batch_size,), (batch_size, 4*7), (batch_size,) -> (batch_size,)
        idxs = (o * 7 + v).unsqueeze(1) # note that o is the first argument
        return torch.gather(q, 1, idxs).view(v.shape[0])

    def neg(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (9, batch_size), (9, batch_size) -> (batch_size, 7)
        return self._merge_negation(p[IDX_NEG], h[IDX_NEG])

    def negp(self, n: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # (batch_size, 7), (batch_size,) -> (batch_size,)
        return torch.gather(n, 1, v.unsqueeze(1)).view(v.shape[0])

    def subj_noun(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (9, batch_size), (9, batch_size) -> (batch_size,)
        return (p[IDX_N_S] == h[IDX_N_S]).type(torch.long)

    def subj_adj(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (9, batch_size), (9, batch_size) -> (batch_size, 2)
        return self._intersective_projection(p[IDX_A_S], h[IDX_A_S])

    def subj(self, a: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        # (batch_size, 2), (batch_size,) -> (batch_size,)
        return torch.gather(a, 1, n.unsqueeze(1)).view(n.shape[0])

    def sentence_q(self, p: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        # (9, batch_size), (9, batch_size) -> (batch_size, 4*7)
        return self._merge_quantifiers(p[IDX_Q_S], h[IDX_Q_S])

    def root(self, q: torch.Tensor, s: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        # (batch_size, 4*7), (batch_size,), (batch_size,) -> (batch_size,)
        idxs = (s * 7 + n).unsqueeze(1)
        res = torch.gather(q, 1, idxs).view(n.shape[0])
        res = output_remapping[res]
        return res


def _generate_indices_fn(indices):
    def fn(x):
        return x[:, indices]
    return fn

def _generate_only_variable_fn(indices):
    def fn(x, y):
        x[:, indices] = y
        return x
    return fn


# indices_to_test = [torch.tensor([IDX_A_O, 9 + IDX_ADV]),
#                    torch.tensor([IDX_Q_O, 9 + IDX_A_S]),
#                    torch.tensor([IDX_A_S, 9 + IDX_N_O, IDX_NEG, IDX_Q_O]),
#                    torch.tensor([IDX_N_S, 9 + IDX_N_S, IDX_NEG, 9 + IDX_V]),
#                    torch.tensor([IDX_Q_O, 9 + IDX_NEG, IDX_V, IDX_Q_S, 9 + IDX_Q_O, IDX_N_S, IDX_NEG])]

indices_to_test = [torch.tensor([11, 20]),
                   torch.tensor([9,15]),
                   torch.tensor([3, 25,6,10]),
                   torch.tensor([4,17,6,21]),
                   torch.tensor([9,10,22,23,4,6])]

new_indices = [torch.tensor([0,8]), torch.tensor([5,1]), torch.tensor([10,17]),torch.tensor([8,15])]

new_indices2 = [torch.tensor(x) for x in itertools.combinations([7,8,16,17], 2)]
new_indices2 += [torch.tensor(x) for x in itertools.combinations([7,8,16,17], 3)]

new_indices3 += [[torch.tensor([7,8,16,17] + [x]) for x in [1,2,3,4,5,6,9,10,14,15]]

new_indices4 = [torch.tensor(x) for x in itertools.combinations([0,3,6,9,12,15], 2)]


class Random_MQNLI_Logic_CompGraph(Full_MQNLI_Logic_CompGraph):
    def __init__(self, random_node_idxs: torch.tensor, data: MQNLIData, device=None):
        self.random_node_subset_idxs = random_node_idxs
        super(Random_MQNLI_Logic_CompGraph, self).__init__(data, device)

    def get_input_node(self):
        input_node = GraphNode.leaf("input")
        fn = _generate_indices_fn(self.random_node_subset_idxs)
        only_variable = GraphNode(input_node, name="random_node_subset", forward=fn)
        fn = _generate_only_variable_fn(self.random_node_subset_idxs)
        input_node = GraphNode(input_node, only_variable, name="input2", forward=fn)
        return input_node


class Random_Abstr_MQNLI_Logic_CompGraph(AbstractableCompGraph):
    def __init__(self, graph: Random_MQNLI_Logic_CompGraph):
        super(Random_Abstr_MQNLI_Logic_CompGraph, self).__init__(
            graph=graph, abstract_nodes=["random_node_subset"]
        )
