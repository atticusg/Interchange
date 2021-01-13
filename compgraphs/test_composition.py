from compgraphs.mqnli_logic import negation_signatures as vectorized_neg_signatures
from compgraphs.mqnli_logic import relation_composition as vectorized_relation_composition
from compgraphs.mqnli_logic import quantifier_signatures as vectorized_determiner_signatures
from compgraphs.mqnli_logic import Abstr_MQNLI_Logic_CompGraph, compgraph_structure
from intervention import GraphInput
from datasets.mqnli import MQNLIData

from mqnli.natural_logic_model import relation_composition, negation_merge, \
    determiner_signatures, get_label, compute_simple_relation_gentest

idx2rln = ["independence", "equivalence", "entails", "reverse entails", "contradiction", "alternation", "cover"]
rln2idx = {r: i for i, r in enumerate(idx2rln)}

def test_relation_composition():
    for r_1 in range(7):
        for r_2 in range(7):
            r_1_str = idx2rln[r_1]
            r_2_str = idx2rln[r_2]
            res = vectorized_relation_composition[r_1, r_2]
            res_str = idx2rln[res]
            assert relation_composition[(r_1_str, r_2_str)] == res_str

def test_negation_signatures():
    for p_neg in [False, True]:
        for h_neg in [False, True]:
            sig = negation_merge(p_neg, h_neg)
            p_i = 1 if p_neg else 0
            h_i = 1 if h_neg else 0
            vectorized_sig = vectorized_neg_signatures[p_i*2 + h_i].tolist()
            for k, v in enumerate(vectorized_sig):
                key_str = idx2rln[k]
                val_str = idx2rln[v]
                assert sig[key_str] == val_str


def test_determiner_signatures():
    got_sigs = vectorized_determiner_signatures
    for neg_1, neg_1_bool in enumerate([False, True]):
        for neg_2, neg_2_bool in enumerate([False, True]):
            neg_sig = negation_merge(neg_1_bool, neg_2_bool)
            for det_1, det_1_str in enumerate(["some", "every"]):
                for det_2, det_2_str in enumerate(["some", "every"]):
                    det_sigs = determiner_signatures[(det_1_str, det_2_str)]
                    for rln_1, rln_1_str in enumerate(idx2rln[:4]):
                        for rln_2, rln_2_str in enumerate(idx2rln):
                            got = got_sigs[4*(det_1+neg_1*2) + det_2+neg_2*2,
                                           rln_1*7 + rln_2]
                            got_str = idx2rln[got]

                            exp_str = det_sigs[(rln_1_str, rln_2_str)]
                            exp_str = neg_sig[exp_str]

                            assert got_str == exp_str


mqnli_mini_data = MQNLIData("../data/mqnli/raw/easy_mini/train.txt",
                     "../data/mqnli/raw/easy_mini/dev.txt",
                     "../data/mqnli/raw/easy_mini/test.txt", store_text=True)

examples = mqnli_mini_data.dev[:20][0]
labels = mqnli_mini_data.dev[:20][1]
example_5batch = examples.transpose(0,1)

graph = Abstr_MQNLI_Logic_CompGraph(mqnli_mini_data, list(compgraph_structure.keys()))

IDX_Q_S, IDX_A_S, IDX_N_S, IDX_NEG, IDX_ADV, IDX_V, \
        IDX_Q_O, IDX_A_O, IDX_N_O = range(9)


class Example:
    def __init__(self, ex):
        self.subject_negation = ex[IDX_Q_S].item() in {graph.no, graph.notevery}
        self.natlog_subject_determiner = "some" \
            if ex[IDX_Q_S].item() in {graph.some, graph.no} else "every"
        self.subject_noun = ex[IDX_N_S].item()
        self.subject_adjective = ex[IDX_A_S].item()
        self.verb_negation = ex[IDX_NEG].item()
        self.verb = ex[IDX_V].item()
        self.adverb = ex[IDX_ADV].item()
        self.object_negation = ex[IDX_Q_O].item() in {graph.no, graph.notevery}
        self.natlog_object_determiner = "some" \
            if ex[IDX_Q_O].item() in {graph.some, graph.no} else "every"
        self.object_noun = ex[IDX_N_O].item()
        self.object_adjective = ex[IDX_A_O].item()

idx2labels = ["neutral", "entailment", "contradiction"]


def test_composition_step_by_step():
    i = GraphInput({"input": example_5batch})
    graph_res = graph.compute(i)

    exs = [(Example(examples[i, :9]), Example(examples[i, 9:])) for i in range(20)]

    got = [get_label(compute_simple_relation_gentest(p, h)[0]) for p, h in exs]
    expected = [idx2labels[l] for l in labels]
    graph_res = [idx2labels[l] for l in graph_res]
    for i, (g, gr, e) in enumerate(zip(got, graph_res, expected)):
        print(f"{i} got:", g, "graph_res:", gr, "expected:", e)


