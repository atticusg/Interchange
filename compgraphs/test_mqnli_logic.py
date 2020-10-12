import pytest
import torch
import time
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph

from mqnli.make_subphrase_labels import get_intermediate_labels

from intervention import GraphInput
from datasets.utils import my_collate
from datasets.mqnli import MQNLIData
from torch.utils.data import DataLoader


mqnli_mini_data = MQNLIData("../mqnli_data/mini.train.txt",
                     "../mqnli_data/mini.dev.txt",
                     "../mqnli_data/mini.test.txt", store_text=True)



# dummy_compgraph = MQNLI_Logic_CompGraph(mqnli_mini_data)

node_name_test_set = [
    (["negp"], ["input", "negp",  "sentence"]),
    (["negp", "vp"], ["input", "vp", "negp", "sentence"]),
    (["vp", "negp"], ["input", "vp", "negp", "sentence"]),
    (["subj", "negp"], ["input", "negp", "subj", "sentence"]),
    (["subj_adj", "subj_noun", "negp"], ["input", "negp", "subj_noun", "subj_adj",  "sentence"])
]

@pytest.mark.parametrize("intermediate_nodes, expected",node_name_test_set)
def test_get_node_names(intermediate_nodes, expected):
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    res = g.get_node_names(intermediate_nodes)
    assert all(res_name == expected_name for res_name, expected_name in zip(res, expected)), \
        f"Got {res}, expected {expected}"

"""
get_children_test_set = [
    ("sentence", {"input",  "sentence"}, ["input"]),
    ("sentence", {"input", "negp",  "sentence"}, ["input", "negp"]),
    ("negp", {"input", "negp",  "sentence"}, ["input"]),
    ("negp", {"input", "vp", "negp",  "sentence"}, ["input", "vp"]),
    ("sentence", {"input", "neg", "subj", "sentence"}, ["input", "subj", "neg"]),
    ("sentence", {"input", "negp", "subj", "sentence"}, ["input", "subj", "negp"]),
    ("sentence", {"input", "sentence_q", "subj", "sentence"}, ["sentence_q", "subj", "input"]),
    ("negp", {"input", "vp", "neg", "sentence"}, ["neg", "vp"])
]

@pytest.mark.parametrize("node, node_set, expected", get_children_test_set)
def test_get_children(node, node_set, expected):
    res = dummy_compgraph.get_children(node, node_set)
    assert all(res_name == expected_name for res_name, expected_name in
               zip(res, expected)), \
        f"Got {res}, expected {expected}"
"""


def has_children(g, node, child_names):
    node = g.nodes[node]
    if isinstance(child_names, list):
        assert len(node.children) == len(child_names)
        assert all(child_name in node.children_dict for child_name in child_names)
    else:
        assert len(node.children) == 1 and child_names in node.children_dict

def test_graph_structure1():
    intermediate_nodes = ["vp"]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    has_children(g, "sentence", ["input", "vp"])
    has_children(g, "vp", "input")


def test_graph_structure2():
    intermediate_nodes = ["vp", "neg"]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    has_children(g, "sentence", ["input", "neg", "vp"])
    has_children(g, "neg", "input")
    has_children(g, "vp", "input")

def test_graph_structure3():
    intermediate_nodes = ["negp", "vp"]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    has_children(g, "sentence", ["input", "negp"])
    has_children(g, "negp", ["input", "vp"])
    has_children(g, "vp", "input")

def test_graph_run_without_errors():
    intermediate_nodes = ["vp"]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    i = GraphInput({"input": torch.zeros(18, 10)})
    res = g.compute(i)
    """
    sentence(
        sentence_q(
            get_p(<input>), get_h(<input>)
        ), 
        subj(
            subj_adj(get_p(<input>), get_h(<input>)), 
            subj_noun(get_p(<input>), get_h(<input>))
        ), 
        negp(
            neg(get_p(<input>), get_h(<input>)), 
            vp(
                v_bar(
                    v_adv(get_p(<input>), get_h(<input>)), 
                    v_verb(get_p(<input>), get_h(<input>))
                ), 
                vp_q(get_p(<input>), get_h(<input>)), 
                obj(
                    obj_adj(get_p(<input>), get_h(<input>)), 
                    obj_noun(get_p(<input>), get_h(<input>))
                )
            )
        )
    )
"""

def get_example_data(data, batch_size):
    example_strs = data.dev.example_text[:batch_size]
    return [get_intermediate_labels(s) for s in example_strs]


batch_size = 5
examples = mqnli_mini_data.dev[:batch_size][0]
labels = mqnli_mini_data.dev[:batch_size][1]
example_batch = examples.transpose(0, 1)
example_data = get_example_data(mqnli_mini_data, batch_size)

INDEP, EQUIV, ENTAIL, REV_ENTAIL, CONTRADICT, ALTER, COVER = range(7)

SOME, EVERY, NO, NOTEVERY = range(4)

dataidx2node = ["subj_adj", "subj_noun", "v_adv", "v_verb", "obj_adj", "obj_noun",
                "subj", "v_bar", "obj", "vp", "negp", "sentence"]
node2dataidx = {k: i for i, k in enumerate(dataidx2node)}
idx2rln = ["independence", "equivalence", "entails", "reverse entails", "contradiction", "alternation", "cover"]
rln2idx = {r: i for i, r in enumerate(idx2rln)}

DATA_A_S, DATA_N_S, DATA_ADV, DATA_V, DATA_A_O, DATA_N_O, \
    DATA_SUBJ, DATA_VBAR, DATA_OBJ, DATA_VP, DATA_NEGP, DATA_SENT = range(12)

@pytest.mark.parametrize("node", ["vp", "negp"])
def test_intermediate_values(node):
    for i, ex in enumerate(examples):
        words = mqnli_mini_data.decode(ex)
        print(f"\n{i} p: {words[:9]}")
        print(f"{i} h: {words[9:]}")
        print(f"{i} data: {example_data[i]}")
        print(f"{i} label: {labels[i]}")

    i = GraphInput({"input": example_batch})

    intermediate_nodes = [node]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    g.compute(i)
    got = g.get_result(node, i)

    idx_in_data = node2dataidx[node]

    expected = [rln2idx[example_data[i]['gold_label'][idx_in_data]] for i in range(batch_size)]
    expected = torch.tensor(expected, dtype=torch.long)
    assert torch.all(expected == got)

def test_final_value():
    intermediate_nodes = ["vp"]
    i = GraphInput({"input": example_batch})
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    res = g.compute(i)
    print("res", res)
    print("labels", labels)
    assert all(r == l for r, l in zip(res.tolist(), labels))

######### old tests ##########

example_0 = mqnli_mini_data.dev[0]
example_1 = mqnli_mini_data.dev[1]
example_2 = mqnli_mini_data.dev[2]
example_0_str = mqnli_mini_data.decode(example_0[0])
example_1_str = mqnli_mini_data.decode(example_1[0])
example_2_str = mqnli_mini_data.decode(example_2[0])

example_3batch = torch.stack((example_0[0], example_1[0], example_2[0]), dim=1)


def test_get_p_h():
    intermediate_nodes = ["get_p", "get_h"]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)

    i = GraphInput({"input": example_3batch})
    g.compute(i)

    p = g.get_result("get_p", i)
    h = g.get_result("get_h", i)

    assert torch.all(p == example_3batch[:9, :])
    assert torch.all(h == example_3batch[9:, :])
    # i = GraphInput({})

test_set = [("obj_noun", torch.tensor([1,1,0,1,1])),
            ("obj_adj", torch.tensor([[INDEP, ENTAIL],
                             [INDEP, ENTAIL],
                             [INDEP, EQUIV],
                             [INDEP, EQUIV],
                             [INDEP, REV_ENTAIL]])),
            ("obj", torch.tensor([ENTAIL, ENTAIL, INDEP, EQUIV, REV_ENTAIL])),
            ("subj", torch.tensor([EQUIV, EQUIV, REV_ENTAIL, ENTAIL, ENTAIL])),
            ("v_bar", torch.tensor([REV_ENTAIL, EQUIV, INDEP, REV_ENTAIL, ENTAIL]))]

@pytest.mark.parametrize("node,expected", test_set)
def test_nodes(node, expected):
    intermediate_nodes = [node]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    i = GraphInput({"input": example_batch})
    g.compute(i)
    res = g.get_result(node, i)
    assert torch.all(res == expected)


def test_object_quantifier_signatures():
    intermediate_nodes = ["vp_q"]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    i = GraphInput({"input": example_batch})
    g.compute(i)
    res = g.get_result("vp_q", i)
    assert res.shape == (batch_size, 4*7)
    exp_idx = torch.tensor([NO*4 + NOTEVERY,
                            NOTEVERY*4 + NO,
                            NOTEVERY*4 + SOME,
                            EVERY*4 + NOTEVERY,
                            NOTEVERY*4 + EVERY], dtype=torch.long)
    assert exp_idx.shape == (batch_size,)
    exp = quantifier_signatures.index_select(0, exp_idx)
    assert exp.shape == (batch_size, 4*7)
    assert torch.all(res == exp)

def test_vp_shape():
    intermediate_nodes = ["vp"]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    i = GraphInput({"input": example_batch})
    g.compute(i)
    got = g.get_result("vp", i)

    assert got.shape == (5,)
    print(got)

def test_negp_shape():
    intermediate_nodes = ["negp"]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    i = GraphInput({"input": example_batch})
    g.compute(i)
    got = g.get_result("negp", i)

    assert got.shape == (batch_size,)
    print(got)


def test_neg_signatures():
    intermediate_nodes = ["neg"]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    i = GraphInput({"input": example_batch})
    g.compute(i)
    res = g.get_result("neg", i)
    assert res.shape == (batch_size, 7)
    DOESNOT = 1
    E = 0
    exp_idx = torch.tensor([DOESNOT*2 + E,
                            DOESNOT*2 + DOESNOT,
                            E*2 + DOESNOT,
                            E*2 + E,
                            DOESNOT*2 + DOESNOT], dtype=torch.long)
    assert exp_idx.shape == (batch_size,)
    exp = negation_signatures.index_select(0, exp_idx)
    assert exp.shape == (batch_size, 7)
    assert torch.all(res == exp)

def test_subject_quantifier_signatures():
    intermediate_nodes = ["sentence_q"]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    i = GraphInput({"input": example_batch})
    g.compute(i)
    res = g.get_result("sentence_q", i)
    assert res.shape == (batch_size, 4*7)
    exp_idx = torch.tensor([NO*4 + NO,
                            NO*4 + NOTEVERY,
                            NO*4 + EVERY,
                            NOTEVERY*4 + NO,
                            SOME*4 + NOTEVERY], dtype=torch.long)
    assert exp_idx.shape == (batch_size,)
    exp = quantifier_signatures.index_select(0, exp_idx)
    assert exp.shape == (batch_size, 4*7)
    assert torch.all(res == exp)


@pytest.fixture
def mqnli_data():
    return MQNLIData("../mqnli_data/mqnli.train.txt",
                     "../mqnli_data/mqnli.dev.txt",
                     "../mqnli_data/mqnli.test.txt")


def test_whole_graph(mqnli_data):
    intermediate_nodes = ["subj", "negp", "vp", "v_bar", "obj"]
    graph = MQNLI_Logic_CompGraph(mqnli_data, intermediate_nodes)
    with torch.no_grad():
        collate_fn = lambda batch: my_collate(batch, batch_first=False)
        dataloader = DataLoader(mqnli_data.train, batch_size=2048, shuffle=False,
                                collate_fn=collate_fn)
        start_time = time.time()
        for i, input_tuple in enumerate(dataloader):
            graph_input = GraphInput({"input": input_tuple[0]})
            graph_pred = graph.compute(graph_input, store_cache=True)
            labels = input_tuple[1]

            assert torch.all(labels == graph_pred), f"on batch {i}"
        duration = time.time() - start_time
        print(f"---- Ran {len(mqnli_data.train)} examples in {duration:.2f} s ----")


def test_whole_graph_single_batch():
    intermediate_nodes = ["subj", "negp", "vp", "v_bar", "obj"]
    graph = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    with torch.no_grad():
        collate_fn = lambda batch: my_collate(batch, batch_first=False)
        dataloader = DataLoader(mqnli_mini_data.train, batch_size=1, shuffle=False,
                                collate_fn=collate_fn)
        start_time = time.time()
        for i, input_tuple in enumerate(dataloader):
            input_tensor = input_tuple[0]
            graph_input = GraphInput({"input": input_tuple[0]})
            graph_pred = graph.compute(graph_input, store_cache=True)
            labels = input_tuple[1]

            assert torch.all(labels == graph_pred), f"on batch {i}"
        duration = time.time() - start_time
        print(f"---- Ran {len(mqnli_mini_data.train)} examples in {duration:.2f} s ----")