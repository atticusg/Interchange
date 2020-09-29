import pytest
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph
from intervention import GraphInput
from datasets.mqnli import MQNLIData


mqnli_mini_data = MQNLIData("../mqnli_data/mini.train.txt",
                     "../mqnli_data/mini.dev.txt",
                     "../mqnli_data/mini.test.txt")


dummy_compgraph = MQNLI_Logic_CompGraph(mqnli_mini_data, [])

node_name_test_set = [
    (["negp"], ["input", "negp",  "sentence"]),
    (["negp", "vp"], ["input", "vp", "negp", "sentence"]),
    (["vp", "negp"], ["input", "vp", "negp", "sentence"]),
    (["subj", "negp"], ["input", "negp", "subj", "sentence"]),
    (["subj_adj", "subj_noun", "negp"], ["input",  "subj_noun", "subj_adj", "negp",  "sentence"])
]

@pytest.mark.parametrize("intermediate_nodes, expected",node_name_test_set)
def test_get_node_names(intermediate_nodes, expected):
    res = dummy_compgraph.get_node_names(intermediate_nodes)
    assert all(res_name == expected_name for res_name, expected_name in zip(res, expected)), \
        f"Got {res}, expected {expected}"

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

def test_graph_value():
    intermediate_nodes = ["vp"]
    g = MQNLI_Logic_CompGraph(mqnli_mini_data, intermediate_nodes)
    i = GraphInput({"input": "<input>"})
    res = g.compute(i)
    assert g.get_result("vp", i) == """vp(v_bar(v_adv(get_p(<input>), get_h(<input>)), v_verb(get_p(<input>), get_h(<input>))), vp_q(get_p(<input>), get_h(<input>)), obj(obj_adj(get_p(<input>), get_h(<input>)), obj_noun(get_p(<input>), get_h(<input>))))"""
    print(res)
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
