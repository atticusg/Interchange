from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph
from intervention import GraphInput
from modeling.utils import load_model
from modeling.lstm import LSTMModule
from datasets.utils import my_collate
from datasets.mqnli import MQNLIData

import torch
from torch.utils.data import DataLoader
import pytest
from tqdm import tqdm

def test_func_defs():
    funcs = []

    for i in range(5):
        def dynamic_func():
            print(f"I am func number {i}")
            return i
        funcs.append(dynamic_func)

    for i, f in enumerate(funcs):
        assert f() == i


def has_only_child(node, child_name):
    assert len(node.children_dict) == 1 and child_name in node.children_dict

def has_children(node, child_names):
    assert len(node.children_dict) == len(child_names)
    assert all(child_name in node.children_dict for child_name in child_names)


def g_has_children(g, node, child_names):
    node = g.nodes[node]
    if isinstance(child_names, list):
        assert len(node.children) == len(child_names)
        assert all(child_name in node.children_dict for child_name in child_names)
    else:
        assert len(node.children) == 1 and child_names in node.children_dict


@pytest.fixture
def mqnli_sep_data():
    return torch.load("../data/mqnli/preprocessed/lstm-easy.pt")

@pytest.fixture
def mqnli_lstm_sep_model():
    model, _ = load_model(LSTMModule, "../data/models/lstm-easy-best.pt")
    model.eval()
    return model

@pytest.fixture
def mqnli_lstm_sep_comp_graph(mqnli_lstm_sep_model):
    return MQNLI_LSTM_CompGraph(mqnli_lstm_sep_model)

def test_lstm_sep_compgraph_batch_match(mqnli_sep_data, mqnli_lstm_sep_model,
                                        mqnli_lstm_sep_comp_graph):
    model = mqnli_lstm_sep_model
    graph = mqnli_lstm_sep_comp_graph
    with torch.no_grad():
        collate_fn = lambda batch: my_collate(batch, batch_first=False)
        dataloader = DataLoader(mqnli_sep_data.dev, batch_size=1024, shuffle=False,
                                collate_fn=collate_fn)
        for i, input_tuple in enumerate(dataloader):
            input_tuple = [x.to(model.device) for x in input_tuple]

            graph_input = GraphInput({"input": input_tuple})
            graph_pred = graph.compute(graph_input, store_cache=True)

            logits = model(input_tuple)
            model_pred = torch.argmax(logits, dim=1)


            assert torch.all(model_pred == graph_pred), f"on batch {i}"


def test_lstm_sep_abstractable(mqnli_sep_data, mqnli_lstm_sep_model,
                               mqnli_lstm_sep_comp_graph):
    model = mqnli_lstm_sep_model
    graph = Abstr_MQNLI_LSTM_CompGraph(mqnli_lstm_sep_comp_graph,
                                       ["lstm_0"])
    g_has_children(graph, "root", "lstm_0")
    g_has_children(graph, "lstm_0", "input")


    with torch.no_grad():
        collate_fn = lambda batch: my_collate(batch, batch_first=False)
        dataloader = DataLoader(mqnli_sep_data.dev, batch_size=1024, shuffle=False,
                                collate_fn=collate_fn)
        for i, input_tuple in enumerate(dataloader):
            input_tuple = [x.to(model.device) for x in input_tuple]

            graph_input = GraphInput({"input": input_tuple})
            graph_pred = graph.compute(graph_input, store_cache=True)

            logits = model(input_tuple)
            model_pred = torch.argmax(logits, dim=1)

            assert torch.all(model_pred == graph_pred), f"on batch {i}"