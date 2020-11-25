from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph
from intervention import GraphInput
from trainer import load_model
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

@pytest.fixture
def mqnli_data():
    return MQNLIData("../mqnli_data/mqnli.train.txt",
                     "../mqnli_data/mqnli.dev.txt",
                     "../mqnli_data/mqnli.test.txt")

@pytest.fixture
def mqnli_lstm_model():
    model, _ = load_model(LSTMModule, "../mqnli_models/lstm_best.pt")
    model.eval()
    return model

@pytest.fixture
def mqnli_lstm_comp_graph(mqnli_lstm_model):
    return MQNLI_LSTM_CompGraph(mqnli_lstm_model)


def has_only_child(node, child_name):
    assert len(node.children_dict) == 1 and child_name in node.children_dict

def has_children(node, child_names):
    assert len(node.children_dict) == len(child_names)
    assert all(child_name in node.children_dict for child_name in child_names)

def test_lstm_compgraph_structure(mqnli_lstm_comp_graph):
    graph = mqnli_lstm_comp_graph
    has_only_child(graph.root, "logits")
    has_only_child(graph.nodes["logits"], "activation2")
    has_only_child(graph.nodes["activation2"], "ffnn2")
    has_only_child(graph.nodes["ffnn2"], "activation1")
    has_only_child(graph.nodes["activation1"], "ffnn1")
    has_only_child(graph.nodes["ffnn1"], "concat_final_state")
    has_children(graph.nodes["concat_final_state"], ["premise_lstm_3", "hypothesis_lstm_3"])
    has_only_child(graph.nodes["premise_lstm_3"], "premise_lstm_2")
    has_only_child(graph.nodes["premise_lstm_2"], "premise_lstm_1")
    has_only_child(graph.nodes["premise_lstm_1"], "premise_lstm_0")
    has_only_child(graph.nodes["premise_lstm_0"], "premise_emb")
    has_only_child(graph.nodes["premise_emb"], "embed")
    has_only_child(graph.nodes["hypothesis_lstm_3"], "hypothesis_lstm_2")
    has_only_child(graph.nodes["hypothesis_lstm_2"], "hypothesis_lstm_1")
    has_only_child(graph.nodes["hypothesis_lstm_1"], "hypothesis_lstm_0")
    has_only_child(graph.nodes["hypothesis_lstm_0"], "hypothesis_emb")
    has_only_child(graph.nodes["hypothesis_emb"], "embed")
    has_only_child(graph.nodes["embed"], "input")


def test_lstm_compgraph_batch_match(mqnli_data, mqnli_lstm_model, mqnli_lstm_comp_graph):
    model = mqnli_lstm_model
    graph = mqnli_lstm_comp_graph
    with torch.no_grad():
        collate_fn = lambda batch: my_collate(batch, batch_first=False)
        dataloader = DataLoader(mqnli_data.dev, batch_size=1024, shuffle=False,
                                collate_fn=collate_fn)
        for i, input_tuple in enumerate(dataloader):
            input_tuple = [x.to(model.device) for x in input_tuple]

            graph_input = GraphInput({"input": input_tuple[0]})
            graph_pred = graph.compute(graph_input, store_cache=True)

            logits = model(input_tuple)
            model_pred = torch.argmax(logits, dim=1)


            assert torch.all(model_pred == graph_pred), f"on batch {i}"

def test_lstm_compgraph_single_match(mqnli_data, mqnli_lstm_model, mqnli_lstm_comp_graph):
    model = mqnli_lstm_model
    graph = mqnli_lstm_comp_graph

    with torch.no_grad():
        collate_fn = lambda batch: my_collate(batch, batch_first=False)
        dataloader = DataLoader(mqnli_data.dev, batch_size=1, shuffle=False,
                                collate_fn=collate_fn)
        for i, input_tuple in enumerate(dataloader):
            graph_input = GraphInput({"input": mqnli_data.dev[i][0].to(model.device)})
            graph_pred = graph.compute(graph_input, store_cache=True)

            logits = model(input_tuple)
            model_pred = torch.argmax(logits, dim=1)

            assert torch.all(model_pred == graph_pred), f"result mismatch on batch {i}"
            if i == 100:
                break


def g_has_children(g, node, child_names):
    node = g.nodes[node]
    if isinstance(child_names, list):
        assert len(node.children) == len(child_names)
        assert all(child_name in node.children_dict for child_name in child_names)
    else:
        assert len(node.children) == 1 and child_names in node.children_dict


def test_lstm_abstractable(mqnli_data, mqnli_lstm_model, mqnli_lstm_comp_graph):
    model = mqnli_lstm_model
    graph = Abstr_MQNLI_LSTM_CompGraph(mqnli_lstm_comp_graph,
                                       ["premise_lstm_0"])
    g_has_children(graph, "root", ["premise_lstm_0", "input"])
    g_has_children(graph, "premise_lstm_0", "input")


    with torch.no_grad():
        collate_fn = lambda batch: my_collate(batch, batch_first=False)
        dataloader = DataLoader(mqnli_data.dev, batch_size=1024, shuffle=False,
                                collate_fn=collate_fn)
        for i, input_tuple in enumerate(dataloader):
            input_tuple = [x.to(model.device) for x in input_tuple]

            graph_input = GraphInput({"input": input_tuple[0]})
            graph_pred = graph.compute(graph_input, store_cache=True)

            logits = model(input_tuple)
            model_pred = torch.argmax(logits, dim=1)

            assert torch.all(model_pred == graph_pred), f"on batch {i}"



@pytest.fixture
def mqnli_sep_data():
    return MQNLIData("../mqnli_data/mqnli.train.txt",
                     "../mqnli_data/mqnli.dev.txt",
                     "../mqnli_data/mqnli.test.txt",
                     use_separator=True)

@pytest.fixture
def mqnli_lstm_sep_model():
    model, _ = load_model(LSTMModule, "../mqnli_models/lstm_sep_best.pt")
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

            graph_input = GraphInput({"input": input_tuple[0]})
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

            graph_input = GraphInput({"input": input_tuple[0]})
            graph_pred = graph.compute(graph_input, store_cache=True)

            logits = model(input_tuple)
            model_pred = torch.argmax(logits, dim=1)

            assert torch.all(model_pred == graph_pred), f"on batch {i}"