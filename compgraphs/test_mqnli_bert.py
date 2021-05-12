import torch
from torch.utils.data import DataLoader
import pytest
from tqdm import tqdm

from compgraphs.mqnli_bert import MQNLI_Bert_CompGraph, Full_MQNLI_Bert_CompGraph
from compgraphs.mqnli_bert import Abstr_MQNLI_Bert_CompGraph
from antra import GraphInput
from modeling.utils import load_model
from modeling.pretrained_bert import PretrainedBertModule

@pytest.fixture
def mqnli_data():
    return torch.load("../data/mqnli/preprocessed/bert-easy.pt")

@pytest.fixture
def mqnli_bert_model():
    opts = {"tokenizer_vocab_path": "../data/tokenization/bert-vocab.txt"}
    model, _ = load_model(PretrainedBertModule, "../data/models/bert-easy-best.pt", opts=opts)
    # print(f"Loading model, {type(model)=}")
    model.eval()
    return model

@pytest.fixture
def mqnli_bert_compgraph(mqnli_bert_model):
    return MQNLI_Bert_CompGraph(mqnli_bert_model)


@pytest.fixture
def full_mqnli_bert_compgraph(mqnli_bert_model):
    return Full_MQNLI_Bert_CompGraph(mqnli_bert_model)

def has_only_child(node, child_name):
    assert len(node.children_dict) == 1 and child_name in node.children_dict

def has_children(node, child_names):
    assert len(node.children_dict) == len(child_names)
    assert all(child_name in node.children_dict for child_name in child_names)

def test_lstm_compgraph_structure(mqnli_bert_compgraph):
    graph = mqnli_bert_compgraph
    has_only_child(graph.root, "logits")
    has_only_child(graph.nodes["logits"], "pool")
    has_only_child(graph.nodes["pool"], "bert_layer_11")
    num_layers = len(mqnli_bert_compgraph.model.bert.encoder.layer)
    for layer_i in range(num_layers-1, 0, -1):
        has_children(graph.nodes[f"bert_layer_{layer_i}"], [f"bert_layer_{layer_i-1}", "metainfo"])

    has_children(graph.nodes["bert_layer_0"], ["embed", "metainfo"])
    has_only_child(graph.nodes["embed"], "input")
    has_only_child(graph.nodes["metainfo"], "input")


def test_bert_compgraph_batch_match(mqnli_data, mqnli_bert_model, mqnli_bert_compgraph):
    model = mqnli_bert_model
    graph = mqnli_bert_compgraph
    with torch.no_grad():
        dataloader = DataLoader(mqnli_data.dev, batch_size=32, shuffle=False)
        for i, input_tuple in enumerate(tqdm(dataloader, total=100)):
            input_tuple = [x.to(model.device) for x in input_tuple]

            graph_input = GraphInput({"input": input_tuple}, cache_results=False)
            graph_pred = graph.compute(graph_input)

            logits = model(input_tuple)
            model_pred = torch.argmax(logits, dim=1)

            assert torch.all(model_pred == graph_pred), f"on batch {i}"
            if i == 100:
                break

def test_bert_compgraph_single_match(mqnli_data, mqnli_bert_model, mqnli_bert_compgraph):
    model = mqnli_bert_model
    graph = mqnli_bert_compgraph
    with torch.no_grad():
        dataloader = DataLoader(mqnli_data.dev, batch_size=1, shuffle=False)
        for i, input_tuple in enumerate(tqdm(dataloader, total=100)):
            input_tuple = [x.to(model.device) for x in input_tuple]

            graph_input = GraphInput({"input": input_tuple}, cache_results=False)
            graph_pred = graph.compute(graph_input)

            logits = model(input_tuple)
            model_pred = torch.argmax(logits, dim=1)

            assert torch.all(model_pred == graph_pred), f"on batch {i}"
            if i == 100:
                break


def g_has_children(g, node, child_names):
    node = g.nodes[node]
    if isinstance(child_names, list):
        assert len(node.children) == len(child_names)
        assert all(child_name in node.children_dict for child_name in child_names)
    else:
        assert len(node.children) == 1 and child_names in node.children_dict

def test_bert_abstractable(mqnli_data, mqnli_bert_model, mqnli_bert_compgraph):
    model = mqnli_bert_model
    g = Abstr_MQNLI_Bert_CompGraph(mqnli_bert_compgraph, ["bert_layer_0"])
    g_has_children(g, "root", ["bert_layer_0", "input"])
    g_has_children(g, "bert_layer_0", ["input"])

    with torch.no_grad():
        dataloader = DataLoader(mqnli_data.dev, batch_size=1, shuffle=False)
        for i, input_tuple in enumerate(tqdm(dataloader, total=100)):
            input_tuple = [x.to(model.device) for x in input_tuple]

            graph_input = GraphInput({"input": input_tuple}, cache_results=False)
            graph_pred = g.compute(graph_input)

            logits = model(input_tuple)
            model_pred = torch.argmax(logits, dim=1)

            assert torch.all(model_pred == graph_pred), f"on batch {i}"
            if i == 100:
                break


def test_full_bert_compgraph_structure(full_mqnli_bert_compgraph):
    graph = full_mqnli_bert_compgraph
    has_only_child(graph.root, "logits")
    has_only_child(graph.nodes["logits"], "pool")
    has_only_child(graph.nodes["pool"], "bert_layer_11")
    num_layers = len(graph.model.bert.encoder.layer)
    for layer_i in range(num_layers-1, 0, -1):
        has_children(graph.nodes[f"bert_layer_{layer_i}"], [f"bert_layer_{layer_i-1}", "input_preparation"])

    has_children(graph.nodes["bert_layer_0"], ["embed", "input_preparation"])
    # has_only_child(graph.nodes["embed"], "input")
    # has_only_child(graph.nodes["metainfo"], "input")


def test_full_bert_compgraph_batch_match(mqnli_data, mqnli_bert_model, full_mqnli_bert_compgraph):
    model = mqnli_bert_model
    graph = full_mqnli_bert_compgraph
    with torch.no_grad():
        dataloader = DataLoader(mqnli_data.dev, batch_size=32, shuffle=False)
        for i, input_tuple in enumerate(tqdm(dataloader, total=100)):
            input_dict = {
                "input_ids": input_tuple[0].to(model.device),
                "token_type_ids": input_tuple[1].to(model.device),
                "attention_mask": input_tuple[2].to(model.device)
            }
            # input_tuple = [x.to(model.device) for x in input_tuple]

            graph_input = GraphInput.batched(input_dict, batch_dim=0, cache_results=False)
            graph_pred = graph.compute(graph_input)

            logits = model(input_tuple)
            model_pred = torch.argmax(logits, dim=1)

            assert torch.all(model_pred == graph_pred), f"on batch {i}"
            if i == 100:
                break