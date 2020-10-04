from compgraphs.mqnli_cbow import MQNLI_CBOW_CompGraph
from intervention import GraphInput
from train import load_model
from modeling.cbow import CBOWModule
from datasets.utils import my_collate
from datasets.mqnli import MQNLIData

import torch
from torch.utils.data import DataLoader
import pytest

@pytest.fixture
def mqnli_data():
    return MQNLIData("../mqnli_data/mqnli.train.txt",
                     "../mqnli_data/mqnli.dev.txt",
                     "../mqnli_data/mqnli.test.txt")

@pytest.fixture
def mqnli_cbow_model():
    model, _ = load_model(CBOWModule, "../mqnli_models/cbow.pt")
    model.eval()
    return model

@pytest.fixture
def mqnli_cbow_compgraph(mqnli_cbow_model):
    return MQNLI_CBOW_CompGraph(mqnli_cbow_model)


def has_only_child(node, child_name):
    assert len(node.children_dict) == 1 and child_name in node.children_dict

def has_children(node, child_names):
    assert len(node.children_dict) == len(child_names)
    assert all(child_name in node.children_dict for child_name in child_names)

def test_lstm_compgraph_structure(mqnli_cbow_compgraph):
    graph = mqnli_cbow_compgraph
    has_only_child(graph.root, "logits")
    has_only_child(graph.nodes["logits"], "activation2")
    has_only_child(graph.nodes["activation2"], "ffnn2")
    has_only_child(graph.nodes["ffnn2"], "activation1")
    has_only_child(graph.nodes["activation1"], "ffnn1")
    has_only_child(graph.nodes["ffnn1"], "concat")
    has_children(graph.nodes["concat"], ["premise_emb", "hypothesis_emb"])
    has_only_child(graph.nodes["premise_emb"], "input")
    has_only_child(graph.nodes["hypothesis_emb"], "input")


def test_lstm_compgraph_batch_match(mqnli_data, mqnli_cbow_model, mqnli_cbow_compgraph):
    model = mqnli_cbow_model
    graph = mqnli_cbow_compgraph
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