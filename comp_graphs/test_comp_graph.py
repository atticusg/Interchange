from comp_graphs.mqnli_lstm import MQNLI_LSTM_CompGraph
from intervention import GraphInput
from train import load_model
from modeling.lstm import LSTMModule
from datasets.utils import my_collate
from datasets.mqnli import MQNLIData

import torch
from torch.utils.data import DataLoader
import pytest

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
    has_only_child(graph.nodes["premise_emb"], "input")
    has_only_child(graph.nodes["hypothesis_lstm_3"], "hypothesis_lstm_2")
    has_only_child(graph.nodes["hypothesis_lstm_2"], "hypothesis_lstm_1")
    has_only_child(graph.nodes["hypothesis_lstm_1"], "hypothesis_lstm_0")
    has_only_child(graph.nodes["hypothesis_lstm_0"], "hypothesis_emb")
    has_only_child(graph.nodes["hypothesis_emb"], "input")


def test_lstm_compgraph_match_results(mqnli_data, mqnli_lstm_model, mqnli_lstm_comp_graph):
    model = mqnli_lstm_model
    graph = mqnli_lstm_comp_graph
    with torch.no_grad():
        collate_fn = lambda batch: my_collate(batch, batch_first=False)
        dataloader = DataLoader(mqnli_data.dev, batch_size=100, shuffle=False,
                                collate_fn=collate_fn)
        for i, input_tuple in enumerate(dataloader):
            input_tuple = [x.to(model.device) for x in input_tuple]

            graph_input = GraphInput({"input": input_tuple[0]})
            graph_pred = graph.compute(graph_input, store_cache=True)

            emb_x = model.embedding(input_tuple)
            assert torch.all(emb_x == graph.get_result("input", graph_input))

            premise = model.premise_emb(emb_x)
            assert torch.all(premise == graph.get_result("premise_emb", graph_input))

            prem_lstm_0_out, _  = model.lstm_layers[0](premise)
            assert torch.all(prem_lstm_0_out == graph.get_result("premise_lstm_0", graph_input))
            prem_lstm_1_out, _ = model.lstm_layers[1](prem_lstm_0_out)
            assert torch.all(prem_lstm_1_out == graph.get_result("premise_lstm_1", graph_input))

            prem_lstm_2_out, _ = model.lstm_layers[2](prem_lstm_1_out)
            assert torch.all(prem_lstm_2_out == graph.get_result("premise_lstm_2", graph_input))

            prem_lstm_3_out, _ = model.lstm_layers[3](prem_lstm_2_out)
            assert torch.all(prem_lstm_3_out == graph.get_result("premise_lstm_3", graph_input))


            premise_h = model._run_lstm(premise)
            assert torch.all(premise_h == prem_lstm_3_out)


            hypothesis = model.hypothesis_emb(emb_x)
            hypothesis_h = model._run_lstm(hypothesis)
            repr = model.concat_final_state(premise_h, hypothesis_h)
            repr = model.dropout0(repr)

            output = model.feed_forward1(repr)
            output = model.activation1(output)
            output = model.dropout1(output)

            output = model.feed_forward2(output)
            output = model.activation2(output)
            output = model.dropout2(output)
            output = model.logits(output)

            model_explicit_pred = torch.argmax(output, dim=1)

            logits = model(input_tuple)
            model_implicit_pred = torch.argmax(logits, dim=1)

            assert torch.all(model_explicit_pred == model_implicit_pred)

            assert torch.all(model_explicit_pred == graph_pred), f"on batch {i}"