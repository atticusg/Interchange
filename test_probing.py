import time

import torch
from trainer import load_model
from modeling.pretrained_bert import PretrainedBertModule

from intervention import LOC
from compgraphs import MQNLI_Logic_CompGraph
from compgraphs import MQNLI_Bert_CompGraph, Abstr_MQNLI_Bert_CompGraph

from probing.modules import Probe
from probing.trainer import ProbeTrainer
from probing.dataset import ProbingData, ProbingDataset



def test_probe_dataset():
    print("loading model")
    model_path = "mqnli_models/bert-easy-best.pt"
    data_path = "mqnli_data/mqnli-bert-default.pt"
    compgraph_save_path = "probing_results/compgraph-state-dict.pt"
    lo_node_name = "bert_layer_0"
    num_examples = 6400

    model, _ = load_model(PretrainedBertModule, model_path)
    device = torch.device("cuda")

    model = model.to(device)
    model.eval()

    print("loading data")
    data = torch.load(data_path)

    print("constructing compgraph")
    base_graph = MQNLI_Bert_CompGraph(model)
    intermediate_nodes = [lo_node_name]
    hi_compgraph = MQNLI_Logic_CompGraph(data)
    lo_compgraph = Abstr_MQNLI_Bert_CompGraph(base_graph, intermediate_nodes)
    lo_compgraph.set_cache_device(torch.device("cpu"))

    print("Constructing dataset for probing")
    probing_dataset = ProbingDataset(
        data.train, num_examples, hi_compgraph, lo_compgraph, lo_node_name,
        preprocessing_device=device,
        preprocessing_batch_size=64,
        probe_correct_examples_only=True
    )

def test_train():
    print("loading model")
    model_path = "mqnli_models/bert-easy-best.pt"
    data_path = "mqnli_data/mqnli-bert-default.pt"

    hi_node_name = "subj"
    lo_node_name = "bert_layer_0"
    low_loc = LOC[3,:]
    num_examples = 6400
    probe_save_dir = "probing_results/test_train/"

    model, _ = load_model(PretrainedBertModule, model_path)
    device = torch.device("cuda")

    model = model.to(device)
    model.eval()

    print("loading data")
    data = torch.load(data_path)

    print("constructing compgraph")
    base_graph = MQNLI_Bert_CompGraph(model)
    intermediate_nodes = [lo_node_name]
    hi_compgraph = MQNLI_Logic_CompGraph(data)
    lo_compgraph = Abstr_MQNLI_Bert_CompGraph(base_graph, intermediate_nodes)
    lo_compgraph.set_cache_device(torch.device("cpu"))

    probe = Probe(hi_node_name, lo_node_name, low_loc, is_control=False,
                  probe_output_classes=7, probe_input_dim=768, probe_max_rank=12,
                  probe_dropout=0.1)

    print("obtaining hidden variables")
    probe_data = ProbingData(data, hi_compgraph, lo_compgraph, lo_node_name,
                             probe_train_num_examples=num_examples,
                             probe_train_num_dev_examples=1600)
    trainer = ProbeTrainer(probe_data, probe,
                           probe_train_batch_size=128,
                           res_save_dir=probe_save_dir,
                           probe_train_lr_patience_epochs=10,
                           device=device)
    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time
    print(f"Training took {duration:.2}s")

def test_load_probe():
    save_path = "probing_results/test_train/subj-bert_layer_0-3_x-1212_103748.pt"
    model_path = "mqnli_models/bert-easy-best.pt"
    data_path = "mqnli_data/mqnli-bert-default.pt"

    probe = Probe.from_checkpoint(save_path)

    model, _ = load_model(PretrainedBertModule, model_path)
    device = torch.device("cuda")

    model = model.to(device)
    model.eval()

    print("loading data")
    data = torch.load(data_path)

    print("constructing compgraph")
    base_graph = MQNLI_Bert_CompGraph(model)
    intermediate_nodes = [probe.low_node]
    hi_compgraph = MQNLI_Logic_CompGraph(data)
    lo_compgraph = Abstr_MQNLI_Bert_CompGraph(base_graph, intermediate_nodes)
    lo_compgraph.set_cache_device(torch.device("cpu"))

    print("obtaining hidden variables")
    probe_data = ProbingData(data, hi_compgraph, lo_compgraph, probe.low_node,
                             probe_train_num_examples=1600,
                             probe_train_num_dev_examples=6400)

    trainer = ProbeTrainer(probe_data, probe,
                           probe_train_batch_size=128,
                           res_save_dir="",
                           probe_train_lr_patience_epochs=10,
                           device=device)

    acc, loss = trainer.eval()
    print(f"Got acc {acc:.2%} loss {loss:.4}")