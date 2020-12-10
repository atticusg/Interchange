import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from trainer import load_model
from modeling.pretrained_bert import PretrainedBertModule

import intervention
from intervention.utils import serialize
from compgraphs import MQNLI_Logic_CompGraph
from compgraphs import MQNLI_Bert_CompGraph, Abstr_MQNLI_Bert_CompGraph

from probing.dataset import ProbingDataset

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
    subset = Subset(data.dev, list(range(num_examples)))

    print("constructing compgraph")
    base_graph = MQNLI_Bert_CompGraph(model)
    intermediate_nodes = [lo_node_name]
    hi_compgraph = MQNLI_Logic_CompGraph(data)
    lo_compgraph = Abstr_MQNLI_Bert_CompGraph(base_graph, intermediate_nodes)
    lo_compgraph.set_cache_device(torch.device("cpu"))

    probing_dataset = ProbingDataset(
        subset, hi_compgraph, lo_compgraph, lo_node_name,
        preprocessing_device=device,
        preprocessing_batch_size=64
    )
