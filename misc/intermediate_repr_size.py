import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from trainer import load_model
from modeling.pretrained_bert import PretrainedBertModule

import intervention
from intervention.utils import serialize
from compgraphs import MQNLI_Bert_CompGraph, Abstr_MQNLI_Bert_CompGraph

print("loading model")
model_path = "../data/models/bert-easy-best.pt"
data_path = "../data/mqnli/preprocessed/bert-easy.pt"
compgraph_save_path = "probing_results/compgraph-state-dict.pt"
num_examples = 6400

model, _ = load_model(PretrainedBertModule, model_path)
device = torch.device("cuda")

model = model.to(device)
model.eval()

print("constructing compgraph")
base_graph = MQNLI_Bert_CompGraph(model)
intermediate_nodes = [f"bert_layer_{i}" for i in range(1)]
compgraph = Abstr_MQNLI_Bert_CompGraph(base_graph, intermediate_nodes)
compgraph.set_cache_device(torch.device("cpu"))

print(list(compgraph.nodes.keys()))

print("loading data")
data = torch.load(data_path)
subset = Subset(data.dev, list(range(num_examples)))
dataloader = DataLoader(subset, batch_size=32, shuffle=False)

with torch.no_grad():
    for i, input_tuple in enumerate(tqdm(dataloader)):
        key = [serialize(x) for x in input_tuple[0]]
        input_tuple_for_graph = [x.to(device) for x in input_tuple]
        graph_input = intervention.GraphInput.batched(
            {"input": input_tuple_for_graph}, key, batch_dim=0
        )
        low_output = compgraph.compute(graph_input)

# compgraph has cached all the results. See how large it is.
print("Saving state dict")
state_dict = compgraph.get_state_dict()
torch.save(state_dict, compgraph_save_path)
