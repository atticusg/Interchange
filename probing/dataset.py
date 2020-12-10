import torch
from torch.utils.data import Dataset, Subset, DataLoader
import intervention
from intervention.utils import serialize

from collections import defaultdict
from typing import Union

excluded_high_nodes = {"input", "root", "get_p", "get_h"}

class ProbingData:
    def __init__(self, data, hi_compgraph, lo_compgraph, lo_node_name,
                 probe_train_num_examples: int=6000,
                 probe_train_num_dev_examples: int=600,
                 probe_preprocessing_device: Union[str, torch.device]= "cuda",
                 probe_preprocessing_batch_size: int=128):
        super(ProbingData, self).__init__()

        self.lo_node_name = lo_node_name

        train_subset = Subset(data.train, list(range(probe_train_num_examples)))
        self.train = ProbingDataset(
            train_subset, hi_compgraph, lo_compgraph, lo_node_name,
            preprocessing_device=probe_preprocessing_device,
            preprocessing_batch_size=probe_preprocessing_batch_size
        )

        dev_subset = Subset(data.dev, list(range(probe_train_num_dev_examples)))
        self.train = ProbingDataset(
            dev_subset, hi_compgraph, lo_compgraph, lo_node_name,
            preprocessing_device=probe_preprocessing_device,
            preprocessing_batch_size=probe_preprocessing_batch_size
        )

class ProbingDataset(Dataset):
    def __init__(self, dataset, hi_compgraph, lo_compgraph, lo_node_name,
                 preprocessing_device: Union[str, torch.device]="cuda",
                 preprocessing_batch_size: int=128):
        super(ProbingDataset, self).__init__()
        dataloader = DataLoader(dataset, batch_size=preprocessing_batch_size, shuffle=False)

        if isinstance(preprocessing_device, str):
            device = torch.device(preprocessing_device)
        else:
            device = preprocessing_device

        self.num_examples = len(dataset)
        self.labels = defaultdict(list)
        self.inputs = []

        with torch.no_grad():
            for i, input_tuple in enumerate(dataloader):
                if "bert" in lo_node_name:
                    high_input_tensor = input_tuple[-2]
                elif "lstm" in lo_node_name:
                    high_input_tensor = torch.cat((input_tuple[0][:, :9],
                                                   input_tuple[0][:, 10:]),
                                                  dim=1)
                high_key = [serialize(x) for x in high_input_tensor]
                hi_input = intervention.GraphInput.batched(
                    {"input": high_input_tensor.T}, high_key, batch_dim=0)
                # high model uses batch_dim = 0 because all intermediate outputs are
                # in batch-first order.

                hi_output = hi_compgraph.compute(hi_input)
                # get intermediate labels from high computation graph
                for hi_node_name in hi_compgraph.nodes:
                    if hi_node_name not in excluded_high_nodes:
                        hi_node_values = hi_compgraph.get_result(hi_node_name, hi_input)
                        self.labels[hi_node_name].extend(hi_node_values)

                # question: what high intermediate nodes should we care about?
                # question: complex intermediate nodes, e.g. projection quantifiers

                key = [serialize(x) for x in input_tuple[0]]
                input_tuple_for_graph = [x.to(device) for x in input_tuple]
                lo_input = intervention.GraphInput.batched(
                    {"input": input_tuple_for_graph}, key, batch_dim=0
                )
                lo_output = lo_compgraph.compute(lo_input)
                low_hidden_values = lo_compgraph.get_result(lo_node_name, lo_input)
                self.inputs.extend(low_hidden_values)
                # get low hidden
                # question: should we keep the results in case where lo_output != high_output?

        print(f"Got {len(self.inputs)} inputs")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        res = {"hidden": self.inputs[item]}
        res.update({name: values[item] for name, values in self.labels})
        return res
