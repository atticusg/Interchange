import torch
from torch.utils.data import Dataset, DataLoader
import intervention
from intervention.utils import serialize

from tqdm import tqdm
from collections import defaultdict
from typing import Union

from probing.utils import get_num_classes

excluded_high_nodes = {"input", "root", "get_p", "get_h"}

# map each projectivity signature to an index
NEG_SIG_TO_IDX = {
    (0, 1, 2, 3, 4, 5, 6): 0,
    (0, 4, 5, 6, 1, 2, 3): 1,
    (0, 4, 6, 5, 1, 3, 2): 2,
    (0, 1, 3, 2, 4, 6, 5): 3
}

QUANTIFIER_SIG_TO_IDX = {
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 6, 0, 6, 0, 2, 2, 0, 6, 0, 6, 0, 3, 0,
     3, 6, 0, 6): 0,
    (0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 4, 5, 6, 0, 3, 0, 3, 5, 5, 0, 0, 3, 0,
     3, 6, 0, 6): 1,
    (0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6, 3, 0, 3, 0, 5, 5, 0, 3, 0, 3, 0, 6, 0,
     6, 3, 0, 3): 2,
    (0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 6, 1, 2, 3, 0, 6, 0, 6, 2, 2, 0, 0, 6, 0,
     6, 3, 0, 3): 3,
    (0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 4, 5, 6, 0, 2, 2, 0, 6, 0, 6, 0, 2, 2,
     0, 5, 5, 0): 4,
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 5, 5, 0, 0, 3, 0, 3, 5, 5, 0, 0, 2, 2,
     0, 5, 5, 0): 5,
    (0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 1, 2, 3, 0, 5, 5, 0, 3, 0, 3, 0, 5, 5,
     0, 2, 2, 0): 6,
    (0, 0, 0, 0, 0, 0, 0, 0, 4, 5, 6, 2, 2, 0, 0, 6, 0, 6, 2, 2, 0, 0, 5, 5,
     0, 2, 2, 0): 7,
    (0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 5, 2, 0, 2, 0, 6, 6, 0, 2, 0, 2, 0, 5, 0,
     5, 2, 0, 2): 8,
    (0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 5, 1, 3, 2, 0, 5, 0, 5, 3, 3, 0, 0, 5, 0,
     5, 2, 0, 2): 9,
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 2, 5, 0, 5, 0, 3, 3, 0, 5, 0, 5, 0, 2, 0,
     2, 5, 0, 5): 10,
    (0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 4, 6, 5, 0, 2, 0, 2, 6, 6, 0, 0, 2, 0,
     2, 5, 0, 5): 11,
    (0, 0, 0, 0, 0, 0, 0, 0, 6, 6, 0, 1, 3, 2, 0, 6, 6, 0, 2, 0, 2, 0, 6, 6,
     0, 3, 3, 0): 12,
    (0, 0, 0, 0, 0, 0, 0, 0, 4, 6, 5, 3, 3, 0, 0, 5, 0, 5, 3, 3, 0, 0, 6, 6,
     0, 3, 3, 0): 13,
    (0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 4, 6, 5, 0, 3, 3, 0, 5, 0, 5, 0, 3, 3,
     0, 6, 6, 0): 14,
    (0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 2, 6, 6, 0, 0, 2, 0, 2, 6, 6, 0, 0, 3, 3,
     0, 6, 6, 0): 15
}

ADJ_SIG_TO_IDX = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (0, 3): 3
}

class ProbingData:
    def __init__(self, data, hi_compgraph, lo_compgraph, lo_node_name,
                 probe_train_num_examples: int=6000,
                 probe_train_num_dev_examples: int=600,
                 probe_preprocessing_device: Union[str, torch.device]= "cuda",
                 probe_preprocessing_batch_size: int=128,
                 probe_correct_examples_only: bool=False, **kwargs):
        super(ProbingData, self).__init__()

        self.lo_node_name = lo_node_name

        self.train = ProbingDataset(
            data.train, probe_train_num_examples, hi_compgraph, lo_compgraph, lo_node_name,
            preprocessing_device=probe_preprocessing_device,
            preprocessing_batch_size=probe_preprocessing_batch_size,
            probe_correct_examples_only=probe_correct_examples_only
        )

        self.dev = ProbingDataset(
            data.dev, probe_train_num_dev_examples, hi_compgraph, lo_compgraph, lo_node_name,
            preprocessing_device=probe_preprocessing_device,
            preprocessing_batch_size=probe_preprocessing_batch_size,
            probe_correct_examples_only=probe_correct_examples_only
        )

class ProbingDataset(Dataset):
    def __init__(self, dataset, num_examples, hi_compgraph, lo_compgraph, lo_node_name,
                 preprocessing_device: Union[str, torch.device]="cuda",
                 preprocessing_batch_size: int=128,
                 probe_correct_examples_only: bool=False):

        super(ProbingDataset, self).__init__()
        dataloader = DataLoader(dataset, batch_size=preprocessing_batch_size, shuffle=False)

        if isinstance(preprocessing_device, str):
            device = torch.device(preprocessing_device)
        else:
            device = preprocessing_device

        self.low_node = lo_node_name
        self.num_examples = num_examples
        self.labels = defaultdict(list)
        self.inputs = []
        self.correct_only = probe_correct_examples_only

        with torch.no_grad():
            with tqdm(total=self.num_examples, desc="[ProbingDataset]") as pbar:
                for input_tuple in dataloader:
                    if len(self.inputs) == self.num_examples:
                        break
                    if "bert" in lo_node_name:
                        high_input_tensor = input_tuple[-2]
                    elif "lstm" in lo_node_name:
                        high_input_tensor = torch.cat((input_tuple[0][:, :9],
                                                       input_tuple[0][:, 10:]),
                                                      dim=1)
                    high_key = [serialize(x) for x in high_input_tensor]
                    hi_input = intervention.GraphInput.batched(
                        {"input": high_input_tensor.T}, high_key, batch_dim=0)
                    # high model uses batch_dim = 0 because all intermediate
                    # outputs are in batch-first order.

                    hi_output = hi_compgraph.compute(hi_input).cpu()

                    key = [serialize(x) for x in input_tuple[0]]
                    input_tuple_for_graph = [x.to(device) for x in input_tuple]
                    lo_input = intervention.GraphInput.batched(
                        {"input": input_tuple_for_graph}, key, batch_dim=0
                    )
                    lo_output = lo_compgraph.compute(lo_input)
                    lo_output = lo_output.to(torch.device("cpu"))

                    correct = (hi_output == lo_output) if self.correct_only else torch.ones(len(lo_output), dtype=torch.bool)
                    num_new_exs = min(sum(correct).item(), self.num_examples - len(self.inputs))

                    # gather results
                    for hi_node_name in hi_compgraph.nodes:
                        if hi_node_name not in excluded_high_nodes:
                            hi_node_values = hi_compgraph.get_result(hi_node_name, hi_input)
                            hi_node_values = hi_node_values[correct][:num_new_exs]

                            # high node value may be a projectivity signature,
                            # so remap them to a single integer
                            if hi_node_name in {"subj_adj", "v_adv", "obj_adj",
                                                "vp_q", "sentence_q", "neg"}:
                                if hi_node_name in {"subj_adj", "v_adv", "obj_adj"}:
                                    sig_to_idx = ADJ_SIG_TO_IDX
                                elif hi_node_name in {"vp_q", "sentence_q"}:
                                    sig_to_idx = QUANTIFIER_SIG_TO_IDX
                                else:
                                    sig_to_idx = NEG_SIG_TO_IDX
                                hi_node_values = torch.tensor(
                                    [sig_to_idx[tuple(x.tolist())]
                                     for x in hi_node_values]
                                )

                            self.labels[hi_node_name].extend(hi_node_values)

                    low_hidden_values = lo_compgraph.get_result(lo_node_name, lo_input)
                    low_hidden_values = low_hidden_values[correct][:num_new_exs]
                    low_hidden_values = low_hidden_values.to(torch.device("cpu"))
                    self.inputs.extend(low_hidden_values)
                    pbar.update(num_new_exs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        res = {"hidden": self.inputs[item]}
        res.update({name: values[item] for name, values in self.labels.items()})
        return res

    def get_collate_fn(self, high_node, low_node, low_loc, is_control):
        if low_node != self.low_node:
            raise RuntimeError(f"This dataset does not contain hidden vectors for low node {low_node}")

        if is_control:
            def _collate_fn_ctrl(batch):
                hidden = torch.stack([x["hidden"][low_loc] for x in batch])
                label = torch.randint(get_num_classes(high_node), (len(batch),))
                return (hidden, label)
            return _collate_fn_ctrl
        else:
            def _collate_fn(batch):
                hidden = torch.stack([x["hidden"][low_loc] for x in batch])
                label = torch.stack([x[high_node] for x in batch])
                return (hidden, label)

            return _collate_fn
