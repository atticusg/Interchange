import torch
from torch.utils.data import Dataset, DataLoader
import intervention
from intervention.utils import serialize

from tqdm import tqdm
from collections import defaultdict
from typing import Union

from probing.utils import get_num_classes
from datasets.mqnli import get_collate_fxn

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

"""
notevery | bad | singer | doesnot | badly | sings | every | good | song | ...
0          1     2        3         4       5       6       7      8      

"""

SUBTREE_IDXS = {
    "sentence_q": [0, 9],
    "subj": [1, 2, 10, 11],
    "subj_adj": [1, 10],
    "subj_noun": [2, 11],
    "negp": [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17],
    "neg": [3, 12],
    "vp": [4, 5, 6, 7, 8, 13, 14, 15, 16, 17],
    "vp_q": [6, 15],
    "v_bar": [4, 5, 13, 14],
    "v_adv": [4, 13],
    "v_verb": [5, 14],
    "obj": [7, 8, 16, 17],
    "obj_adj": [7, 16],
    "obj_noun": [8, 17]
}

class ProbingData:
    def __init__(self, data, hi_compgraph, lo_compgraph, lo_node_name, lo_model_type,
                 probe_train_num_examples: int=6000,
                 probe_train_num_dev_examples: int=600,
                 probe_preprocessing_device: Union[str, torch.device]= "cuda",
                 probe_preprocessing_batch_size: int=128,
                 probe_correct_examples_only: bool=False, **kwargs):
        super(ProbingData, self).__init__()

        self.lo_node_name = lo_node_name

        self.train = ProbingDataset(
            data.train, probe_train_num_examples, hi_compgraph,
            lo_compgraph, lo_node_name, lo_model_type,
            preprocessing_device=probe_preprocessing_device,
            preprocessing_batch_size=probe_preprocessing_batch_size,
            probe_correct_examples_only=probe_correct_examples_only
        )

        self.dev = ProbingDataset(
            data.dev, probe_train_num_dev_examples, hi_compgraph,
            lo_compgraph, lo_node_name, lo_model_type,
            preprocessing_device=probe_preprocessing_device,
            preprocessing_batch_size=probe_preprocessing_batch_size,
            probe_correct_examples_only=probe_correct_examples_only
        )

class ProbingDataset(Dataset):
    def __init__(self, dataset, num_examples: int,
                 high_compgraph: intervention.ComputationGraph,
                 low_compgraph: intervention.ComputationGraph,
                 low_node: str, low_model_type: str,
                 preprocessing_device: Union[str, torch.device]="cuda",
                 preprocessing_batch_size: int=128,
                 probe_correct_examples_only: bool=False):

        super(ProbingDataset, self).__init__()
        dataloader = DataLoader(dataset, batch_size=preprocessing_batch_size, shuffle=False)

        if isinstance(preprocessing_device, str):
            device = torch.device(preprocessing_device)
        else:
            device = preprocessing_device

        self.low_node = low_node
        self.num_examples = num_examples
        self.labels = defaultdict(list)
        self.inputs = []
        self.high_nodes = [n for n in high_compgraph.nodes if n not in excluded_high_nodes]
        self.ctrl_mapping = {n: {} for n in self.high_nodes}
        self.ctrl_labels = defaultdict(list)
        self.correct_only = probe_correct_examples_only

        with torch.no_grad():
            with tqdm(total=self.num_examples, desc="[ProbingDataset]") as pbar:
                for input_tuple in dataloader:
                    if len(self.inputs) == self.num_examples:
                        break
                    if low_model_type == "bert":
                        high_input_tensor = input_tuple[-2]
                    elif low_model_type == "lstm":
                        high_input_tensor = torch.cat((input_tuple[0][:, :9],
                                                       input_tuple[0][:, 10:]),
                                                      dim=1)
                    high_key = [serialize(x) for x in high_input_tensor]
                    high_input = intervention.GraphInput.batched(
                        {"input": high_input_tensor.T}, high_key,
                        batch_dim=0,
                    )
                    # high model uses batch_dim = 0 because all intermediate
                    # outputs are in batch-first order.

                    high_output = high_compgraph.compute(high_input).cpu()

                    key = [serialize(x) for x in input_tuple[0]]
                    if low_model_type == "bert":
                        input_tuple_for_graph = [x.to(device) for x in input_tuple]
                    elif low_model_type == "lstm":
                        input_tuple_for_graph = [input_tuple[0].T.to(device),
                                                 input_tuple[1].to(device)]

                    low_input = intervention.GraphInput.batched(
                        {"input": input_tuple_for_graph}, key,
                        batch_dim=(1 if low_model_type == "lstm" else 0)
                    )
                    low_output = low_compgraph.compute(low_input)
                    low_output = low_output.to(torch.device("cpu"))

                    correct = (high_output == low_output) if self.correct_only \
                        else torch.ones(len(low_output), dtype=torch.bool)
                    num_new_exs = min(sum(correct).item(),
                                      self.num_examples - len(self.inputs))

                    # gather results
                    for high_node in self.high_nodes:
                        high_node_vals = high_compgraph.get_result(high_node, high_input)
                        high_node_vals = high_node_vals[correct][:num_new_exs]

                        # high node value may be a projectivity signature,
                        # so remap them to a single integer
                        if high_node in {"subj_adj", "v_adv", "obj_adj",
                                            "vp_q", "sentence_q", "neg"}:
                            if high_node in {"subj_adj", "v_adv", "obj_adj"}:
                                sig_to_idx = ADJ_SIG_TO_IDX
                            elif high_node in {"vp_q", "sentence_q"}:
                                sig_to_idx = QUANTIFIER_SIG_TO_IDX
                            else:
                                sig_to_idx = NEG_SIG_TO_IDX
                            high_node_vals = torch.tensor(
                                [sig_to_idx[tuple(x.tolist())]
                                 for x in high_node_vals]
                            )

                        self.labels[high_node].extend(high_node_vals)

                        # get control labels
                        correct_input_tensors = high_input_tensor[correct][:num_new_exs]
                        ctrl_idxs = torch.tensor(SUBTREE_IDXS[high_node])
                        ctrl_inputs = correct_input_tensors[:, ctrl_idxs]
                        for i, ctrl_input in enumerate(ctrl_inputs):
                            ctrl_input_key = serialize(ctrl_input)
                            if high_node in {"sentence_q", "neg", "vp_q"}:
                                # closed-class inputs, assign numeric label depending on order of occurrence
                                ctrl_label = self.ctrl_mapping[high_node]\
                                    .get(ctrl_input_key, torch.tensor(len(self.ctrl_mapping[high_node])))
                            else:
                                ctrl_label = torch.randint(get_num_classes(high_node),[])
                            self.ctrl_mapping[high_node][ctrl_input_key] = ctrl_label
                            self.ctrl_labels[high_node].append(ctrl_label)
                                # if i == 0:
                                #     print("high_node", high_node, "ctrl_input", ctrl_input_key, "label", ctrl_label)

                    low_hidden_values = low_compgraph.get_result(low_node, low_input)
                    if low_model_type == "lstm": # make batch first
                        low_hidden_values = low_hidden_values.transpose(0,1)
                    low_hidden_values = low_hidden_values[correct][:num_new_exs]
                    low_hidden_values = low_hidden_values.to(torch.device("cpu"))
                    self.inputs.extend(low_hidden_values)
                    pbar.update(num_new_exs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, item):
        res = {"hidden": self.inputs[item]}
        res.update({name: values[item] for name, values in self.labels.items()})
        res.update({name + "_ctrl": values[item] for name, values in self.ctrl_labels.items()})
        return res

    def get_collate_fn(self, high_node, low_node, low_loc, is_control):
        if low_node != self.low_node:
            raise RuntimeError(f"This dataset does not contain hidden vectors for low node {low_node}")

        if is_control:
            def _collate_fn_ctrl(batch):
                hidden = torch.stack([x["hidden"][low_loc] for x in batch])
                label = torch.stack([x[high_node+"_ctrl"] for x in batch])
                return (hidden, label)
            return _collate_fn_ctrl
        else:
            def _collate_fn(batch):
                hidden = torch.stack([x["hidden"][low_loc] for x in batch])
                label = torch.stack([x[high_node] for x in batch])
                return (hidden, label)

        return _collate_fn
