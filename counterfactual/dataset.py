import pickle

import torch
from tqdm import tqdm

import antra
import antra.counterfactual.dataset as antra_cfd


class MQNLICounterfactualDataset(antra_cfd.ListCounterfactualDataset):
    def __init__(
            self,
            base_dataset,
            high_model,
            pairs,
            mapping,
            num_random_bases=100000,
            num_random_ivn_srcs=20
    ):
        self.high_model = high_model
        self.num_random_bases = num_random_bases
        self.num_random_ivn_srcs = num_random_ivn_srcs
        intervened_nodes = {n for n in mapping.keys() if n not in {"input", "root"}}
        # assume only one node to intervene on for now
        self.intervened_high_node = list(intervened_nodes)[0]

        if isinstance(pairs, list):
            intervention_pairs = pairs
        elif pairs == "random":
            intervention_pairs = self.get_random_ivn_pairs(base_dataset)
        elif isinstance(pairs, str):
            with open(pairs, "wb") as f:
                intervention_pairs = pickle.load(f)
        else:
            raise ValueError(f"Invalid value for `pairs` {pairs}")

        super(MQNLICounterfactualDataset, self).__init__(
            base_dataset=base_dataset,
            intervention_pairs=intervention_pairs,
            mapping = mapping,
            batch_dim = 0
        )

    def get_random_ivn_pairs(self, base_dataset):
        print("Getting random intervention pairs")
        base_idxs = torch.randperm(len(base_dataset))[:self.num_random_bases]
        pairs = []
        for base_idx in tqdm(base_idxs):
            ivn_src_idxs = torch.randperm(len(base_dataset))[:self.num_random_ivn_srcs]
            pairs.extend((base_idx.item(), ivn_src_idx.item()) for ivn_src_idx in ivn_src_idxs)
        return pairs

    def construct_intervention(self, base, ivn_source):
        base_input_tensor = base[-2].unsqueeze(0)
        ivn_input_tensor = ivn_source[-2].unsqueeze(0)
        base_gi = antra.GraphInput({"input": base_input_tensor}, cache_results=False)
        ivn_src_gi = antra.GraphInput({"input": ivn_input_tensor}, cache_results=False)
        ivn_val = self.high_model.compute_node(self.intervened_high_node, ivn_src_gi)

        return antra.Intervention(
            base_gi, {self.intervened_high_node: ivn_val}, cache_results=False
        )

class MQNLIRandomIterableCfDataset(antra_cfd.RandomCounterfactualDataset):
    def __init__(
            self,
            base_dataset,
            high_model,
            mapping,
            num_random_bases=50000,
            num_random_ivn_srcs=20
    ):
        self.high_model = high_model

        intervened_nodes = {n for n in mapping.keys() if n not in {"input", "root"}}
        # assume only one node to intervene on for now
        self.intervened_high_node = list(intervened_nodes)[0]

        super(MQNLIRandomIterableCfDataset, self).__init__(
            base_dataset=base_dataset,
            mapping = mapping,
            batch_dim = 0,
            num_random_bases=num_random_bases,
            num_random_ivn_srcs=num_random_ivn_srcs
        )

    def construct_high_intervention(self, base_ex, ivn_src_ex):
        base_input_tensor = base_ex[-2]
        ivn_input_tensor = ivn_src_ex[-2]
        base_gi = antra.GraphInput({"input": base_input_tensor}, cache_results=False)
        ivn_src_gi = antra.GraphInput({"input": ivn_input_tensor.unsqueeze(0)}, cache_results=False)
        ivn_val = self.high_model.compute_node(self.intervened_high_node, ivn_src_gi).squeeze(0)
        return antra.Intervention(
            base_gi, {self.intervened_high_node: ivn_val}, cache_results=False
        )

    def construct_low_input(self, example):
        gi_dict = {
            "input_ids": example[0],
            "token_type_ids": example[1],
            "attention_mask": example[2]
        }

        return antra.GraphInput(gi_dict, cache_results=False)

