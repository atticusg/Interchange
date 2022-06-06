import logging
import pickle
from typing import Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import antra
from antra import GraphInput, Intervention
from antra.interchange.mapping import AbstractionMapping
import antra.counterfactual.dataset as antra_cfd
from compgraphs.mqnli_logic import Full_MQNLI_Logic_CompGraph

from datasets.mqnli import MQNLIBertDataset

logger = logging.getLogger(__name__)

def construct_bert_input(example):
    gi_dict = {
        "input_ids": example[0],
        "token_type_ids": example[1],
        "attention_mask": example[2]
    }
    return antra.GraphInput(gi_dict, cache_results=False)

def construct_high_intervention(base_ex, ivn_src_ex, high_model, high_node):
    base_input_tensor = base_ex[-2]
    ivn_input_tensor = ivn_src_ex[-2]
    base_gi = antra.GraphInput({"input": base_input_tensor}, cache_results=False)
    ivn_src_gi = antra.GraphInput({"input": ivn_input_tensor.unsqueeze(0)}, cache_results=False)
    ivn_val = high_model.compute_node(high_node, ivn_src_gi).squeeze(0)
    return antra.Intervention(
        base_gi, {high_node: ivn_val}, cache_results=False
    )


class MQNLIBertGraphInputDataset(Dataset):
    """ Wraps around a basic MQNLI dataset, but outputs batched GraphInput
    objects instead"""
    def __init__(self, base_dataset: MQNLIBertDataset):
        super(MQNLIBertGraphInputDataset, self).__init__()
        assert isinstance(base_dataset, MQNLIBertDataset)

        self.base_dataset = base_dataset
        self.base_dataset_len = len(base_dataset)
        assert self.base_dataset.variant == "basic"


    def __len__(self):
        return self.base_dataset_len

    def __getitem__(self, item):
        ex = self.base_dataset[item]

        return {
            "label": ex[-1],
            "low_input": construct_bert_input(ex)
        }

    def collate_fn(self, batch):
        low_gi = antra_cfd.graph_input_collate_fn([d["low_input"] for d in batch])
        labels = torch.tensor([d["label"] for d in batch])
        return {
            "inputs": low_gi,
            "labels": labels
        }

    def get_dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=self.collate_fn, **kwargs)



class MQNLICounterfactualDataset(antra_cfd.ListCounterfactualDataset):
    def __init__(
            self,
            base_dataset: MQNLIBertDataset,
            high_model: Full_MQNLI_Logic_CompGraph,
            pairs: Sequence[Tuple[int, int]],
            mapping: AbstractionMapping,
            num_random_bases: int = 100000,
            num_random_ivn_srcs: int = 20
    ):
        """ Counterfactual dataset with a specified list of pairs of example indices

        :param base_dataset:
        :param high_model:
        :param pairs:
        :param mapping:
        :param num_random_bases:
        :param num_random_ivn_srcs:
        """
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
        return construct_high_intervention(
            base, ivn_source, self.high_model, self.intervened_high_node)

class MQNLIRandomCfDataset(antra_cfd.RandomCounterfactualDataset):
    def __init__(
            self,
            base_dataset: MQNLIBertDataset,
            high_model: Full_MQNLI_Logic_CompGraph,
            mapping: AbstractionMapping,
            num_random_bases=50000,
            num_random_ivn_srcs=20,
            fix_examples=False,
    ):
        """ Randomly sample bases and intervention sources for counterfactual training

        :param base_dataset: MQNLIBertDataset
        :param high_model: High-level compgraph
        :param mapping: mapping between high-level model nodes and low-level n
            odes and locations
        :param num_random_bases: Number of examples to sample for bases
        :param num_random_ivn_srcs: Number of examples to sample for intervention sources
        :param fix_examples: Fix the same set of examples in different iterations
        """
        assert isinstance(base_dataset, MQNLIBertDataset)
        self.high_model = high_model

        intervened_nodes = {n for n in mapping.keys() if n not in {"input", "root"}}
        # assume only one node to intervene on for now
        self.intervened_high_node = list(intervened_nodes)[0]

        super(MQNLIRandomCfDataset, self).__init__(
            base_dataset=base_dataset,
            mapping = mapping,
            batch_dim = 0,
            num_random_bases=num_random_bases,
            num_random_ivn_srcs=num_random_ivn_srcs,
            fix_examples=fix_examples
        )

    def construct_high_intervention(self, base_ex, ivn_src_ex):
        return construct_high_intervention(
            base_ex, ivn_src_ex, self.high_model, self.intervened_high_node)

    def construct_low_input(self, example):
        return construct_bert_input(example)


class MQNLIImpactfulCFDataset(MQNLIRandomCfDataset):
    def __init__(
            self,
            base_dataset: MQNLIBertDataset,
            high_model: Full_MQNLI_Logic_CompGraph,
            mapping: AbstractionMapping,
            num_random_bases=50000,
            num_random_ivn_srcs=20,
            impactful_ratio=0.5,
            max_attempts=10,
            fix_examples=True,
            shuffle=True,
    ):
        super(MQNLIImpactfulCFDataset, self).__init__(
            base_dataset=base_dataset,
            high_model=high_model,
            mapping=mapping,
            num_random_bases=num_random_bases,
            num_random_ivn_srcs=num_random_ivn_srcs,
            fix_examples=fix_examples
        )
        self.max_attempts = max_attempts
        self.impactful_ratio = impactful_ratio
        self.rand_base_idxs = None
        self.shuffle = shuffle

    def get_ivn_srcs(self, base_idx):
        # for each base example, find a set of ivn srcs such that a fixed ratio
        # of them are impactful.
        if self.fix_examples and base_idx in self.base_idxs_to_ivn_src_idxs:
            return self.base_idxs_to_ivn_src_idxs[base_idx]

        all_src_idxs = torch.randperm(self.base_dataset_len)
        res = []

        impactful_accepted = 0
        impactful_to_accept = round(self.impactful_ratio * self.num_random_ivn_srcs)
        attempts = 0
        batch_start = 0
        batch_size = self.num_random_ivn_srcs
        # use high model to calculate whether an ivn is impactful

        base_tensor = self.base_dataset[base_idx][-2].repeat(batch_size, 1)
        base_gi = GraphInput.batched({"input": base_tensor}, cache_results=True)

        while len(res) < self.num_random_ivn_srcs and attempts < self.max_attempts:
            batch_ivn_src_idxs = all_src_idxs[batch_start:batch_start+batch_size]

            ivn_src_tensors = [self.base_dataset[i.item()][-2] for i in batch_ivn_src_idxs]
            ivn_src_tensor = torch.stack(ivn_src_tensors, dim=0)
            ivn_src_gi = GraphInput.batched({"input": ivn_src_tensor}, cache_results=False)
            ivn_vals = self.high_model.compute_node(self.intervened_high_node, ivn_src_gi)

            ivn = Intervention.batched(base_gi, {self.intervened_high_node: ivn_vals}, cache_results=False)
            base_res, ivn_res = self.high_model.intervene(ivn)

            for i, eq in enumerate(torch.eq(base_res, ivn_res)):
                if not eq and impactful_accepted < impactful_to_accept:
                    impactful_accepted += 1
                    res.append(batch_ivn_src_idxs[i].item())
                elif eq and (len(res) - impactful_accepted) < (self.num_random_ivn_srcs - impactful_to_accept):
                    res.append(batch_ivn_src_idxs[i].item())

            batch_start += batch_size
            attempts += 1

        # if attempts >= self.max_attempts and len(res) < self.num_random_ivn_srcs:
        #     logger.warning("Did not get enough examples for base input")

        # clear cache
        self.high_model.clear_caches(base_gi)

        if self.fix_examples:
            self.base_idxs_to_ivn_src_idxs[base_idx] = res

        return res

    def __iter__(self):
        if not self.fix_examples or self.rand_base_idxs is None:
            rand_base_idxs = []
            for base_idx in torch.randperm(self.base_dataset_len):
                base_idx = base_idx.item()
                ivn_src_idxs = self.get_ivn_srcs(base_idx)
                if len(ivn_src_idxs) < self.num_random_ivn_srcs:
                    continue

                rand_base_idxs.append(base_idx)

                for ivn_src_idx in ivn_src_idxs:
                    base = self.base_dataset[base_idx]
                    ivn_src = self.base_dataset[ivn_src_idx]
                    yield self.prepare_one_example(base, ivn_src, base_idx, ivn_src_idx)
                if len(rand_base_idxs) >= self.num_random_bases:
                    break
            if self.fix_examples:
                self.rand_base_idxs = rand_base_idxs
        else:
            # use base indexes sampled during first iteration.
            rand_base_idxs = torch.tensor(self.rand_base_idxs)
            if self.shuffle:
                rand_base_idxs = rand_base_idxs[torch.randperm(len(self.rand_base_idxs))]

            for base_idx in rand_base_idxs:
                base_idx = base_idx.item()
                ivn_src_idxs = self.get_ivn_srcs(base_idx)
                if len(ivn_src_idxs) < self.num_random_ivn_srcs:
                    continue
                for ivn_src_idx in ivn_src_idxs:
                    base = self.base_dataset[base_idx]
                    ivn_src = self.base_dataset[ivn_src_idx]
                    yield self.prepare_one_example(base, ivn_src, base_idx, ivn_src_idx)

