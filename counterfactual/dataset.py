import math
from typing import *
from collections import defaultdict
import pickle
from multiprocessing import Pool

import torch
from torch.utils.data import IterableDataset, Dataset, DataLoader
from tqdm import tqdm

import antra
from antra.utils import serialize
from antra.interchange.mapping import AbstractionMapping
from antra.counterfactual import CounterfactualDataset



class IsolatedPairFinder:
    def __init__(
            self,
            high_model_class,
            high_model_params,
            base_dataset: Dataset,
            mapping: AbstractionMapping,
            pairs_pickle_path: str,
            num_base_inputs: int = 50,
            num_ivn_srcs: int = 20,
            pair_finding_batch_size: int = 256,
            num_workers: int = 1,
    ):
        self.high_model_class = high_model_class
        self.high_model_params = high_model_params
        self.base_dataset = base_dataset
        self.mapping = mapping
        self.num_base_inputs = num_base_inputs
        self.num_ivn_srcs = num_ivn_srcs
        self.pair_finding_batch_size = pair_finding_batch_size
        self.num_workers = num_workers

        self.ignored_nodes = {"input"}


        self.intervention_pairs = self.get_intervention_pairs()
        with open(pairs_pickle_path, "wb") as f:
            pickle.dump(self.intervention_pairs, f)


    def get_intervention_pairs(self):
        base_idxs = torch.randperm(len(self.base_dataset))[:self.num_base_inputs]
        if self.num_workers == 1:
            pairs = self.find_pair_worker_fn(base_idxs)
            print(f"{len(pairs)=}")
        else:
            pairs = []
            num_idxs_per_worker = math.ceil(len(base_idxs) / self.num_workers)
            base_idxs_for_workers = [base_idxs[i*num_idxs_per_worker:(i+1)*num_idxs_per_worker]
                                     for i in range(self.num_workers)]
            with Pool(processes=self.num_workers) as pool:
                print(f"Launched {self.num_workers} workers")
                all_pairs = list(pool.imap_unordered(self.find_pair_worker_fn, base_idxs_for_workers, chunksize=1))

            for p in all_pairs:
                pairs.extend(p)
            print(f"{len(pairs)=}")

            return pairs

    def find_pair_worker_fn(self, base_idxs):
        high_model = self.high_model_class(**self.high_model_params)

        base_dataset = self.base_dataset
        batch_size = self.pair_finding_batch_size

        idx2high_node = [n for n in high_model.nodes if n not in self.ignored_nodes]
        high_node2idx = {n: i for i, n in enumerate(idx2high_node)}
        intervened_high_nodes = {n for n in self.mapping if n not in {"input", "root"}}
        # assume only one high node for now
        ivn_high_node_idx = high_node2idx[list(intervened_high_nodes)[0]]

        pairs = []
        for base_idx in base_idxs:
            base_input_tuple = base_dataset[base_idx]
            base_input_value = base_input_tuple[-2]
            input_len = base_input_value.shape[0]
            # print(f"{base_input_value.shape=}")
            num_remaining = self.num_ivn_srcs
            # single_base_gi = antra.GraphInput({"input": base_input_value})
            base_inputs = base_input_value.expand(batch_size, input_len)
            base_gi = antra.GraphInput.batched({"input": base_inputs}, cache_results=True)

            batch_start = 0
            num_batches = 0
            ivn_src_idxs = torch.randperm(len(base_dataset))
            # pbar = tqdm(total=math.ceil(len(base_dataset) / batch_size), desc="Finding partitions")
            while num_remaining > 0 and batch_start < len(base_dataset):
                # the following is a batched version of partition code
                if num_batches == 8: break
                batch_end = min(len(base_dataset), batch_start+batch_size)
                batch_idxs = ivn_src_idxs[batch_start:batch_end]
                ivn_src_inputs = [base_dataset[k][-2] for k in batch_idxs]
                ivn_src_inputs = torch.stack(ivn_src_inputs, dim=0)
                ivn_src_gi = antra.GraphInput.batched({"input": ivn_src_inputs})
                all_ivn_src_values = high_model.compute_all_nodes(ivn_src_gi)

                partitions = torch.zeros((len(idx2high_node), batch_size), dtype=torch.long)
                # partitions = [defaultdict(set) for _ in range(batch_size)]
                for i, node in enumerate(idx2high_node):
                    if node == "input": continue
                    ivn = antra.Intervention.batched(
                        base_gi, {node: all_ivn_src_values[node]},
                        cache_results=False
                    )
                    _, ivn_res = high_model.intervene(ivn)
                    # print(f"{i} {node} {ivn_res=}")
                    partitions[i] = ivn_res

                    # for k, res in enumerate(ivn_res):
                    #     ser_res = serialize(res)
                    #     partitions[k][ser_res].add(node)
                # print(f"{partitions=}")

                partition_to_isolate = partitions[ivn_high_node_idx].unsqueeze(0)
                overlap_sizes = (partitions == partition_to_isolate).sum(dim=0)
                accept = (overlap_sizes == 1)
                num_remaining -= accept.sum().item()
                accepted_idxs = batch_idxs[accept]
                pairs.extend((base_idx.item(), ivn_src_idx.item()) for ivn_src_idx in accepted_idxs)

                batch_start = batch_end
                num_batches += 1

            high_model.clear_caches(base_gi)
        return pairs
