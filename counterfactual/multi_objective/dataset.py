import torch

import antra
from antra.counterfactual import dataset as antra_cfd
from antra.interchange.batched import pack_graph_inputs
from antra.interchange.mapping import AbstractionMapping

from counterfactual.dataset import MQNLIImpactfulCFDataset

from compgraphs.mqnli_logic import Full_MQNLI_Logic_CompGraph
from datasets.mqnli import MQNLIBertDataset


# [ <CLS> | not | every | bad  | singer | does | not | badly | sings | <e> | every | good | song  | <SEP> | ]
#         | 0           | 1    | 2      | 3          | 4     | 5     | 6           | 7    | 8     |
#         | 9           | 10   | 11     | 12         | 13    | 14    | 15          | 16   | 17    |
#
#  0      | 1   | 2     | 3    | 4      | 5    | 6   | 7     | 8     | 9   | 10    | 11   | 12    | 13    | -- BERT premise
#         | 14  | 15    | 16   | 17     | 18   | 19  | 20    | 21    | 22  | 23    | 24   | 25    | 26    | -- BERT hypothesis
# -----------------------------------------------------------------------------------------------------------
#         | sentence_q  | subj | subj   | neg        | v     | v     | vp_q        | obj  | obj   |
#         |             | _adj | _noun  |            | _adv  | _verb |             | _adj | _noun |
# -----------------------------------------------------------------------------------------------------------
#                       | ---- subj --- |            | --- v_bar --- |             | --- obj ---  |
#                                                    |    ---------------- vp -----------------   |
#                                       |   ------------------- negp -------------------------    |

high_node_to_high_model_idx = {
    "sentence_q": [0, 9],
    "subj_adj": [1, 10],
    "subj_noun": [2, 11],
    "neg": [3, 12],
    "v_adv": [4, 13],
    "v_verb": [5, 14],
    "vp_q": [6, 15],
    "obj_adj": [7, 16],
    "obj_noun": [8, 17],

    "obj": [7, 8, 16, 17],
    "v_bar": [4, 5, 13, 14],
    "subj": [1, 2, 10, 11],

    "vp": [4, 5, 6, 7, 8, 13, 14, 15, 16, 17],
    "negp": [3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17],
}

high_node_to_bert_idx = {
    "sentence_q": [1, 2, 14, 15],
    "subj_adj": [3, 16],
    "subj_noun": [4, 17],
    "neg": [5, 6, 18, 19],
    "v_adv": [7, 20],
    "v_verb": [8, 21],
    "vp_q": [9, 10, 22, 23],
    "obj_adj": [11, 24],
    "obj_noun": [12, 25],

    "obj": [11, 12, 24, 25],
    "v_bar": [7, 8, 20, 21],
    "subj": [3, 4, 16, 17],

    "vp": [7, 8, 9, 10, 11, 12, 20, 21, 22, 23, 24, 25],
    "negp": [5, 6, 7, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22, 23, 24, 25],
}

class MQNLIMultiObjectiveDataset(MQNLIImpactfulCFDataset):
    def __init__(
            self,
            base_dataset: MQNLIBertDataset,
            high_model: Full_MQNLI_Logic_CompGraph,
            mapping: AbstractionMapping,
            num_random_bases=50000,
            num_random_ivn_srcs=20,
            impactful_ratio=0.5,
            max_attempts=10,
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

        super(MQNLIMultiObjectiveDataset, self).__init__(
            base_dataset=base_dataset,
            high_model=high_model,
            mapping=mapping,
            num_random_bases=num_random_bases,
            num_random_ivn_srcs=num_random_ivn_srcs,
            impactful_ratio=impactful_ratio,
            max_attempts=max_attempts,
            fix_examples=fix_examples
        )

        self.bert_replace_idx = torch.tensor(high_node_to_bert_idx[self.intervened_high_node],
                                             dtype=torch.long)
        self.high_replace_idx = torch.tensor(high_node_to_high_model_idx[self.intervened_high_node],
                                            dtype=torch.long)

    def prepare_one_example(self, base, ivn_src, base_idx, ivn_src_idx):
        # tokenizer = self.base_dataset.tokenizer
        high_ivn = self.construct_high_intervention(base, ivn_src)
        low_base_input = self.construct_low_input(base)
        low_ivn_src = self.construct_low_input(ivn_src)

        low_new_input_tensor = base[0].detach().clone()
        # print("-------")
        # print(f"{low_new_input_tensor.shape=}")
        # print(f"Low input before: {tokenizer.decode(low_new_input_tensor)}\n")
        low_new_input_tensor[self.bert_replace_idx] = ivn_src[0][self.bert_replace_idx]
        # print(f"Low input after: {tokenizer.decode(low_new_input_tensor)}\n")
        gi_dict = {
            "input_ids": low_new_input_tensor,
            "token_type_ids": base[1], # assume token_type_ids and attn mask
            "attention_mask": base[2]  # remain unchanged
        }

        low_new_gi = antra.GraphInput(gi_dict, cache_results=False)

        hi_base_gi = antra.GraphInput({"input": base[-2].unsqueeze(0)}, cache_results=False)

        with torch.no_grad():
            base_label = self.high_model.compute(hi_base_gi)

        hi_new_input_tensor = base[-2].detach().clone()
        # print(f"High input before: {tokenizer.decode(hi_new_input_tensor)}\n")
        hi_new_input_tensor[self.high_replace_idx] = ivn_src[-2][self.high_replace_idx]
        # print(f"High input after: {tokenizer.decode(hi_new_input_tensor)}")
        hi_new_gi = antra.GraphInput({"input": hi_new_input_tensor.unsqueeze(0)}, cache_results=False)
        with torch.no_grad():
            aug_label = self.high_model.compute(hi_new_gi)

        return {
            "high_intervention": high_ivn,  # ok
            "low_base_input": low_base_input,  # ok
            "low_intervention_source": low_ivn_src,  # ok
            "low_aug_input": low_new_gi,  # ok
            "base_label": base_label.squeeze(0),
            "aug_label": aug_label.squeeze(0),  # ok
            "mapping": self.mapping,  # ok
            "base_idx": base_idx,  # ok
            "ivn_src_idx": ivn_src_idx  # ok
        }

    def collate_fn(self, batch):
        res = antra_cfd.cf_collate_fn(batch, self.mapping, self.batch_dim)

        low_aug_input = batch[0]["low_aug_input"]
        low_key_leaves, low_non_batch_leaves = low_aug_input.key_leaves, low_aug_input.non_batch_leaves

        low_aug_input_gi_dict = pack_graph_inputs(
            [d["low_aug_input"] for d in batch],
            batch_dim=self.batch_dim,
            non_batch_inputs=low_non_batch_leaves
        )

        low_aug_input = antra.GraphInput.batched(
            low_aug_input_gi_dict,
            batch_dim=self.batch_dim,
            cache_results=False,
            key_leaves=low_key_leaves,
            non_batch_leaves=low_non_batch_leaves
        )

        res['low_aug_input'] = low_aug_input
        res['aug_labels'] = torch.stack([d["aug_label"] for d in batch])
        res['base_labels'] = torch.stack([d['base_label'] for d in batch])
        return res
