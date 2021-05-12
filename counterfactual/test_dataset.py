import torch
import numpy as np

import torch.utils.data as torch_data
from antra import LOC
from counterfactual.dataset import MQNLIRandomIterableCfDataset
from compgraphs.mqnli_logic import Full_MQNLI_Logic_CompGraph

# def test_dataset_constructor():
#     seed = 39
#     np.random.seed(seed)
#     torch.random.manual_seed(seed)
#
#     dataset_path = "data/mqnli/preprocessed/bert-hard_abl.pt"
#     print("loading dataset")
#     data = torch.load(dataset_path)
#     base_dataset = data.train
#     mapping = {
#         "vp": {"bert_layer_3": LOC[:,10,:]}
#     }
#     pairs_pickle_path = "data/counterfactual/test_pairs.pk"
#     high_model = Full_MQNLI_Logic_CompGraph(data)
#     IsolatedPairFinder(
#         Full_MQNLI_Logic_CompGraph,
#         {"data": data}, base_dataset, mapping,
#         pairs_pickle_path=pairs_pickle_path,
#         num_base_inputs=200,
#         num_ivn_srcs=32,
#         pair_finding_batch_size=24,
#         num_workers=2,
#     )


def test_dataset_constructor():
    seed = 39
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    dataset_path = "data/mqnli/preprocessed/bert-hard_abl.pt"
    print("loading dataset")
    data = torch.load(dataset_path)
    base_dataset = data.train
    mapping = {
        "vp": {"bert_layer_3": LOC[:,10,:]}
    }
    high_model = Full_MQNLI_Logic_CompGraph(data)
    dataset = MQNLIRandomIterableCfDataset(
        base_dataset, high_model, mapping
    )
    # sampler = torch_data.RandomSampler(dataset)
    dataloader = torch_data.DataLoader(
        dataset, batch_size=4, collate_fn=dataset.collate_fn)

    for i, batch in enumerate(dataloader):
        # print(batch)
        if i == 500: break
        low_base = batch["low_base_input"]
        low_ivn_src = batch["low_intervention_source"]
        # base_input_ids = low_base.values['input_ids']
        # print(f"{base_input_ids.shape=}")

        for j, (base_idx, ivn_src_idx) in enumerate(zip(batch['base_idxs'], batch['ivn_src_idxs'])):
            base_ex = base_dataset[base_idx]
            ivn_src_ex = base_dataset[ivn_src_idx]

            for k, arg in enumerate(["input_ids", "token_type_ids", "attention_mask"]):
                assert torch.allclose(base_ex[k], low_base[arg][j])
                assert torch.allclose(ivn_src_ex[k], low_ivn_src[arg][j])
