import torch
import numpy as np
from collections import Counter
import torch.utils.data as torch_data

from antra import LOC
from counterfactual.dataset import MQNLIRandomIterableCfDataset
from counterfactual.multidataloader import MultiTaskDataLoader
from counterfactual.scheduling import LinearCfTrainingSchedule
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


def mYSchEDule(epoch):
    warmup = [6, 0, 0]
    phase1 = [5, 1, 1]
    phase2 = [4, 3, 2]

    if epoch == 0:
        return warmup
    elif 1 <= epoch <= 2:
        return phase1
    else:
        return phase2


def test_multi_dataloader():
    data0 = [0, 1, 2, 3, 4, 5]
    data1 = [10, 11, 12]
    data2 = [20, 21]

    dl0 = torch_data.DataLoader(data0, shuffle=True)
    dl1 = torch_data.DataLoader(data1, shuffle=True)
    dl2 = torch_data.DataLoader(data2, shuffle=True)

    mtdl = MultiTaskDataLoader([dl0, dl1, dl2], mYSchEDule, return_task_name=True)
    task_cnt = Counter()
    d0cnt = Counter()
    d1cnt = Counter()
    d2cnt = Counter()
    num_elems = [6, 3, 2]
    rescnt = [d0cnt, d1cnt, d2cnt]
    n_epochs = 6

    for epoch in range(n_epochs):
        for ex in mtdl:
            res, task = ex
            task_cnt[task] += 1
            rescnt[task][res.item()] += 1

    assert task_cnt[0] == 6 + 5 * 2 + (n_epochs - 3) * 4
    assert task_cnt[1] == 1 * 2 + (n_epochs - 3) * 3
    assert task_cnt[2] == 1 * 2 + (n_epochs - 3) * 2

    print(f"{rescnt=}")

    for task, cnts in enumerate(rescnt):
        tasklen = num_elems[task]
        full_cycles = task_cnt[task] // tasklen

        for elem, cnt in cnts.items():
            assert cnt == full_cycles or cnt == full_cycles + 1


def test_lin_schedule():
    base_dataset = torch.zeros((240002,))
    batch_size = 32

    s = LinearCfTrainingSchedule(
        base_dataset, batch_size,
    )

    for epoch in range(40):
        print(f"Epoch {epoch}, schedule {s(epoch)}")

