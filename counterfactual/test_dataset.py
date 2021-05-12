import torch
import numpy as np

from antra import LOC
from counterfactual.dataset import IsolatedPairFinder
from compgraphs.mqnli_logic import Full_MQNLI_Logic_CompGraph

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
    pairs_pickle_path = "data/counterfactual/test_pairs.pk"
    high_model = Full_MQNLI_Logic_CompGraph(data)
    IsolatedPairFinder(
        Full_MQNLI_Logic_CompGraph,
        {"data": data}, base_dataset, mapping,
        pairs_pickle_path=pairs_pickle_path,
        num_base_inputs=200,
        num_ivn_srcs=32,
        pair_finding_batch_size=24,
        num_workers=2,
    )
    # 536, base=200, num_ivn_srcs=32, pair_finding_bs = 24
