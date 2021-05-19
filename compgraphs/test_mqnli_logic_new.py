import torch
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

from antra import GraphInput
from antra.utils import serialize

import compgraphs.mqnli_logic as mqnli_logic
from datasets.mqnli import MQNLIData

nodes_to_compare = ["obj_noun", "obj_adj", "obj",
                    "subj_noun", "subj_adj", "subj",
                    "v_verb", "v_adv", "v_bar", "vp_q", "vp",
                    "neg", "negp", "sentence_q", "root"]

def test_full_compgraph():
    mqnli_data = MQNLIData("data/mqnli/raw/easy/train.txt",
                           "data/mqnli/raw/easy/dev.txt",
                           "data/mqnli/raw/easy/test.txt")


    graph = mqnli_logic.Full_MQNLI_Logic_CompGraph(mqnli_data)
    # abstr_graph = mqnli_logic.Full_Abstr_MQNLI_Logic_CompGraph(graph, intermediate_nodes)

    old_graph = mqnli_logic.MQNLI_Logic_CompGraph(mqnli_data)

    with torch.no_grad():
        # collate_fn = lambda batch: my_collate(batch, batch_first=False)
        dataloader = DataLoader(mqnli_data.train, batch_size=2048, shuffle=False)

        start_time = time.time()
        for i, input_tuple in enumerate(tqdm(dataloader)):
            input_values = input_tuple[0]
            input_values = torch.cat((input_values[:, :9], input_values[:, -9:]), dim=1)
            # print(f"{input_values.shape=}")
            keys = [serialize(x) for x in input_values]
            graph_input = GraphInput.batched({"input": input_values}, cache_results=False)
            # new_graph_pred = graph.compute(graph_input)
            graph_pred = graph.compute_all_nodes(graph_input)
            # pprint(graph_pred)

            old_graph_input = GraphInput.batched({"input": input_values.T}, keys=keys, cache_results=False)
            # old_graph_root = old_graph.compute(old_graph_input)
            old_graph_pred = old_graph.compute_all_nodes(old_graph_input)
            # pprint(old_graph_pred)
            labels = input_tuple[1]

            for node in nodes_to_compare:
                assert torch.all(graph_pred[node] == old_graph_pred[node])

            # assert torch.all(old_graph_root == labels)
            assert torch.all(graph_pred["root"] == labels)
            # break

            # assert torch.all(labels == graph_pred), f"on batch {i}"
        duration = time.time() - start_time
        print(f"---- Ran {len(mqnli_data.train)} examples in {duration:.2f} s ----")
