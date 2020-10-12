from intervention.abstraction_torch import find_abstractions

from intervention import Intervention, LOC
from datasets.mqnli import MQNLIData
from datasets.utils import my_collate
from modeling.lstm import LSTMModule
from train import load_model

from compgraphs import mqnli_logic
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph
from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph

import time
import torch
from torch.utils.data import DataLoader


loc_mapping = {
    "obj_adj": LOC[mqnli_logic.IDX_A_O],
}

def main():
    data_path = "mqnli_data/mqnli.pt"
    # data = MQNLIData("mqnli_data/mqnli.train.txt",
    #                  "mqnli_data/mqnli.dev.txt",
    #                  "mqnli_data/mqnli.test.txt")
    # torch.save(data, data_path)

    high_intermediate_node = "obj_adj"

    model_path = "mqnli_models/lstm_best.pt"
    high_intermediate_nodes = [high_intermediate_node]
    low_intermediate_nodes = ["premise_lstm_0"]
    num_inputs = 100

    data = torch.load(data_path)

    module, _ = load_model(LSTMModule, model_path, device=torch.device("cpu"))
    module.eval()

    interv_info = {
        "target_loc": loc_mapping[high_intermediate_node]
    }

    base_compgraph = MQNLI_LSTM_CompGraph(module)
    low_model = Abstr_MQNLI_LSTM_CompGraph(base_compgraph,
                                           low_intermediate_nodes,
                                           interv_info=interv_info)
    high_model = MQNLI_Logic_CompGraph(data, high_intermediate_nodes)

    collate_fn = lambda batch: my_collate(batch, batch_first=False)
    dataloader = DataLoader(data.dev, batch_size=1, shuffle=False,
                            collate_fn=collate_fn)

    inputs = []
    high_interventions = []
    for i, input_tuple in enumerate(dataloader):
        if i == num_inputs: break

        input_value = input_tuple[0].to(module.device)
        base_input = Intervention({"input": input_value}, {})
        inputs.append(base_input)
        for proj in mqnli_logic.intersective_projections:
            intervention = Intervention({"input": input_value}, {high_intermediate_node: proj})
            high_interventions.append(intervention)
    fixed_assignments = {x: {x: LOC[:]} for x in ["root", "input"]}
    input_mapping = lambda x : x #identity function

    unwanted_nodes = {"root", "input"}

    start_time = time.time()
    with torch.no_grad():
        res = find_abstractions(
            low_model=low_model,
            high_model=high_model,
            high_inputs=inputs,
            total_high_interventions=high_interventions,
            fixed_assignments=fixed_assignments,
            input_mapping=input_mapping,
            unwanted_low_nodes=unwanted_nodes
        )
        duration = time.time() - start_time
        print(f"Finished finding abstractions, took {duration:.2f} s")
    

if __name__ == "__main__":
    main()