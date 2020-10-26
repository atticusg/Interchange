import pickle
import json
import argparse
import torch

from experiment import Experiment
from train import load_model
from modeling.lstm import LSTMModule

from intervention.analysis import construct_graph, find_cliques
from compgraphs.mqnli_logic import MQNLI_Logic_CompGraph
from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph


class GraphExperiment(Experiment):
    def experiment(self, opts):
        module, _ = load_model(LSTMModule, opts["model_path"],
                               device=torch.device("cpu"))
        module.eval()
        data = torch.load(opts["data_path"], map_location=torch.device('cpu'))
        abstraction = json.loads(opts["abstraction"])

        high_intermediate_node = abstraction[0]
        low_intermediate_nodes = abstraction[1]
        high_intermediate_nodes = [high_intermediate_node]

        base_compgraph = MQNLI_LSTM_CompGraph(module)
        low_model = Abstr_MQNLI_LSTM_CompGraph(base_compgraph,
                                               low_intermediate_nodes)
        high_model = MQNLI_Logic_CompGraph(data, high_intermediate_nodes)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--abstraction", type=str)
    parser.add_argument("--num_inputs", type=int)

    parser.add_argument("--id", type=int)
    parser.add_argument("--db_path", type=str)

    args = parser.parse_args()

    e = GraphExperiment(finished_status=3)
    e.run(vars(args))

if __name__ == "__main__":
    main()