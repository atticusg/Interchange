import torch
import re

from intervention import ComputationGraph, GraphNode, LOC
from compgraphs.abstractable import AbstractableCompGraph

from typing import List, Any, Optional


def generate_lstm_fxn(lstm_layer):
    def _lstm_layer_fxn(x):
        res, _ = lstm_layer(x)
        return res
    return _lstm_layer_fxn


class MQNLI_LSTM_CompGraph(ComputationGraph):
    def __init__(self, lstm_model, root_output_device=None):
        self.model = lstm_model

        @GraphNode()
        def input(x):
            return x

        self.h_dim = self.model.lstm_hidden_dim * 2

        @GraphNode(input)
        def embed(x):
            # assert x[0].shape[0] == 19, f"x.shape is {x.shape}"
            # print("embedding input shape", x.shape)
            res = self.model.embedding(x)
            # print("embedding output shape", res.shape)
            return res

        lstm_node = embed
        for i in range(len(self.model.lstm_layers)):
            lstm_forward_fxn = generate_lstm_fxn(self.model.lstm_layers[i])
            lstm_node = GraphNode(lstm_node, name=f"lstm_{i}", forward=lstm_forward_fxn)

        @GraphNode(lstm_node)
        def concat_final_state(x):
            return self.model.concat_final_state(x)

        @GraphNode(concat_final_state)
        def ffnn1(x):
            # (batch_size, self.h_dim)
            assert (len(x.shape) == 2 and x.shape[1] == self.h_dim)
            return self.model.feed_forward1(x)

        @GraphNode(ffnn1)
        def activation1(x):
            # (batch_size, self.h_dim)
            assert (len(x.shape) == 2 and x.shape[1] == self.h_dim)
            return self.model.activation1(x)

        @GraphNode(activation1)
        def ffnn2(x):
            # (batch_size, self.h_dim)
            assert (len(x.shape) == 2 and x.shape[1] == self.h_dim)
            return self.model.feed_forward2(x)

        @GraphNode(ffnn2)
        def activation2(x):
            assert (len(x.shape) == 2 and x.shape[1] == self.h_dim)
            return self.model.activation2(x)

        @GraphNode(activation2)
        def logits(x):
            assert (len(x.shape) == 2 and x.shape[1] == self.h_dim)
            return self.model.logits(x)

        @GraphNode(logits)
        def root(x):
            return torch.argmax(x, dim=1)
        super(MQNLI_LSTM_CompGraph, self).__init__(root, root_output_device=root_output_device)

    @property
    def device(self):
        return self.model.device


class Abstr_MQNLI_LSTM_CompGraph(AbstractableCompGraph):
    def __init__(self, base_compgraph: MQNLI_LSTM_CompGraph,
                 intermediate_nodes: List[str], interv_info: Any = None,
                 root_output_device: Optional[torch.device]=None):
        self.base = base_compgraph
        self.interv_info = interv_info

        full_graph = {node_name: [child.name for child in node.children]
                      for node_name, node in base_compgraph.nodes.items()}

        forward_functions = {node_name: node.forward
                        for node_name, node in base_compgraph.nodes.items()}

        super(Abstr_MQNLI_LSTM_CompGraph, self).__init__(
            full_graph=full_graph,
            root_node_name="root",
            abstract_nodes=intermediate_nodes,
            forward_functions=forward_functions,
            root_output_device=root_output_device,
        )

    @property
    def device(self):
        return self.base.device

    def get_indices(self, node: str):
        if re.match(r".*lstm_[0-9]*", node):
            return [LOC[:,i,:] for i in self.interv_info["target_locs"]]
        else:
            return super().get_indices(node)
