import torch
import re

from intervention import ComputationGraph, GraphNode
from compgraphs.abstractable import AbstractableCompGraph

from typing import List, Any, Optional


def generate_lstm_fxn(lstm_layer):
    def _lstm_layer_fxn(x):
        res, _ = lstm_layer(x)
        return res
    return _lstm_layer_fxn


class MQNLI_LSTM_CompGraph(ComputationGraph):
    def __init__(self, lstm_model, root_output_device=None):
        if lstm_model.task != "mqnli":
            raise ValueError("The LSTM model must be for MQNLI!")
        self.model = lstm_model

        @GraphNode()
        def input(x):
            return x

        self.h_dim = self.model.lstm_hidden_dim
        if hasattr(lstm_model, "p_h_separator") and lstm_model.p_h_separator is not None:
            if self.model.bidirectional:
                self.h_dim *= 2

            @GraphNode(input)
            def embed(x):
                if len(x.shape) == 1:
                    x = x.unsqueeze(1)
                assert len(x.shape) == 2 and x.shape[0] == 18, f"x.shape is {x.shape}"
                x = self.model.add_separator((x,))
                return self.model.embedding(x)

            lstm_node = embed
            for i in range(len(self.model.lstm_layers)):
                lstm_forward_fxn = generate_lstm_fxn(self.model.lstm_layers[i])
                lstm_node = GraphNode(lstm_node, name=f"lstm_{i}",
                                      forward=lstm_forward_fxn)

            @GraphNode(lstm_node)
            def concat_final_state(x):
                return self.model.concat_final_state(x)

        else:
            self.h_dim *= 4 if self.model.bidirectional else 4
            @GraphNode(input)
            def embed(x):
                # (18,) or (18, batch_size)
                if len(x.shape) == 1:
                    x = x.unsqueeze(1)
                assert len(x.shape) == 2 and x.shape[0] == 18, f"x.shape is {x.shape}"
                return self.model.embedding((x,))

            @GraphNode(embed)
            def premise_emb(x):
                # (18, emb_size) or (18, batch_size, emb_size)
                assert len(x.shape) == 3 and x.shape[0] == 18 and x.shape[2] == self.model.embed_dim, \
                    f"x.shape is {x.shape}"
                return self.model.premise_emb(x)

            @GraphNode(embed)
            def hypothesis_emb(x):
                # (18, emb_size) or (18, batch_size, emb_size)
                assert len(x.shape) == 3 and x.shape[0] == 18 and x.shape[2] == self.model.embed_dim, \
                    f"x.shape is {x.shape}"
                return self.model.hypothesis_emb(x)

            # Create lstm nodes
            prem_node = premise_emb
            for i in range(len(self.model.lstm_layers)):
                lstm_forward_fxn = generate_lstm_fxn(self.model.lstm_layers[i])
                prem_node = GraphNode(prem_node, name=f"premise_lstm_{i}", forward=lstm_forward_fxn)

            # Create lstm nodes
            hyp_node = hypothesis_emb
            for i, lstm_layer in enumerate(self.model.lstm_layers):
                lstm_forward_fxn = generate_lstm_fxn(self.model.lstm_layers[i])
                hyp_node = GraphNode(hyp_node, name=f"hypothesis_lstm_{i}", forward=lstm_forward_fxn)

            @GraphNode(prem_node, hyp_node)
            def concat_final_state(prem_h, hyp_h):
                # (9, batch_size, self.h_dim)
                self.h_dim = self.model.lstm_hidden_dim * (2 if self.model.bidirectional else 1)
                assert len(prem_h.shape) == 3 and prem_h.shape[0] == 9 and prem_h.shape[2] == self.h_dim, \
                    f"hypothesis_lstm_{i} x.shape is {prem_h.shape}"
                assert len(hyp_h.shape) == 3 and hyp_h.shape[0] == 9 and hyp_h.shape[2] == self.h_dim, \
                    f"hypothesis_lstm_{i} x.shape is {hyp_h.shape}"

                return self.model.concat_final_state(prem_h, hyp_h)

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
            return self.interv_info["target_locs"]
        else:
            return super().get_indices(node)
