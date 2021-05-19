import antra
import torch
import re

from antra import ComputationGraph, GraphNode, LOC
from antra.abstractable import AbstractableCompGraph
from antra.compgraphs.bert import generate_bert_compgraph
from typing import List, Any, Optional

def generate_bert_layer_fxn(layer_module, i):
    def _bert_layer_fxn(hidden_states, metainfo):
        attention_mask = metainfo.get("attention_mask", None)
        head_mask = metainfo.get("head_mask", None)
        encoder_hidden_states = metainfo.get("encoder_hidden_states", None)
        encoder_attention_mask = metainfo.get("encoder_attention_mask", None)
        output_attentions = metainfo.get("output_attentions", False)

        layer_head_mask = head_mask[i] if head_mask is not None else None
        layer_outputs = layer_module(
            hidden_states,
            attention_mask,
            layer_head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        hidden_states = layer_outputs[0]
        return hidden_states
    return _bert_layer_fxn

class MQNLI_Bert_CompGraph(ComputationGraph):
    def __init__(self, bert_model):
        if bert_model.task != "mqnli":
            raise ValueError("The model must be for MQNLI!")

        self.model = bert_model

        input = antra.GraphNode.leaf("input")

        bert = self.model.bert

        @GraphNode(input)
        def metainfo(input_tuple):
            input_ids = input_tuple[0]
            attention_mask = input_tuple[2]
            input_shape = input_ids.shape
            device = input_ids.device

            extended_attention_mask = bert.get_extended_attention_mask(
                attention_mask, input_shape, device)
            return {"attention_mask": extended_attention_mask,
                    "head_mask": [None] * 12,
                    "encoder_hidden_states": None,
                    "encoder_extended_attention_mask": None,
                    "output_attentions": False,
                    "output_hidden_states": False,
                    "return_dict": False}

        @GraphNode(input)
        def embed(input_tuple):
            input_ids = input_tuple[0]
            token_type_ids = input_tuple[1]
            return bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids) # [B, 27, 768]

        hidden_node = embed # node that outputs hidden vector values
        for i in range(len(bert.encoder.layer)):
            f = generate_bert_layer_fxn(bert.encoder.layer[i], i)
            hidden_node = GraphNode(hidden_node, metainfo,
                                    name=f"bert_layer_{i}",
                                    forward=f)

        @GraphNode(hidden_node)
        def pool(h):
            return bert.pooler(h)

        @GraphNode(pool)
        def logits(x):
            return self.model.logits(x)

        @GraphNode(logits)
        def root(x):
            return torch.argmax(x, dim=1)
        
        super(MQNLI_Bert_CompGraph, self).__init__(root)

    @property
    def device(self):
        return self.model.device


class Abstr_MQNLI_Bert_CompGraph(AbstractableCompGraph):
    def __init__(self, base_compgraph: MQNLI_Bert_CompGraph,
                 intermediate_nodes: List[str]):
        self.base = base_compgraph
        # self.interv_info = interv_info

        # full_graph = {node_name: [child.name for child in node.children]
        #               for node_name, node in base_compgraph.nodes.items()}
        #
        # forward_functions = {node_name: node.forward
        #                 for node_name, node in base_compgraph.nodes.items()}

        super(Abstr_MQNLI_Bert_CompGraph, self).__init__(
            graph=base_compgraph,
            abstract_nodes=intermediate_nodes,
        )

    @property
    def device(self):
        return self.base.device

    # def get_indices(self, node: str):
    #     if re.match(r".*bert_layer_[0-9]*", node):
    #         return [LOC[:,i,:] for i in self.interv_info["target_locs"]]
    #     else:
    #         raise ValueError(f"Cannot get indices for node {node}")


class Full_MQNLI_Bert_CompGraph(ComputationGraph):
    def __init__(self, bert_model):
        if bert_model.task != "mqnli":
            raise ValueError("The model must be for MQNLI!")

        self.model = bert_model
        bert = self.model.bert

        pool = generate_bert_compgraph(bert, final_node="pool")

        @GraphNode(pool)
        def logits(x):
            return self.model.logits(x)

        # @GraphNode(logits)
        # def root(x):
        #     return torch.argmax(x, dim=1)

        super(Full_MQNLI_Bert_CompGraph, self).__init__(logits)

    @property
    def device(self):
        return self.model.device
