import torch
import re

from intervention import ComputationGraph, GraphNode
from compgraphs.abstractable import AbstractableCompGraph

class MQNLI_Bert_CompGraph(ComputationGraph):
    def __init__(self, bert_model):
        if bert_model.task != "mqnli":
            raise ValueError("The model must be for MQNLI!")

        self.model = bert_model

        @GraphNode()
        def input(x):
            return x

        bert = self.model.bert

        @GraphNode(input)
        def embed(x):
            input_ids = x[0]
            token_type_ids = x[1]