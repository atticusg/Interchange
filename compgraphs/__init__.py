from compgraphs.mqnli_lstm import MQNLI_LSTM_CompGraph, Abstr_MQNLI_LSTM_CompGraph
from compgraphs.mqnli_bert import MQNLI_Bert_CompGraph, Abstr_MQNLI_Bert_CompGraph

_name_to_compgraph_class = {
    "lstm": MQNLI_LSTM_CompGraph,
    "bert": MQNLI_Bert_CompGraph,
}

def get_compgraph_class_by_name(name: str):
    return _name_to_compgraph_class[name]


_name_to_abstr_compgraph_class = {
    "lstm": Abstr_MQNLI_LSTM_CompGraph,
    "bert": Abstr_MQNLI_Bert_CompGraph,
}

def get_abstr_compgraph_class_by_name(name: str):
    return _name_to_abstr_compgraph_class[name]