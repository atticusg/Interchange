from modeling.lstm import LSTMModule
from modeling.pretrained_bert import PretrainedBertModule

_name_to_module = {
    "lstm": LSTMModule,
    "bert": PretrainedBertModule
}

def get_module_class_by_name(name: str):
    return _name_to_module[name]