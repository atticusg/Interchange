from modeling.lstm import LSTMModule
from modeling.pretrained_bert import PretrainedBertModule
from modeling.raw_bert import RawBertModule

_name_to_module = {
    "lstm": LSTMModule,
    "bert": PretrainedBertModule,
    "raw_bert": RawBertModule
}

def get_module_class_by_name(name: str):
    return _name_to_module[name]