HIGH_NODE_LABEL_SPACE = {
    "root": 3,
    "sentence_q": 16,
    "subj": 4,
    "subj_adj": 4,
    "subj_noun": 2,
    "negp": 7,
    "neg": 4,
    "vp": 7,
    "vp_q": 16,
    "v_bar": 4,
    "v_adv": 4,
    "v_verb": 2,
    "obj": 4,
    "obj_adj": 4,
    "obj_noun": 2
}

def get_num_classes(high_node: str) -> int:
    return HIGH_NODE_LABEL_SPACE[high_node]

def get_low_nodes(model_type):
    if model_type == "lstm":
        return ["lstm_0"]
    elif model_type == "bert":
        return [f"bert_layer_{i}" for i in range(11)]

def get_low_hidden_dim(model_type, model):
    if model_type == "lstm":
        return model.lstm_hidden_dim * 2 if model.bidirectional else model.lstm_hidden_diM
    elif model_type == "bert":
        return model.bert.config.hidden_size