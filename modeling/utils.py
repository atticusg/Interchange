import math
import torch
import torch.nn as nn

class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, fix_weights=False, scale_by_dim=False):
        super(EmbeddingModule, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.fix_weights = fix_weights
        self.scale_by_dim = scale_by_dim

        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim)
        if fix_weights:
            for param in self.embedding.parameters():
                param.requires_grad = False

    def forward(self, input_tuple):
        res = self.embedding(input_tuple[0])
        if self.scale_by_dim:
            return res * math.sqrt(self.embedding_dim)
        else:
            return res

class MeanModule(nn.Module):
    def __init__(self, length_dim):
        super(MeanModule, self).__init__()
        self.length_dim = length_dim

    def forward(self, emb_x, input_tuple):
        sums = emb_x.sum(dim=self.length_dim)
        lengths = input_tuple[2]
        count = lengths.unsqueeze(-1)
        return sums / count

class ConcatModule(nn.Module):
    def __init__(self, dim):
        super(ConcatModule, self).__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat((x, y), self.dim)

def init_scaling(scaling, *args):
    for submodule in args:
        nn.init.uniform_(submodule.weight, -scaling, scaling)

def modularize(forward_fxn, init_fxn=None, name=None):
    if not name:
        name = forward_fxn.__name__

    submodule_class = type(name, (nn.Module,), {
        "__init__": None
    })
    class SubModule(nn.Module):
        def __init__(self):
            super(SubModule, self).__init__()

    return SubModule()

def get_target_loc_dict(data_variant):
    if "lstm" in data_variant:
        d = {"sentence_q": [0, 10],
             "subj_adj": [1, 11],
             "subj_noun": [2, 12],
             "neg": [3, 13],
             "v_adv": [4, 14],
             "v_verb": [5, 15],
             "vp_q": [6, 16],
             "obj_adj": [7, 17],
             "obj_noun": [8, 18],
             "obj": [7, 8, 17, 18],
             "vp": [6, 16],
             "v_bar": [4, 5, 14, 15],
             "negp": [3, 13],
             "subj": [1, 2, 11, 12]}
    elif "bert" in data_variant:
        d = {"sentence_q": [0, 1, 2, 14, 15],
             "subj_adj": [0, 3, 16],
             "subj_noun": [0, 4, 17],
             "neg": [0, 5, 6, 18, 19],
             "v_adv": [0, 7, 20],
             "v_verb": [0, 8, 21],
             "vp_q": [0, 9, 10, 22, 23],
             "obj_adj": [0, 11, 24],
             "obj_noun": [0, 12, 25],
             "obj": [0, 11, 12, 24, 25],
             "vp": [0, 8, 9, 10, 21, 22, 23],
             "v_bar": [0, 7, 8, 20, 21],
             "negp": [0, 5, 6, 18, 19],
             "subj": [0, 3, 4, 16, 17]}
    else:
        raise ValueError(f"Cannot recognize data variant {data_variant}")
    return d

def get_target_locs(high_node_name: str, loc_mapping_type: str= "lstm"):
    """ Get list of indices for locations to intervene given type of model

    :param high_node_name:
    :param loc_mapping_type: type of model and type of mapping
    :return:
    """
    if "lstm" in loc_mapping_type:
        # mapping for lstm model
        d = {"sentence_q": [0, 10],
             "subj_adj": [1, 11],
             "subj_noun": [2, 12],
             "neg": [3, 13],
             "v_adv": [4, 14],
             "v_verb": [5, 15],
             "vp_q": [6, 16],
             "obj_adj": [7, 17],
             "obj_noun": [8, 18],
             "obj": [7, 8, 17, 18],
             "vp": [6, 16],
             "v_bar": [4, 5, 14, 15],
             "negp":[3, 13],
             "subj": [1, 2, 11, 12]}

        return d[high_node_name]

    if loc_mapping_type == "bert_cls_only":
        d = {"sentence_q": [0],
             "subj_adj": [0],
             "subj_noun": [0],
             "neg": [0],
             "v_adv": [0],
             "v_verb": [0],
             "vp_q": [0],
             "obj_adj": [0],
             "obj_noun": [0],
             "obj": [0],
             "vp": [0],
             "v_bar": [0],
             "negp": [0],
             "subj": [0]}
        return d[high_node_name]

    if "bert" in loc_mapping_type:
        # mapping for bert model
        # [ <CLS> | not | every | bad | singer | does | not | badly | sings | <e> | every | good | song ]
        #  0        1     2       3     4        5      6     7       8       9     10      11     12

        d = {"sentence_q": [0, 1, 2, 13, 14, 15, 26],
             "subj_adj": [0, 3, 13, 16, 26],
             "subj_noun": [0, 4, 13, 17, 26],
             "neg": [0, 5, 6, 13, 18, 19, 26],
             "v_adv": [0, 7, 13, 20, 26],
             "v_verb": [0, 8, 13, 21, 26],
             "vp_q": [0, 9, 10, 13, 22, 23, 26],
             "obj_adj": [0, 11, 13, 24, 26],
             "obj_noun": [0, 12, 13, 25, 26],
             "obj": [0, 11, 12, 13, 24, 25, 26],
             "vp": [0, 8, 9, 10, 13, 21, 22, 23, 26],
             "v_bar": [0, 7, 8, 13, 20, 21, 26],
             "negp": [0, 5, 6, 13, 18, 19, 26],
             "subj": [0, 3, 4, 13, 16, 17, 26]}
        return d[high_node_name]