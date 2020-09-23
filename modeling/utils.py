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