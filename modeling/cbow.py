import torch
import torch.nn as nn
from modeling.utils import EmbeddingModule, MeanModule, ConcatModule, \
    init_scaling


class PremiseMeanModule(nn.Module):
    def __init__(self, length_dim):
        super(PremiseMeanModule, self).__init__()
        self.length_dim = length_dim

    def forward(self, x):
        return x[:9,:,:].mean(dim=self.length_dim)


class HypothesisMeanModule(nn.Module):
    def __init__(self, length_dim):
        super(HypothesisMeanModule, self).__init__()
        self.length_dim = length_dim

    def forward(self, x):
        return x[9:,:,:].mean(dim=self.length_dim)


class CBOWModule(nn.Module):
    def __init__(self, task="sentiment", output_classes=2, vocab_size=10000,
                 hidden_dim=100, activation_type="relu", dropout=0.1,
                 embed_init_scaling=0.1, fix_embeddings=False,
                 batch_first=False, device=None):
        super(CBOWModule, self).__init__()

        self.task = task
        self.output_classes = output_classes
        self.vocab_size = vocab_size

        self.hidden_dim = hidden_dim
        self.activation_type = activation_type
        self.dropout = dropout
        self.embed_init_scaling = embed_init_scaling
        self.fix_embeddings = fix_embeddings
        self.batch_first = batch_first
        self.device = device if device else torch.device("cpu")

        self.embedding = EmbeddingModule(num_embeddings=vocab_size,
                                         embedding_dim=hidden_dim,
                                         fix_weights=fix_embeddings)

        print("Embedding Module:", self.embedding)
        if task == "sentiment":
            self.mean = MeanModule(length_dim=(1 if batch_first else 0))
        elif task == "mqnli":
            self.premise_mean = PremiseMeanModule(length_dim=(1 if batch_first else 0))
            self.hypothesis_mean = HypothesisMeanModule(length_dim=(1 if batch_first else 0))
            self.concat = ConcatModule(dim=1)
        else:
            raise ValueError("Task type \'%s\' is undefined" % self.task)

        if task == "mqnli":
            hidden_dim *= 2

        self.feed_forward1 = nn.Linear(hidden_dim, hidden_dim)
        self.feed_forward2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim,output_classes)

        if self.activation_type == "relu":
            self.activation1 = nn.ReLU()
            self.activation2 = nn.ReLU()
        elif self.activation_type == "tanh":
            self.activation1 = nn.Tanh()
            self.activation2 = nn.Tanh()
        else:
            raise ValueError("CBOWModule does not support this activation type")

        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        init_scaling(embed_init_scaling, self.embedding.embedding)
        print("Initialized model with parameters:\n", self.config())

    def config(self):
        return {
            "task": self.task,
            "output_classes": self.output_classes,
            "vocab_size": self.vocab_size,

            "hidden_dim": self.hidden_dim,
            "activation_type": self.activation_type,
            "dropout": self.dropout,
            "embed_init_scaling": self.embed_init_scaling,
            "fix_embeddings": self.fix_embeddings,
            "batch_first": self.batch_first,
        }

    def forward(self, input_tuple):
        emb = self.embedding(input_tuple)
        emb = self.dropout0(emb)

        if self.task == "sentiment":
            input_repr = self.mean(emb, input_tuple)
        elif self.task == "mqnli":
            prem_repr = self.premise_mean(emb)
            hyp_repr = self.hypothesis_mean(emb)
            input_repr = self.concat(prem_repr, hyp_repr)
        else:
            raise ValueError("Task type \'%s\' is undefined" % self.task)

        output = self.feed_forward1(input_repr)
        output = self.activation1(output)
        output = self.dropout1(output)

        output = self.feed_forward2(output)
        output = self.activation2(output)
        output = self.dropout2(output)

        output = self.logits(output)
        return output

