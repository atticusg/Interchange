import torch
import torch.nn as nn
import math

from modeling.utils import EmbeddingModule, init_scaling

class MaskingModule(nn.Module):
    # refer to `BertPooler` class in HuggingFace BERT implementation
    def __init__(self):
        super(MaskingModule, self).__init__()

    def forward(self, input_tuple):
        return (input_tuple[0] == 0).T

class PoolingModule(torch.nn.Module):
    def __init__(self, hidden_dim=64, dropout=0.1):
        super(PoolingModule, self).__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[0,:,:]
        output = self.linear(first_token_tensor)
        output = self.activation(output)
        output = self.dropout(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModule(nn.Module):
    def __init__(self, task="sentiment", hidden_dim=64, num_transformer_heads=4,
                 num_transformer_layers=6, vocab_size=10000, output_classes=3,
                 embed_init_scaling=0.1, fix_embeddings=False, dropout=0.1, device=None):
        super(TransformerModule, self).__init__()

        self.task = task
        self.vocab_size = vocab_size
        self.output_classes = output_classes

        self.hidden_dim = hidden_dim
        self.num_transformer_heads = num_transformer_heads
        self.num_transformer_layers = num_transformer_layers
        self.embed_init_scaling = embed_init_scaling
        self.fix_embeddings = fix_embeddings
        self.dropout = dropout

        self.device = device if device else torch.device("cpu")

        self.masking = MaskingModule()
        self.embedding = EmbeddingModule(num_embeddings=vocab_size,
                                         embedding_dim=hidden_dim,
                                         fix_weights=fix_embeddings,
                                         scale_by_dim=True)

        self.positional_encoding = PositionalEncoding(
            hidden_dim, dropout=dropout)
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_transformer_heads,
            dim_feedforward=hidden_dim, dropout=dropout)

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layers, num_transformer_layers)

        self.pooling = PoolingModule(hidden_dim, dropout)
        self.logits = nn.Linear(hidden_dim, output_classes)

        init_scaling(self.embed_init_scaling, self.embedding.embedding)
        print("Initialized model with parameters:\n", self.config())


    def config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "vocab_size": self.vocab_size,
            "num_transformer_heads": self.num_transformer_heads,
            "num_transformer_layers": self.num_transformer_layers,
            "embed_init_scaling": self.embed_init_scaling,
            "fix_embeddings": self.fix_embeddings,
            "dropout": self.dropout,
            "output_classes": self.output_classes
        }


    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -self.embed_init_scaling,
                         self.embed_init_scaling)


    def forward(self, input_tuple):
        mask = self.masking(input_tuple)
        emb_x = self.embedding(input_tuple)
        emb_x = self.positional_encoding(emb_x)
        output = self.transformer_encoder(emb_x, src_key_padding_mask=mask)
        output = self.pooling(output)
        output = self.logits(output)
        return output
