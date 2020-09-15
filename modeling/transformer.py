import torch
import torch.nn as nn
import math

# refer to `BertPooler` class in HuggingFace BERT implementation


class MaskingModule(nn.Module):
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
    def __init__(self, hidden_dim=64, num_transformer_heads=4,
                 num_transformer_layers=6, vocab_size=10000,
                 embed_init_scaling=0.1, dropout=0.1, device=None):
        super(TransformerModule, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_transformer_heads = num_transformer_heads
        self.num_transformer_layers = num_transformer_layers
        self.embed_init_scaling = embed_init_scaling
        self.dropout = dropout
        self.vocab_size = vocab_size
        self.device = device if device else torch.device("cpu")

        self.masking = MaskingModule()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                            embedding_dim=hidden_dim)
        self.positional_encoding = PositionalEncoding(
            hidden_dim, dropout=dropout)
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_transformer_heads,
            dim_feedforward=hidden_dim, dropout=dropout)

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layers, num_transformer_layers)

        self.pooling = PoolingModule(hidden_dim, dropout)
        self.logits = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()
        print(self.config())

    def config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "vocab_size": self.vocab_size,
            "num_transformer_heads": self.num_transformer_heads,
            "num_transformer_layers": self.num_transformer_layers,
            "embed_init_scaling": self.embed_init_scaling,
            "dropout": self.dropout
        }


    def init_weights(self):
        nn.init.uniform_(self.embedding.weight, -self.embed_init_scaling,
                         self.embed_init_scaling)

    def forward(self, input_tuple):
        x_batch, x_lengths = input_tuple[0], input_tuple[2]
        mask = self.masking(input_tuple)
        emb_x = self.embedding(x_batch) * math.sqrt(self.hidden_dim)
        emb_x = self.positional_encoding(emb_x)
        output = self.transformer_encoder(emb_x, src_key_padding_mask=mask)
        output = self.pooling(output)
        logits = self.logits(output)
        scores = self.sigmoid(logits).squeeze()
        preds = torch.round(scores).type(torch.int)
        return scores, preds


