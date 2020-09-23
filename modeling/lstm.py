import torchimport torch.nn as nnimport torch.nn.functional as Fimport torch.optim as optimfrom modeling.utils import EmbeddingModule, MeanModule, init_scaling# TODO: update LSTM to output logitsclass LSTMWrapperModule(nn.Module):    def __init__(self):        super(LSTMWrapperModule, self).__init__()    def forward(self):        passclass PremiseSliceModule(nn.Module):    def __init__(self, length_dim=0):        if not 0 <= length_dim <= 1:            raise ValueError("Invalid length dim!")        super(PremiseSliceModule, self).__init__()        self.length_dim = length_dim    def forward(self, emb):        if self.length_dim == 0:            return emb[:9,:,:]        elif self.length_dim == 1:            return emb[:,:9,:]class HypothesisSliceModule(nn.Module):    def __init__(self, length_dim=0):        if not 0 <= length_dim <= 1:            raise ValueError("Invalid length dim!")        super(HypothesisSliceModule, self).__init__()        self.length_dim = length_dim    def forward(self, emb):        if self.length_dim == 0:            return emb[9:,:,:]        elif self.length_dim == 1:            return emb[:,9:,:]class ConcatFinalStateModule(nn.Module):    def __init__(self, length_dim=0, bidirectional=True):        if length_dim == 1:            raise NotImplementedError("Currently does not support length_dim=1")        if length_dim != 0:            raise ValueError("Invalid length dim!")        super(ConcatFinalStateModule, self).__init__()        self.length_dim = length_dim        self.bidirectional = bidirectional    def forward(self, premise_hidden, hypothesis_hidden):        if self.bidirectional:            hidden_dim = premise_hidden.shape[-1] // 2            premise_forward_out = premise_hidden[-1,:,:hidden_dim]            premise_backward_out = premise_hidden[0,:,hidden_dim:]            hypothesis_forward_out = hypothesis_hidden[-1, :, :hidden_dim]            hypothesis_backward_out = hypothesis_hidden[0, :, hidden_dim:]            return torch.cat((premise_forward_out, premise_backward_out,                              hypothesis_forward_out, hypothesis_backward_out), dim=1)        else:            premise_out = premise_hidden[-1,:,:]            hypothesis_out = hypothesis_hidden[-1,:,:]            return torch.cat((premise_out, hypothesis_out), dim=1)class LSTMModule(nn.Module):    def __init__(self, task="sentiment", output_classes=2, vocab_size=10000,                 embed_dim=40, lstm_hidden_dim=20, activation_type="relu",                 bidirectional=True, num_lstm_layers=1, dropout=0.1,                 embed_init_scaling=0.1, fix_embeddings=False,                 batch_first=False, device=None):        super(LSTMModule, self).__init__()        self.task = task        self.output_classes = output_classes        self.vocab_size = vocab_size        self.embed_dim = embed_dim        self.lstm_hidden_dim = lstm_hidden_dim        self.bidirectional = bidirectional        self.num_lstm_layers = num_lstm_layers        self.dropout = dropout        self.embed_init_scaling = embed_init_scaling        self.fix_embeddings = fix_embeddings        self.activation_type = activation_type        self.batch_first = batch_first        hidden_dim = lstm_hidden_dim * 2 if bidirectional else \            lstm_hidden_dim        if self.task == "mqnli":            hidden_dim *= 2        self.device = device if device else torch.device("cpu")        self.embedding = EmbeddingModule(num_embeddings=vocab_size,                                         embedding_dim=embed_dim,                                         fix_weights=fix_embeddings)        if self.task == "mqnli":            self.premise_emb = PremiseSliceModule(length_dim=(1 if batch_first else 0))            self.hypothesis_emb = HypothesisSliceModule(length_dim=(1 if batch_first else 0))            self.concat_final_state = ConcatFinalStateModule(length_dim=(1 if batch_first else 0))        self.lstm_layers = nn.ModuleList()        self.lstm_layers.append(            nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=batch_first,                    bidirectional=bidirectional))        for _ in range(1, num_lstm_layers):            self.lstm_layers.append(                nn.LSTM(lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim,                        lstm_hidden_dim, batch_first=batch_first,                        bidirectional=bidirectional))        if self.task == "sentiment":            self.mean = MeanModule(length_dim=(1 if batch_first else 0))        self.feed_forward1 = nn.Linear(hidden_dim, hidden_dim)        self.feed_forward2 = nn.Linear(hidden_dim, hidden_dim)        self.logits = nn.Linear(hidden_dim, output_classes)        if self.activation_type == "relu":            self.activation1 = nn.ReLU()            self.activation2 = nn.ReLU()        elif self.activation_type == "tanh":            self.activation1 = nn.Tanh()            self.activation2 = nn.Tanh()        else:            raise ValueError("CBOWModule does not support this activation type")        self.lstm_dropout = nn.Dropout(dropout)        self.dropout0 = nn.Dropout(dropout)        self.dropout1 = nn.Dropout(dropout)        self.dropout2 = nn.Dropout(dropout)        init_scaling(self.embed_init_scaling, self.embedding.embedding)    def config(self):        return {            "task": self.task,            "output_classes": self.output_classes,            "vocab_size": self.vocab_size,            "embed_dim": self.embed_dim,            "lstm_hidden_dim": self.lstm_hidden_dim,            "bidirectional": self.bidirectional,            "num_lstm_layers": self.num_lstm_layers,            "dropout": self.dropout,            "embed_init_scaling": self.embed_init_scaling,            "fix_embeddings": self.fix_embeddings,            'batch_first': self.batch_first        }    def forward(self, input_tuple, verbose=False):        emb_x = self.embedding(input_tuple)        if self.task == "mqnli":            premise = self.premise_emb(emb_x)            premise_h = self._run_lstm(premise)            hypothesis = self.hypothesis_emb(emb_x)            hypothesis_h = self._run_lstm(hypothesis)            repr = self.concat_final_state(premise_h, hypothesis_h)            repr = self.dropout0(repr)        elif self.task == "sentiment":            x_batch, x_lengths = input_tuple[0], input_tuple[2]            hidden = self._run_lstm(emb_x, x_lengths)            repr = self.mean(hidden, input_tuple)        else:            raise ValueError("Task type \'%s\' is undefined" % self.task)        output = self.feed_forward1(repr)        output = self.activation1(output)        output = self.dropout1(output)        output = self.feed_forward2(output)        output = self.activation2(output)        output = self.dropout2(output)        output = self.logits(output)        return output    def _run_lstm(self, x, x_lengths=None):        if self.task == "sentiment":            x = nn.utils.rnn.pack_padded_sequence(x, x_lengths,                                              batch_first=self.batch_first)        for lstm_layer in self.lstm_layers:            x, _ = lstm_layer(x)            x = self.lstm_dropout(x)        if self.task == "sentiment":            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=self.batch_first)        return xclass SelfAttentionModule(nn.Module):    def __init__(self, vocab_size=10000, attn_query_dim=5, hidden_dim=40,                 attn_value_dim=5):        super(SelfAttentionModule, self).__init__()        self.query_embed = nn.Embedding(vocab_size, attn_query_dim, padding_idx=0)        self.key_lin_trans = nn.Linear(hidden_dim, attn_query_dim)        self.value_lin_trans = nn.Linear(hidden_dim, attn_value_dim, bias=False)    def forward(self, x, x_batch):        """ Self attention module        :param keys: Hidden vectors        :param x_batch: Raw tokens in idx form        :return:        """        queries = self.query_embed(x_batch)        keys = self.key_lin_trans(x)        a = torch.matmul(queries, torch.transpose(keys, 1, 2))        s = torch.sum(a, 1)        # print("queries:",queries.shape,"\nkeys:", keys.shape, "\na:", a.shape,        #       "\ns:", s.shape)        padding_mask = torch.ne(x_batch, 0).type(torch.float)        exp = torch.mul(torch.exp(s), padding_mask)        attn = torch.nn.functional.normalize(exp, p=1, dim=1).unsqueeze(2)        values = self.value_lin_trans(x)        # print("attn:", attn.shape, "\nvalues:", values.shape)        return torch.sum(attn * values, 1)class LSTMSelfAttnModule(LSTMModule):    def __init__(self, embed_dim=40, lstm_hidden_dim=20, vocab_size=10000,                 bidirectional=True, num_lstm_layers=1, attn_query_dim=5,                 attn_value_dim=5, device=None):        super(LSTMSelfAttnModule, self).__init__(            embed_dim=embed_dim, lstm_hidden_dim=lstm_hidden_dim,            vocab_size=vocab_size, bidirectional=bidirectional,            num_lstm_layers=num_lstm_layers, device=device)        self.attn_query_dim = attn_query_dim        self.attn_value_dim = attn_value_dim        self.logits = nn.Linear(self.attn_value_dim, 1)        self.self_attn = SelfAttentionModule(vocab_size, attn_query_dim,            self.layer_hidden_dim, attn_value_dim)    def config(self):        return {            "embed_dim": self.embed_dim,            "lstm_hidden_dim": self.lstm_hidden_dim,            "vocab_size": self.vocab_size,            "bidirectional": self.bidirectional,            "num_lstm_layers": self.num_lstm_layers,            "attn_query_dim": self.attn_query_dim,            "attn_value_dim": self.attn_value_dim        }    def forward(self, input_tuple, verbose=False):        x_batch, x_lengths = input_tuple[0], input_tuple[2]        emb_x = self.embedding(x_batch)        hidden = self._run_lstm(emb_x, x_lengths)        h = self.self_attn(hidden, x_batch)        x = self.logits(h)        x = torch.sigmoid(x).squeeze()        pred = torch.round(x).type(torch.int)        return x, pred# TODO: computation graph edition for modeling# TODO: transformer