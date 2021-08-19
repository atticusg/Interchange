import torch
from torch import nn

from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertConfig


class RawBertModule(nn.Module):
    def __init__(
            self,
            tokenizer_vocab_path: str="",
            num_hidden_layers: int = 6,
            hidden_size: int = 768,
            num_attention_heads: int = 12,
            output_classes: int=3,
            task: str="mqnli"
        ):
        super(RawBertModule, self).__init__()
        if not tokenizer_vocab_path:
            raise ValueError("Must provide tokenizer vocabulary!")

        self.tokenizer_vocab_path = tokenizer_vocab_path
        self.tokenizer = BertTokenizer(tokenizer_vocab_path)

        self.task = task
        self.output_classes = output_classes

        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        bert_config = BertConfig(
            vocab_size=len(self.tokenizer),
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads
        )
        self.bert = BertModel(bert_config)

        hidden_dim = hidden_size
        dropout = bert_config.hidden_dropout_prob

        self.dropout = nn.Dropout(dropout)
        self.logits = nn.Linear(hidden_dim, output_classes)

    def config(self):
        return {
            "tokenizer_vocab_path": self.tokenizer_vocab_path,
            "num_hidden_layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "task": self.task,
            "output_classes":  self.output_classes,
        }

    # based on BertForNextSentencePrediction and BertForSequenceClassification
    def forward(self, input_tuple):
        input_ids = input_tuple[0]
        token_type_ids = input_tuple[1]
        attention_mask = input_tuple[2]

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.logits(pooled_output)
        return logits

    @property
    def device(self):
        # all of the parameters in the module are guaranteed to be on the same
        # device, just get the device of one of them
        return next(self.parameters()).device
