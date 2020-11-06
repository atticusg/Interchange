import torch
from torch import nn

from transformers import BertTokenizer
from transformers import BertModel


class PretrainedBertModule(nn.Module):
    def __init__(self, tokenizer_vocab_path: str="",
                 pretrained_bert_type: str="bert-base-uncased",
                 task: str="mqnli", output_classes: int=3,
                 device=None):
        super(PretrainedBertModule, self).__init__()
        if not tokenizer_vocab_path:
            raise ValueError("Must provide tokenizer vocabulary!")

        self.tokenizer_vocab_path = tokenizer_vocab_path
        self.tokenizer = BertTokenizer(tokenizer_vocab_path)

        self.task = task
        self.output_classes = output_classes
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device if device else torch.device("cpu")

        self.bert = BertModel.from_pretrained(pretrained_bert_type)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        hidden_dim = self.bert.config.hidden_size
        dropout = self.bert.config.hidden_dropout_prob

        self.dropout = nn.Dropout(dropout)
        self.logits = nn.Linear(hidden_dim, output_classes)


    def config(self):
        return {
            "tokenizer_vocab_path": self.tokenizer_vocab_path,
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
