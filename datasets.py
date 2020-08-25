import boolean
import itertools
import torch
import numpy as np
from torch.utils.data import Dataset


def remove_duplicates(l):
    return list(dict.fromkeys(l))


class LogicalFormDataset(Dataset):
    def __init__(self, expr):
        """Construct a dataset given an arbitrary boolean expression with variables

        :param expr: str, a boolean expression in string form,
            e.g. "~(var1 | var2) & var3" where var1 var2 and var3 stand for
            variables. Use `~ | &` for NOT, OR, AND.
        """
        super(LogicalFormDataset, self).__init__()
        self.algebra = boolean.BooleanAlgebra()
        self.expr = self.algebra.parse(expr)
        self.X, self.y = self.truth_table()

    def truth_table(self):
        symbols = remove_duplicates(self.expr.get_symbols())
        t, f, _, _, _, _ = self.algebra.definition()
        X, y = [], []
        for vals in itertools.product(*((f, t) for _ in symbols)):
            e = self.expr.subs({s: v for s, v in zip(symbols, vals)})
            res = e.simplify()
            res = (res == t)
            inputs = tuple(v == t for v in vals)
            X.append(inputs)
            y.append(res)
        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SentimentData:
    def __init__(self, train_file, dev_file, test_file, ngram=0):
        self.word_to_id = {}
        self.id_to_word = {}
        self.id_to_word[0] = '<PADDING>'
        max_info = {
            'id': 1,
            'sentence_len': 0
        }
        if ngram > 0:
            print("--- USING NGRAM MODEL ---")
        print("--- Loading Dataset ---")
        self.train = SentimentDataset(train_file, self.word_to_id, self.id_to_word, max_info, ngram)
        print("--- finished loading train set")
        self.dev = SentimentDataset(dev_file, self.word_to_id, self.id_to_word, max_info, ngram)
        print("--- finished loading dev set")
        self.test = SentimentDataset(test_file, self.word_to_id, self.id_to_word, max_info, ngram)
        print("--- finished loading test set")
        self.vocab_size, self.max_sentence_len = max_info['id'], max_info['sentence_len']
        print("--- found {} unique words, max sentence len is {}".format(self.vocab_size, self.max_sentence_len))

class SentimentDataset(Dataset):
    def __init__(self, file_name, word_to_id, id_to_word, max_info, ngram):
        use_ngram = ngram > 0
        print("--- Loading sentences from " + file_name)
        raw_x = []
        raw_y = []
        curr_id, max_sentence_len = max_info['id'], max_info['sentence_len']
        with open(file_name, 'r') as f:
            for line in f:
                pair = line.split('\t')
                if use_ngram:
                    string = pair[0]
                    strlen = len(string)
                    sentence = [string[i:i+ngram] for i in range(strlen-ngram+1)]
                else:
                    sentence = pair[0].split()
                l = len(sentence)
                if l > max_sentence_len:
                    max_sentence_len = l
                ids = []
                for word in sentence:
                    if word not in word_to_id:
                        word_to_id[word] = curr_id
                        id_to_word[curr_id] = word
                        curr_id += 1
                    ids.append(word_to_id[word])
                ids = np.asarray(ids)
                raw_x.append(ids)
                label = int(pair[1].rstrip())
                raw_y.append(label)
        self.raw_x = raw_x
        self.raw_y = raw_y
        self.num_examples = len(raw_x)
        max_info['id'], max_info['sentence_len'] = curr_id, max_sentence_len
        print("--- Loaded {} sentences".format(self.num_examples))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, i):
        sample = (torch.LongTensor(self.raw_x[i]), self.raw_y[i])
        return sample

def my_collate(batch):
    sorted_batch = sorted(batch, key=lambda pair: pair[0].shape[0], reverse=True)
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    lengths = torch.tensor([len(x) for x in sequences])
    labels = torch.tensor([x[1] for x in sorted_batch])
    return (sequences_padded, labels, lengths)