import boolean
import itertools

import pytest

from torch.utils.data import Dataset


def remove_duplicates(l):
    return list(dict.fromkeys(l))

class LogicalFormDataset(Dataset):

    POS_LABEL = 1
    NEG_LABEL = 0

    def __init__(self, expr, embed_dim=None):
        """
        """
        super(LogicalFormDataset, self).__init__()
        self.embed_dim = embed_dim
        self.algebra = boolean.BooleanAlgebra()
        self.expr = self.algebra.parse(expr)

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

    def create(self):
        self.X, self.y = self.truth_table()
        return self.X, self.y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

