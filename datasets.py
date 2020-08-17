import boolean
import itertools

import pytest

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

