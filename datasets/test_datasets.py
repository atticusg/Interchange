import itertools
import pytest

from datasets.logical_form import LogicalFormDataset
from datasets.mqnli import MQNLIData

def bool_product(n):
    return list(itertools.product(*((False, True) for _ in range(n))))


@pytest.mark.parametrize("n, expected", [(1, [(False,), (True,)]),
                                         (2, [(False, False), (False, True),
                                              (True, False),(True, True)])])
def test_bool_product(n, expected):
    res = bool_product(n)
    assert res == expected


testdata = [
    ("~x", bool_product(1), [True, False]),
    ("x & y", bool_product(2), [False, False, False, True]),
    ("(x | y) & z", bool_product(3), [False, False, False, True, False, True, False, True])
]

@pytest.mark.parametrize("expr, inputs, res", testdata)
def test_logical_form_dataset1(expr, inputs, res):
    dataset = LogicalFormDataset(expr)
    assert dataset.X == inputs
    assert dataset.y == res


def test_mqnli_dataset():
    train_file = "mqnli_data/mini.train.txt"
    dev_file = "mqnli_data/mini.dev.txt"
    test_file = "mqnli_data/mini.test.txt"
    data = MQNLIData(train_file, dev_file, test_file, for_transformer=False)
    print("*** First piece of data: ", data.train[0])
    print("*** length: ", data.train[0][0].shape)
    print("Premise:", " ".join(data.id_to_word[w.item()] for w in data.train[0][0][:9]))
    print("Hypothesis:", " ".join(data.id_to_word[w.item()] for w in data.train[0][0][9:]))
    print("*******length of data", len(data.train))


