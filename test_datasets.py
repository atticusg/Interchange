import itertools
import pytest

from datasets import LogicalFormDataset

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
    dataset.create()
    assert dataset.X == inputs
    assert dataset.y == res