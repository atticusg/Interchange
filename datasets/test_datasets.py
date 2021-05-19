import itertools
import pytest

from torch.utils.data import DataLoader
from datasets.logical_form import LogicalFormDataset
from datasets.mqnli import MQNLIData, MQNLIBertData, bert_subphrase_collate
from datasets.mqnli import lstm_subphrase_collate
from datasets.utils import write_pickle, read_pickle

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
    train_file = "../data/mqnli/raw/easy_mini/train-mini.txt"
    dev_file = "../data/mqnli/raw/easy_mini/dev.txt"
    test_file = "../data/mqnli/raw/easy_mini/test.txt"
    data = MQNLIData(train_file, dev_file, test_file)
    print("*** First piece of data: ", data.train[0])
    print("*** length: ", data.train[0][0].shape)
    print("Premise:", " ".join(data.id_to_word[w.item()] for w in data.train[0][0][:9]))
    print("Hypothesis:", " ".join(data.id_to_word[w.item()] for w in data.train[0][0][9:]))
    print("*******length of data", len(data.train))


def test_pickle():
    train_file = "../data/mqnli/raw/easy_mini/train-mini.txt"
    dev_file = "../data/mqnli/raw/easy_mini/dev.txt"
    test_file = "../data/mqnli/raw/easy_mini/test.txt"
    data = MQNLIData(train_file, dev_file, test_file)

    pickle_file = "../mqnli_data/mini.pt"
    write_pickle(data, pickle_file)

    data2 = read_pickle(pickle_file)
    assert isinstance(data2, MQNLIData)

    assert len(data.train) == len(data2.train)


def test_mqnli_hard():
    train_file = "../data/mqnli/raw/hard/train-mini.txt"
    dev_file = "../data/mqnli/raw/hard/dev.txt"
    test_file = "../data/mqnli/raw/hard/test.txt"
    vocab_remapping = "../data/tokenization/bert-remapping.txt"
    data = MQNLIBertData(train_file, dev_file, test_file, vocab_remapping, variant="subphrase")
    # x = data.train.generate_subphrase_inputs(1)
    # for ex in x:
    #     print(data.decode(ex, return_str=True))
    # attn_mask = data.train.generate_subphrase_attn_masks(1)
    # print(attn_mask)

    dataloader = DataLoader(data.train, batch_size=3, collate_fn=bert_subphrase_collate)
    for input_tuple in dataloader:
        print(input_tuple)
        print("input_ids.shape", input_tuple[0].shape)
        print("token_type_ids.shape", input_tuple[1].shape)
        print("attention_mask.shape", input_tuple[2].shape)
        print("labels.shape", input_tuple[-1].shape)
        input_ids = input_tuple[0]
        for ex in input_ids:
            print(data.decode(ex, return_str=True))
        break

def test_mqnli_lstm_subsequence():
    train_file = "../data/mqnli/raw/hard/train-mini.txt"
    dev_file = "../data/mqnli/raw/hard/dev.txt"
    test_file = "../data/mqnli/raw/hard/test.txt"
    data = MQNLIData(train_file, dev_file, test_file, variant="lstm-subphrase")

    dataloader = DataLoader(data.train, batch_size=4, collate_fn=lstm_subphrase_collate)
    for input_tuple in dataloader:
        print(input_tuple)
        break
