import torch
import json
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import Dict, List, Callable, Optional
from intervention import LOC


subphrase_loss_weighting = [1. / 6] * 6 + [1. / 3] * 3 + [0.8] * 2 + [1.]

def get_collate_fxn(dataset, batch_first: bool=False) -> Optional[Callable]:
    if isinstance(dataset, MQNLIDataset):
        if "subphrase" in dataset.variant:
            return lambda batch: lstm_subphrase_collate(batch, batch_first=batch_first)
        else:
            return lambda batch: lstm_collate(batch, batch_first=batch_first)
    elif isinstance(dataset, MQNLIBertDataset):
        if dataset.variant == "subphrase":
            return bert_subphrase_collate
        else:
            return None

class MQNLIData:
    def __init__(self, train_file, dev_file, test_file, variant="lstm"):
        self.output_classes = 10 if "subphrase" in variant else 3
        self.word_to_id = {}
        self.id_to_word = {}
        self.id_to_word[0] = '[PAD]'
        self.word_to_id['[PAD]'] = 0
        self.variant = variant
        max_info = {'id': 1}

        if variant == "lstm" or variant == "lstm-subphrase":
            self.id_to_word[1] = '[SEP]'
            self.word_to_id['[SEP]'] = 1
            max_info['id'] = 2
        elif "transformer" in variant:
            self.id_to_word[1] = '[SEP]'
            self.word_to_id['[SEP]'] = 1
            self.id_to_word[2] = '[CLS]'
            self.word_to_id['[CLS]'] = 2
            max_info['id'] = 3
        else:
            raise ValueError(f"Invalid variant {variant}")

        print("--- Loading Dataset ---")
        self.train = MQNLIDataset(train_file, self.word_to_id, self.id_to_word,
                                  max_info, variant)
        train_ids = max_info["id"]
        print(f"--- loaded train set, {train_ids} unique tokens up to now")

        if variant == "lstm-subphrase": variant = "lstm"

        self.dev = MQNLIDataset(dev_file, self.word_to_id, self.id_to_word,
                                max_info, variant)
        dev_ids = max_info["id"]
        print(f"--- loaded dev set, {dev_ids} unique tokens up to now")

        self.test = MQNLIDataset(test_file, self.word_to_id, self.id_to_word,
                                 max_info, variant)
        test_ids = max_info["id"]
        print(f"--- loaded test set, {test_ids} unique tokens up to now")

        self.vocab_size = max_info['id']
        print(f"--- Got {self.vocab_size} unique words")

    def decode(self, t, return_str=False):
        if return_str:
            if isinstance(t, torch.Tensor) and len(t.shape) == 1:
                return " ".join(self.id_to_word[w.item()] for w in t)
            elif isinstance(t, torch.Tensor) and len(t.shape) == 2:
                return [" ".join(self.id_to_word[w]) for s in t for w in s]
            elif isinstance(t, list):
                return " ".join(self.id_to_word[w] for w in t)
            else:
                raise ValueError("incorrect input type")
        else:
            if isinstance(t, torch.Tensor) and len(t.shape) == 1:
                return [self.id_to_word[w.item()] for w in t]
            elif isinstance(t, torch.Tensor) and len(t.shape) == 2:
                return [[self.id_to_word[w] for w in s] for s in t]
            elif isinstance(t, list):
                return [self.id_to_word[w] for w in t]
            else:
                raise ValueError("incorrect input type")

    def save(self, f):
        torch.save(self, f)

    @staticmethod
    def load(f):
        return torch.load(f)

"""
    notevery | bad | singer | doesnot | badly | sings | every | good | song | [SEP] | ...
    0          1     2        3         4       5       6       7      8      9
[0] bad      | <P> | ...
[1] singer   | <P> | ...
[2] badly    | <P> | ...
[3] sings    | <P> | ...
[4] good     | <P> | ...
[5] song     | <P> | ...
[6] bad      | singer 
[7] badly    | sings
[8] good     | song 
[9] badly sings every good song
[10] doesnot badly sings every good song
"""

lstm_subphrase_loc_mappings = \
    [((0, 0), 1), ((0, 1), 9), ((0, 2), 11),
     ((1, 0), 2), ((1, 1), 9), ((1, 2), 12),
     ((2, 0), 4), ((2, 1), 9), ((2, 2), 14),
     ((3, 0), 5), ((3, 1), 9), ((3, 2), 15),
     ((4, 0), 7), ((4, 1), 9), ((4, 2), 17),
     ((5, 0), 8), ((5, 1), 9), ((5, 2), 18),
     ((6, LOC[:2]), LOC[1:3]), ((6, 2), 9), ((6, LOC[3:5]), LOC[11:13]),
     ((7, LOC[:2]), LOC[4:6]), ((7, 2), 9), ((7, LOC[3:5]), LOC[14:16]),
     ((8, LOC[:2]), LOC[7:9]), ((8, 2), 9), ((8, LOC[3:5]), LOC[17:19]),
     ((9, LOC[:5]), LOC[4:9]), ((9, 5), 9), ((9, LOC[6:11]), LOC[14:19]),
     ((10, LOC[:6]), LOC[3:9]), ((10, 6), 9), ((10, LOC[7:13]), LOC[13:19]),
     ((11, LOC[:]), LOC[:])]
lstm_subphrase_lengths = [3,3,3,3,3,3,5,5,5,11,13,19]


class MQNLIDataset(Dataset):
    def __init__(self, file_name, word_to_id, id_to_word, max_info,
                 variant="lstm"):
        print("--- Loading sentences from " + file_name)
        self.variant = variant
        if "subphrase" in variant:
            label_dict = {"neutral": 0, "entailment": 1, "contradiction": 2,
                          "independence": 3, "equivalence": 4, "entails": 5,
                          "reverse entails": 6, "contradiction2": 7,
                          "alternation": 8, "cover": 9}
        else:
            label_dict = {"neutral": 0, "entailment": 1, "contradiction": 2}
        raw_x = []
        raw_y = []

        curr_id = max_info['id']
        with open(file_name, 'r') as f:
            for line in f:
                example = json.loads(line.strip())
                sentence1 = example["sentence1"].split()
                sentence2 = example["sentence2"].split()

                if "subphrase" not in variant:
                    label = label_dict[example["gold_label"]]
                else:
                    label = [label_dict[l] for l in example["gold_label"]]

                assert len(sentence1) == len(sentence2)

                ids = []
                if "transformer" in variant: ids.append(2) # [CLS] token

                for word in sentence1:
                    if word not in word_to_id:
                        word_to_id[word] = curr_id
                        id_to_word[curr_id] = word
                        curr_id += 1
                    ids.append(word_to_id[word])

                ids.append(1) # [SEP] token

                for word in sentence2:
                    if word not in word_to_id:
                        word_to_id[word] = curr_id
                        id_to_word[curr_id] = word
                        curr_id += 1
                    ids.append(word_to_id[word])

                if "transformer" in variant: ids.append(1) # [SEP] token

                ids = np.asarray(ids)
                raw_x.append(ids)
                raw_y.append(label)

        self.raw_x = raw_x
        self.raw_y = raw_y
        self.num_examples = len(raw_x)
        max_info['id'] = curr_id
        print("--- Loaded {} sentences".format(self.num_examples))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, i):
        if self.variant == "lstm" or self.variant == "transformer":
            sample = (torch.tensor(self.raw_x[i], dtype=torch.long), self.raw_y[i])
        elif self.variant == "lstm-subphrase":
            sample = (self.generate_subphrase_inputs(i),
                      torch.tensor(subphrase_loss_weighting, dtype=torch.float),
                      torch.tensor(lstm_subphrase_lengths, dtype=torch.long),
                      torch.tensor(self.raw_y[i], dtype=torch.long))
            return sample
        else:
            raise ValueError(f"model variant not supported: {self.variant}")
        return sample

    def generate_subphrase_inputs(self, i: int) -> torch.Tensor:
        sentence = self.raw_x[i]
        res = torch.full((12, 19), 0, dtype=torch.long)
        for res_loc, raw_loc in lstm_subphrase_loc_mappings:
            res[res_loc] = torch.tensor(sentence[raw_loc], dtype=torch.long)
        return res


def lstm_collate(batch, batch_first=False):
    sorted_batch = sorted(batch, key=lambda pair: pair[0].shape[0], reverse=True)
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
    labels = torch.tensor([x[1] for x in sorted_batch])
    return (sequences_padded, labels)

# def lstm_collate(batch, batch_first=False):
#     print(type(batch))

def lstm_subphrase_collate(batch, batch_first=False):
    input_ids = torch.cat([x[0] for x in batch])
    loss_weighting = torch.cat([x[-3] for x in batch])
    lengths = torch.cat([x[-2] for x in batch])
    labels = torch.cat([x[-1] for x in batch])

    sorted_lengths, sort_idxs = torch.sort(lengths, descending=True)
    sorted_input_ids = input_ids[sort_idxs]
    sorted_loss_weighting = loss_weighting[sort_idxs]
    sorted_labels = labels[sort_idxs]

    if not batch_first: sorted_input_ids = sorted_input_ids.T

    return sorted_input_ids, sorted_loss_weighting, sorted_lengths, sorted_labels


def load_remapping(file_name: str) -> Dict:
    remapping = {}
    with open(file_name, "r") as f:
        for line in f:
            w1, w2 = line.strip().split(",")
            remapping[w1] = w2
    return remapping


class MQNLIBertData(MQNLIData):
    def __init__(self, train_file: str, dev_file: str, test_file: str,
                 vocab_remapping_file: str, tokenizer_type="bert-base-uncased",
                 variant="basic"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_type)
        self.tokenizer_type = tokenizer_type
        self.variant = variant

        if variant == "basic":
            special_tokens = ["emptystring"]
        elif variant == "subphrase":
            special_tokens = ["emptystring"]
        else:
            raise NotImplementedError(f"Does not support data variant {variant}")

        self.special_tokens = special_tokens
        self.tokenizer.add_tokens(special_tokens)

        self.vocab_remapping = load_remapping(vocab_remapping_file)

        self.word_to_id, self.id_to_word = self.get_id_word_dicts()

        # print(f"Processed vocabulary, got {len(self.word_to_id)} unique words")

        print("--- Loading Dataset ---")
        self.train = MQNLIBertDataset(train_file, self.vocab_remapping,
                                      self.word_to_id, self.tokenizer,
                                      variant=variant)
        print(f"--- finished loading {len(self.train)} examples from train set")

        self.dev = MQNLIBertDataset(dev_file, self.vocab_remapping,
                                    self.word_to_id, self.tokenizer,
                                    variant="basic")
        print(f"--- finished loading {len(self.dev)} examples from dev set")

        self.test = MQNLIBertDataset(test_file, self.vocab_remapping,
                                     self.word_to_id, self.tokenizer,
                                     variant="basic")
        print(f"--- finished loading {len(self.test)} examples from test set")

    def get_id_word_dicts(self):
        tokens = list(self.vocab_remapping.values()) + self.special_tokens + \
                 ["[PAD]", "[CLS]", "[SEP]", "no", "some", "every", "does", "not"]
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        max_id = max(ids)
        word_to_id = {tok: i for tok, i in zip(tokens, ids)}
        id_to_word = {i: tok for i, tok in zip(ids, tokens)}

        max_id += 1
        for keyword in ["notevery", "doesnot"]:
            if keyword not in word_to_id:
                word_to_id[keyword] = max_id
                id_to_word[max_id] = keyword
                max_id += 1

        return word_to_id, id_to_word

"""
   [CLS] | not | every | bad | singer | does | not | badly | sings | <e> | every | good | song ]
    0     1      2       3      4       5      6     7       8       9     10      11     12     13 14 15 16
[0]      | <P> | <P>   | bad | <P>    | <P>  | ...
[1]      | <P> | ...   | <P> | singer | <P>  | ...
[2]      | <P> | ...                     ... | <P> | badly | <P>   | ...
[3]      | <P> | ...                           ... | <P>   | sings | <P> | ...
[4]      | <P> | ...                                                         <P> | good | <P>  |
[5]      | <P> | ...                                                         ... | <P>  | song |
[6]      | <P> | <P>   | bad | singer | <P>  | ...
[7]      | <P> | ...                     ... | <P> | badly | sings | <P> | ...
[8]      | <P> | ...                                                 ... |   <P> | good | song |
[9]      | <P> | ...                     ... | <P> | badly | sings | not | every | good | song |
[10]     | <P> | ...         | <P>    | does | not | badly | sings | not | every | good | song |
[11] <the whole sentence>
"""
bert_subphrase_loc_mappings = [(0, 3), (0, 16), (1, 4), (1, 17), (2, 7), (2, 20),
                               (3, 8), (3, 21), (4, 11), (4, 24), (5, 12), (5, 25),
                               (6, LOC[3:5]), (6, LOC[16:18]),
                               (7, LOC[7:9]), (7, LOC[20:22]),
                               (8, LOC[11:13]), (8, LOC[24:26]),
                               (9, LOC[7:13]), (9, LOC[20:26]),
                               (10, LOC[5:13]), (10, LOC[18:26]), (11, LOC[:])]

class MQNLIBertDataset(Dataset):
    def __init__(self, file_name: str, vocab_remapping: Dict[str,str],
                 word_to_id: Dict[str, int], tokenizer: BertTokenizer,
                 variant="basic"):
        print("--- Loading sentences from " + file_name)
        self.vocab_remapping = vocab_remapping
        self.tokenizer = tokenizer
        self.shifted_idxs = [1, 2, 3, 5, 6, 7, 9, 10, 11]
        self.word_to_id = word_to_id
        self.variant = variant

        if variant == "basic":
            label_dict = {"neutral": 0, "entailment": 1, "contradiction": 2}
        elif variant == "subphrase":
            label_dict = {"neutral": 0, "entailment": 1, "contradiction": 2,
                          "independence": 3, "equivalence": 4, "entails": 5,
                          "reverse entails": 6, "contradiction2": 7,
                          "alternation": 8, "cover": 9}
            self.label_dict = label_dict
        else:
            raise NotImplementedError(f"Does not support data variant {variant}")

        self.raw_bert_x = []
        self.raw_orig_x = []
        self.attention_masks = []
        self.raw_y = []

        with open(file_name, 'r') as f:
            count = 0
            for line in f:
                example = json.loads(line.strip())

                if variant == "basic":
                    label = label_dict[example["gold_label"]]
                else:
                    label = [label_dict[l] for l in example["gold_label"]]

                # attention mask useful for short outputs
                p_toks, unshifted_p_toks, p_attn_mask = \
                    self.remap_and_shift(example, is_p=True)
                h_toks, unshifted_h_toks, h_attn_mask = \
                    self.remap_and_shift(example, is_p=False)

                # convert string form tokens into BERT tokens
                bert_toks = self.tokenize(p_toks, h_toks)
                orig_toks = unshifted_p_toks + unshifted_h_toks
                attention_mask = [1.] + p_attn_mask + [1.] + h_attn_mask + [1.]
                # assert all(t == tt for t, tt in zip(bert_toks, p["input_ids"]))
                self.raw_bert_x.append(bert_toks)
                self.raw_orig_x.append(orig_toks)
                self.attention_masks.append(attention_mask)
                self.raw_y.append(label)
                count += 1

        self.num_examples = len(self.raw_bert_x)

    def remap_and_shift(self, example: Dict, is_p: bool=True) -> (List[str], List[str], List[int]):
        """ map words to other words that won't be split up by BERT, and shift
        them into new positions according to the following schema:
             0          1     2        3         4       5       6       7      8
           [ notevery | bad | singer | doesnot | badly | sings | every | good | song ]
             /   \                       |   \                     |   \
            /     \                      |    \                    |    \
        [ not | every | bad | singer | does | not | badly | sings | <e> | every | good | song ]
          0     1       2     3        4      5     6       7       8     9       10     11
        """
        s = example["sentence1"].split() if is_p else example["sentence2"].split()
        remapped_words = [self.vocab_remapping.get(w.lower(), w) for w in s]

        if self.variant in ["basic", "subphrase"]:
            toks = ["emptystring"] * 12
            for old_idx, new_idx in enumerate(self.shifted_idxs):
                curr_tok = remapped_words[old_idx]
                if old_idx in [0, 6] and curr_tok == "notevery":
                    toks[new_idx-1] = "not"
                    toks[new_idx] = "every"
                elif old_idx == 3 and curr_tok == "doesnot":
                    toks[new_idx-1] = "does"
                    toks[new_idx] = "not"
                else:
                    toks[new_idx] = curr_tok

            unshifted_toks = [self.word_to_id[w] for w in remapped_words]
            attention_mask = [1. for _ in toks]
            return toks, unshifted_toks, attention_mask
        else:
            raise NotImplementedError(f"Does not support tokenziation variant"
                                      f"{self.variant}")

    def tokenize(self, p_toks: List[str], h_toks: List[str]):
        # do not use BertTokenizer's tokenize because it is very slow
        toks = ["[CLS]"] + p_toks + ["[SEP]"] + h_toks + ["[SEP]"]
        return [self.word_to_id[t] for t in toks]

    def __len__(self):
        return self.num_examples

    def generate_subphrase_inputs(self, i: int) -> torch.Tensor:
        """ Given a full sentence, generate examples containing subphrases

        :param i: idx
        :return: torch.Tensor
        """
        sentence = self.raw_bert_x[i]
        res = torch.full((12, 27), self.word_to_id["[PAD]"], dtype=torch.long)
        res[:,0] = self.word_to_id["[CLS]"]
        res[:,13] = self.word_to_id["[SEP]"]
        res[:,26] = self.word_to_id["[SEP]"]

        for loc in bert_subphrase_loc_mappings:
            res[loc] = torch.tensor(sentence[loc[1]], dtype=torch.long)
        return res

    def generate_subphrase_attn_masks(self, i):
        res = torch.full((12, 27), 0.0, dtype=torch.float)
        res[:, 0] = 1.0
        res[:, 13] = 1.0
        res[:, 26] = 1.0

        for loc in bert_subphrase_loc_mappings:
            res[loc] = 1.0
        return res

    def __getitem__(self, i):
        """Returns tuple: (input_ids, token_type_ids, attention_masks,
            raw_original_x, label) """
        if self.variant == "basic":
            sample = (torch.tensor(self.raw_bert_x[i], dtype=torch.long),
                      torch.tensor([0]*14 + [1]*13, dtype=torch.long),
                      torch.tensor(self.attention_masks[i], dtype=torch.float),
                      torch.tensor(self.raw_orig_x[i], dtype=torch.long),
                      self.raw_y[i])
            return sample
        elif "subphrase" in self.variant:
            sample = (self.generate_subphrase_inputs(i),
                      torch.tensor([[0]*14 + [1]*13]*12, dtype=torch.long),
                      self.generate_subphrase_attn_masks(i),
                      torch.tensor(subphrase_loss_weighting, dtype=torch.float),
                      torch.tensor(self.raw_orig_x[i], dtype=torch.long),
                      torch.tensor(self.raw_y[i], dtype=torch.long))
            return sample
        else:
            raise NotImplementedError(f"Does not support tokenziation variant"
                                      f"{self.variant}")


def bert_subphrase_collate(batch):
    input_ids = torch.cat([x[0] for x in batch])
    token_type_ids = torch.cat([x[1] for x in batch])
    attention_mask = torch.cat([x[2] for x in batch])
    loss_weighting = torch.cat([x[-3] for x in batch])
    raw_original_x = torch.stack([x[-2] for x in batch])
    labels = torch.cat([x[-1] for x in batch])
    return (input_ids, token_type_ids, attention_mask, loss_weighting, raw_original_x, labels)

def bert_trainer_collate(batch):
    input_ids = torch.stack([x[0] for x in batch])
    token_type_ids = torch.stack([x[1] for x in batch])
    attention_mask = torch.stack([x[2] for x in batch])
    labels = torch.tensor([x[-1] for x in batch])
    return {"input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "labels": labels}


if __name__ == "__main__":
    train_file = "../mqnli_data/mqnli.train.txt"
    dev_file = "../mqnli_data/mqnli.dev.txt"
    test_file = "../mqnli_data/mqnli.test.txt"
    vocab_remapping_file = "../mqnli_data/bert-remapping.txt"
    pickle_file = "../mqnli_data/mqnli_bert.pt"
    tokenizer_vocab_file = "../mqnli_data/bert-vocab.pt"

    data = MQNLIBertData(train_file, dev_file, test_file, vocab_remapping_file)

    # data.save(pickle_file)
    # data = torch.load(pickle_file)
    tokenizer = data.tokenizer
    print("num vocab in this tokenizer:", len(tokenizer))
    # tokenizer.save_vocabulary(tokenizer_vocab_file)
    new_tokenizer = BertTokenizer(tokenizer_vocab_file)
    print("num vocab in loaded tokenizer", len(new_tokenizer))

    # dataloader = DataLoader(data.train, batch_size=3, shuffle=False, collate_fn=bert_trainer_collate)

    # for batch in dataloader:
    #     print(batch)
    #     break

    # data2 = MQNLIData(train_file, dev_file, test_file, for_transformer=False)
    #
    # for orig_word in data2.word_to_id.keys():
    #     remapped_word = data1.vocab_remapping.get(orig_word.lower(), orig_word.lower())
    #     if remapped_word not in data1.word_to_id:
    #         print(f"Did not find word (orig) {orig_word} / (new) {remapped_word} in new dataset")


def get_data_variant(data_object):
    if isinstance(data_object, MQNLIBertData):
        return "bert-" + getattr(data_object, "variant", "basic")
    elif isinstance(data_object, MQNLIData):
        return "lstm"

