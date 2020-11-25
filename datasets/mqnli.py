import torch
import json
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer
from typing import Dict, List
from intervention import LOC


class MQNLIData:
    def __init__(self, train_file, dev_file, test_file, use_separator=False,
                 for_transformer=False, store_text=False):
        self.output_classes = 3
        self.word_to_id = {}
        self.id_to_word = {}
        self.id_to_word[0] = '[PAD]'
        self.word_to_id['[PAD]'] = 0
        max_info = {
            'id': 1,
            'sentence_len': 0
        }

        if for_transformer:
            self.id_to_word[1] = '[CLS]'
            self.word_to_id['[CLS]'] = 1
            self.id_to_word[2] = '[SEP]'
            self.word_to_id['[SEP]'] = 2
            max_info['id'] = 3
        else:
            if use_separator:
                self.id_to_word[1] = '[SEP]'
                self.word_to_id['[SEP]'] = 1
                max_info['id'] = 2

        print("--- Loading Dataset ---")
        self.train = MQNLIDataset(train_file, self.word_to_id, self.id_to_word,
                                  max_info, for_transformer, store_text)
        train_ids = max_info["id"]
        print(f"--- loaded train set, {train_ids} unique tokens up to now")

        self.dev = MQNLIDataset(dev_file, self.word_to_id, self.id_to_word,
                                max_info, for_transformer, store_text)
        dev_ids = max_info["id"]
        print(f"--- loaded dev set, {dev_ids} unique tokens up to now")

        self.test = MQNLIDataset(test_file, self.word_to_id, self.id_to_word,
                                 max_info, for_transformer, store_text)
        test_ids = max_info["id"]
        print(f"--- loaded test set, {test_ids} unique tokens up to now")

        self.vocab_size, self.max_sentence_len = max_info['id'], max_info[
            'sentence_len']
        print(f"--- Got {self.vocab_size} unique words, "
              f"max sentence len {self.max_sentence_len}")

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


class MQNLIDataset(Dataset):
    def __init__(self, file_name, word_to_id, id_to_word, max_info,
                 for_transformer=False, store_text=False):
        print("--- Loading sentences from " + file_name)
        label_dict = {"neutral": 0, "entailment": 1, "contradiction": 2}
        raw_x = []
        raw_y = []
        if store_text:
            self.example_text = []
        curr_id, max_sentence_len = max_info['id'], max_info['sentence_len']
        with open(file_name, 'r') as f:
            for line in f:
                if store_text:
                    self.example_text.append(line.strip())
                example = json.loads(line.strip())
                sentence1 = example["sentence1"].split()
                sentence2 = example["sentence2"].split()
                label = label_dict[example["gold_label"]]

                if len(sentence1) != len(sentence2):
                    raise RuntimeError("Premise and hypothesis have different lengths!\n"
                                       "P=\'%s\'\nH=\'%s\'" % sentence1, sentence2)

                ids = []
                if for_transformer:
                    ids.append(0)
                for word in sentence1:
                    if word not in word_to_id:
                        word_to_id[word] = curr_id
                        id_to_word[curr_id] = word
                        curr_id += 1
                    ids.append(word_to_id[word])
                if for_transformer:
                    ids.append(1)
                for word in sentence2:
                    if word not in word_to_id:
                        word_to_id[word] = curr_id
                        id_to_word[curr_id] = word
                        curr_id += 1
                    ids.append(word_to_id[word])
                if for_transformer:
                    ids.append(1)

                max_sentence_len = max(max_sentence_len, len(ids))

                ids = np.asarray(ids)
                raw_x.append(ids)
                raw_y.append(label)

        self.raw_x = raw_x
        self.raw_y = raw_y
        self.num_examples = len(raw_x)
        max_info['id'], max_info['sentence_len'] = curr_id, max_sentence_len
        print("--- Loaded {} sentences".format(self.num_examples))

    def __len__(self):
        return self.num_examples

    def __getitem__(self, i):
        sample = (torch.tensor(self.raw_x[i], dtype=torch.long), self.raw_y[i])
        return sample


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

subphrase_loc_mappings = [(0, 3), (0, 16), (1, 4), (1, 17), (2, 7), (2, 20),
                          (3, 8), (3, 21), (4, 11), (4, 24), (5, 12), (5, 25),
                          (6, LOC[3:5]), (6, LOC[16:18]),
                          (7, LOC[7:9]), (7, LOC[20:22]),
                          (8, LOC[11:13]), (8, LOC[24:26]),
                          (9, LOC[7:13]), (9, LOC[20:26]),
                          (10, LOC[5:13]), (10, LOC[18:26]), (11, LOC[:])]
subphrase_loss_weighting = [1. / 6] * 6 + [1. / 3] * 3 + [0.8] * 2 + [1.]

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
                          "alternation": 8, "cover": 9,}
            self.label_dict = label_dict
            self.output_remapping = torch.tensor([0,1,1,0,2,2,0], dtype=torch.long)
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

            [CLS] | not | every | bad | singer | does | not | badly | sings | <e> | every | good | song ]
            0       1    2       3      4       5      6     7       8       9     10      11     12     13 14 15 16
        [0]       | <P> | <P>   | bad | <P>    | <P>  | ...
        [1]       | <P> | ...   | <P> | singer | <P>  | ...
        [2]       | <P> | ...                     ... | <P> | badly | <P>   | ...
        [3]       | <P> | ...                           ... | <P>   | sings | <P> | ...
        [4]       | <P> | ...                                                         <P> | good | <P>  |
        [5]       | <P> | ...                                                         ... | <P>  | song |
        [6]       | <P> | <P>   | bad | singer | <P>  | ...
        [7]       | <P> | ...                     ... | <P> | badly | sings | <P> | ...
        [8]       | <P> | ...                                                 ... |   <P> | good | song |
        [9]       | <P> | ...                     ... | <P> | badly | sings | not | every | good | song |
        [10]      | <P> | ...         | <P>    | does | not | badly | sings | not | every | good | song |
        [11] <the whole sentence>
        :param i: idx
        :return: torch.Tensor
        """
        sentence = self.raw_bert_x[i]
        res = torch.full((12, 27), self.word_to_id["[PAD]"], dtype=torch.long)
        res[:,0] = self.word_to_id["[CLS]"]
        res[:,13] = self.word_to_id["[SEP]"]
        res[:,26] = self.word_to_id["[SEP]"]

        for loc in subphrase_loc_mappings:
            res[loc] = torch.tensor(sentence[loc[1]], dtype=torch.long)
        return res

    def generate_subphrase_attn_masks(self, i):
        res = torch.full((12, 27), 0.0, dtype=torch.float)
        res[:, 0] = 1.0
        res[:, 13] = 1.0
        res[:, 26] = 1.0

        for loc in subphrase_loc_mappings:
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
        elif self.variant == "subphrase":
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


def bert_subsequence_collate(batch):
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

