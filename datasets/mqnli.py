import torch
import json
import numpy as np
from torch.utils.data import Dataset


class MQNLIData:
    def __init__(self, train_file, dev_file, test_file, for_transformer=False, store_text=False):
        self.output_classes = 3
        self.word_to_id = {}
        self.id_to_word = {}
        self.id_to_word[0] = '<PADDING>'
        self.word_to_id['<PADDING>'] = 0
        max_info = {
            'id': 1,
            'sentence_len': 0
        }

        if for_transformer:
            self.id_to_word[1] = '<CLS>'
            self.word_to_id['<CLS>'] = 1
            self.id_to_word[2] = '<SEP>'
            self.word_to_id['<SEP>'] = 2
            max_info['id'] = 3

        print("--- Loading Dataset ---")
        self.train = MQNLIDataset(train_file, self.word_to_id, self.id_to_word,
                                  max_info, for_transformer, store_text)
        train_ids = max_info["id"]
        print("--- finished loading train set, saw %d unique tokens up to now" % train_ids)

        self.dev = MQNLIDataset(dev_file, self.word_to_id, self.id_to_word,
                                max_info, for_transformer, store_text)
        dev_ids = max_info["id"]
        print("--- finished loading dev set, saw %d unique tokens up to now" % dev_ids)

        self.test = MQNLIDataset(test_file, self.word_to_id, self.id_to_word,
                                 max_info, for_transformer, store_text)
        test_ids = max_info["id"]
        print("--- finished loading test set, saw %d unique tokens up to now" % test_ids)

        self.vocab_size, self.max_sentence_len = max_info['id'], max_info[
            'sentence_len']
        print("--- found {} unique words, max sentence len is {}".format(
            self.vocab_size, self.max_sentence_len))

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


if __name__ == "__main__":
    train_file = "../mqnli_data/mqnli.train.txt"
    dev_file = "../mqnli_data/mqnli.dev.txt"
    test_file = "../mqnli_data/mqnli.test.txt"
    data = MQNLIData(train_file, dev_file, test_file, for_transformer=False)
    pickle_file = "../mqnli_data/mqnli.pt"
    data.save(pickle_file)