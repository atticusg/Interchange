import torch
import numpy as np
from torch.utils.data import Dataset


class SentimentData:
    def __init__(self, train_file, dev_file, test_file, ngram=0, for_transformer=False):
        self.output_classes = 2
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

        if ngram > 0:
            print("--- USING NGRAM MODEL ---")

        print("--- Loading Dataset ---")
        self.train = SentimentDataset(train_file, self.word_to_id, self.id_to_word,
                                      max_info, ngram, for_transformer)
        print("--- finished loading train set")
        self.dev = SentimentDataset(dev_file, self.word_to_id, self.id_to_word,
                                    max_info, ngram, for_transformer)
        print("--- finished loading dev set")
        self.test = SentimentDataset(test_file, self.word_to_id, self.id_to_word,
                                     max_info, ngram, for_transformer)
        print("--- finished loading test set")
        self.vocab_size, self.max_sentence_len = max_info['id'], max_info['sentence_len']
        print("--- found {} unique words, max sentence len is {}".format(self.vocab_size, self.max_sentence_len))

class SentimentDataset(Dataset):
    def __init__(self, file_name, word_to_id, id_to_word, max_info, ngram, for_transformer=False):
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
                if for_transformer:
                    ids.append(1)
                for word in sentence:
                    if word not in word_to_id:
                        word_to_id[word] = curr_id
                        id_to_word[curr_id] = word
                        curr_id += 1
                    ids.append(word_to_id[word])
                if for_transformer:
                    ids.append(2)
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