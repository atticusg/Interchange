import torch
import json
from torch.utils.data import DataLoader

from transformers import BertTokenizer


from mqnli import MQNLIData
from utils import my_collate
from tqdm import tqdm

import argparse
import pickle

def print_words_by_category(save_file):
    with open(save_file, "rb") as f:
        pos_to_words = pickle.load(f)

    categories = ["quantifier", "adj", "noun", "neg", "adv", "verb",
                  "quantifier", "adj", "noun"]

    for i, (cat, toks_set) in enumerate(zip(categories, pos_to_words)):
        toks_list = list(toks_set)
        print(f"{i} {cat} {toks_list[:10]}")



    categories_to_chopped_tokens = {
        "adj": set(),
        "noun": set(),
        "adv": set(),
        "verb": set()
    }

    for category, toks_set in zip(categories, pos_to_words):
        if category in categories_to_chopped_tokens:
            for toks in toks_set:
                categories_to_chopped_tokens[category].add(toks)

    categories_to_chopped_words = {
        "adj": set(),
        "noun": set(),
        "adv": set(),
        "verb": set()
    }

    categories_to_all_words = {
        "adj": [],
        "noun": [],
        "adv": [],
        "verb": []
    }

    for category, toks_set in categories_to_chopped_tokens.items():
        for toks in toks_set:
            full_word = "".join(t.strip("#") for t in toks)
            if category != "adv":
                if len(toks) > 1:
                    categories_to_chopped_words[category].add(full_word)
            else:
                if len(toks) > 2:
                    categories_to_chopped_words[category].add(full_word)
            categories_to_all_words[category].append(full_word)

    for category in categories_to_all_words:
        if category == "noun": continue
        categories_to_all_words[category].sort()
        save_file = f"data/tokenization/vocab-{category}.txt"
        with open(save_file, "w") as f:
            for word in categories_to_all_words[category]:
                if word in categories_to_chopped_words[category]:
                    f.write(f"{word},____\n")
                else:
                    f.write(f"{word},{word}\n")




def get_words_by_position(data, dataset, pos_to_words, tokenizer):
    collate_fn = lambda batch: my_collate(batch, batch_first=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            collate_fn=collate_fn)

    for input_tuple in tqdm(dataloader):
        input_value = input_tuple[0].squeeze()
        str_input = [data.id_to_word[i.item()] for i in input_value]
        tokenized_input = [tuple(tokenizer.tokenize(word_str)) for word_str in str_input]
        for pos in range(len(pos_to_words)):
            pos_to_words[pos].add(tokenized_input[pos])
            pos_to_words[pos].add(tokenized_input[9 + pos])


def load_remapping(remap_files):
    remapping = {"doesnot": "doesn't"}
    # old_words = []
    # new_words = []

    for file_name in remap_files:
        with open(file_name, "r") as f:
            for line in f:
                w1, w2 = line.strip().split(",")
                remapping[w1] = w2
                # old_words.append(w1)
                # new_words.append(w2)

    return remapping

    # for w in old_words:
    #     if w not in remapping:
    #         print("Old word not found in remapping:", w)

    # for i, w in enumerate(old_words):
    #     for j in range(i+1, len(old_words)):
    #         w2 = old_words[j]
    #         if w == w2:
    #             print("found repeated old word:", w)

    # print(f"remapping len {len(remapping)} old_words len {len(old_words)} new_words len {len(new_words)}")

def main():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    train_path = "data/tokenization/mqnli.train.txt"
    dev_path = "data/tokenization/mqnli.dev.txt"
    test_path = "data/tokenization/mqnli.test.txt"

    data = MQNLIData(train_path, dev_path, test_path)

    # add new tokens:
    new_tokens = ["emptystring", "notevery"]
    tokenizer.add_tokens(new_tokens)

    remapping_files = ["data/tokenization/remapping-adj.txt",
                       "data/tokenization/remapping-adv.txt",
                       "data/tokenization/remapping-noun.txt",
                       "data/tokenization/remapping-verb.txt"]
    remapping = load_remapping(remapping_files)
    print("remapping afghani:", remapping["Afghani"])

    words = [data.id_to_word[i] for i in range(data.vocab_size)]
    remapped_words = [remapping[w] if w in remapping else w for w in words]
    print(remapped_words)
    tokenized_words = [tokenizer.tokenize(w) for w in remapped_words]
    print("tokenized_words", tokenized_words)
    chopped_up_words = [toks for toks in tokenized_words if len(toks) > 1]
    print(f"Found {len(chopped_up_words)} chopped up words among total of {len(tokenized_words)}")
    print(chopped_up_words)

    with open(dev_path, "r") as f:
        count = 0
        for line in f:
            if count == 10: break
            example = json.loads(line.strip())
            sentence1 = example["sentence1"]
            remapped_prem = " ".join(remapping[w] if w in remapping else w
                for w in sentence1.split())
            toks_prem = tokenizer.tokenize(remapped_prem)
            print("premise:", remapped_prem)
            print("tokenized_premise", toks_prem)

            sentence2 = example["sentence2"]
            remapped_hypo = " ".join(remapping[w] if w in remapping else w
                for w in sentence2.split())
            toks_hypo = tokenizer.tokenize(remapped_hypo)
            print("hypothesis:", remapped_hypo)
            print("tokenized_hypo", toks_hypo)

            count += 1


    # # analyze words by position
    # pos_to_words = []
    # for i in range(9):
    #     pos_to_words.append(set())
    #
    # get_words_by_position(data, data.dev, pos_to_words, tokenizer)
    # get_words_by_position(data, data.train, pos_to_words, tokenizer)
    # get_words_by_position(data, data.test, pos_to_words, tokenizer)
    #
    # save_path = "data/tokenization/pos_to_words.pkl"
    # with open(save_path, "wb") as f:
    #     pickle.dump(pos_to_words, f)


def main2():
    save_path = "data/tokenization/pos_to_words.pkl"
    print_words_by_category(save_path)


def verify_rewording(path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    new_tokens = ["emptystring", "notevery"]
    tokenizer.add_tokens(new_tokens)
    words = set()
    orig_words = set()
    mappings = []
    with open(path, "r") as f:
        for line in f:
            w1, w2 = line.strip().split(",")
            if "!" in w1 or "_" in w2 or "#" in w1:
                mappings.append((w1, w2))
                continue
            words.add(w1)
            orig_words.add(w1)
            if w2 != w1 and w2 in words:
                w1 = "#" + w1
                mappings.append((w1, w2))
                continue
            words.add(w2)
            toks = tokenizer.tokenize(w2)
            if len(toks) != 1 or w2 == "":
                w1 = "!" + w1
            mappings.append((w1, w2))

    for i, (w1, w2) in enumerate(mappings):
        if "!" in w1 or "_" in w2 or "#" in w1:
            continue
        if w1 != w2 and w2 in orig_words:
            w1 = "#" + w1
            mappings[i] = (w1, w2)

    with open(path, "w") as f:
        for w1, w2 in mappings:
            f.write(f"{w1},{w2}\n")

def main3():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    opts = parser.parse_args()
    verify_rewording(opts.path)

if __name__ == "__main__":
    main()