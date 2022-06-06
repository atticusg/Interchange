import torch
from tqdm import tqdm
import pickle

from torch.utils.data import DataLoader
from datasets.mqnli import MQNLIBertData
from transformers import BertTokenizer
VOCAB_PATH = "data/tokenization/bert-vocab.txt"
NEW_VOCAB_PATH = "data/tokenization/raw-bert-vocab.txt"

def main():
    tokenizer = BertTokenizer(NEW_VOCAB_PATH)
    print("len of tokenizer", len(tokenizer))

    tokenizer.add_tokens(['emptystring'])
    print("len of tokenizer")

    # mqnli_bert_data: MQNLIBertData = torch.load("data/mqnli/preprocessed/bert-hard.pt",
    #                                             map_location=torch.device('cpu'))
    toks_in_vocab_path = set()

    with open(VOCAB_PATH, "r") as f:
        for line in f:
            toks_in_vocab_path.add(line.strip())

    toks_in_new_vocab_path = set()
    with open(NEW_VOCAB_PATH, "r") as f:
        for line in f:
            toks_in_new_vocab_path.add(line.strip())

    print('New vocab size', len(toks_in_new_vocab_path))
    print('emptystring in new vocab?', 'emptystring' in toks_in_new_vocab_path)

    # with open("data/tokenization/id_to_toks.pkl", "rb") as f:
    #     encountered_id_to_toks = pickle.load(f)
    #
    # print("All toks in vocab file?", all(tok in toks_in_vocab_path for tok in encountered_id_to_toks.values()))
    #
    # all_toks = sorted((id, tok) for id, tok in encountered_id_to_toks.items())
    # with open(NEW_VOCAB_PATH, "w") as f:
    #     for id, tok in all_toks:
    #         f.write(f"{tok}\n")

    # for dataset in mqnli_bert_data.train, mqnli_bert_data.dev, mqnli_bert_data.test:
    #     dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    #     for batch in tqdm(dataloader):
    #         tok_ids = batch[0].view(-1).tolist()
    #         tok_strs = tokenizer.convert_ids_to_tokens(tok_ids)
    #         for tok_id, tok_str in zip(tok_ids, tok_strs):
    #             encountered_id_to_toks[tok_id] = tok_str


if __name__ == '__main__':
    main()
