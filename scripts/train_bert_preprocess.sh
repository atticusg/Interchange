python train_bert.py preprocess \
    "mqnli_data/mqnli.train.txt" \
    "mqnli_data/mqnli.dev.txt" \
    "mqnli_data/mqnli.test.txt" \
    -o "mqnli_data/mqnli-bert.pt" \
    -v "basic"

