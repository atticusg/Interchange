DIFFICULTY=$1
MODEL=$2
python train.py preprocess \
    $MODEL \
    "mqnli_data/mqnli-${DIFFICULTY}.train.txt" \
    "mqnli_data/mqnli-${DIFFICULTY}.dev.txt" \
    "mqnli_data/mqnli-${DIFFICULTY}.test.txt" \
    -o "mqnli_data/mqnli-${MODEL}-${DIFFICULTY}.pt" \
    -v "subphrase"

