MODEL=$1
DIFFICULTY=$2
VARIANT=$3
python train.py preprocess \
    $MODEL \
    "mqnli_data/mqnli-${DIFFICULTY}.train.txt" \
    "mqnli_data/mqnli-${DIFFICULTY}.dev.txt" \
    "mqnli_data/mqnli-${DIFFICULTY}.test.txt" \
    -o "mqnli_data/mqnli-${MODEL}-${DIFFICULTY}.pt" \
    -v "${VARIANT}"


