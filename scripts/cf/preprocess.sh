MODEL=$1
DIFFICULTY=$2
VARIANT=$3
python train_manager.py preprocess \
    $MODEL \
    "data/mqnli/raw/${DIFFICULTY}/train.txt" \
    "data/mqnli/raw/${DIFFICULTY}/dev.txt" \
    "data/mqnli/raw/${DIFFICULTY}/test.txt" \
    -o "data/mqnli/preprocessed/${MODEL}-${DIFFICULTY}.pt" \
    -v "${VARIANT}"


