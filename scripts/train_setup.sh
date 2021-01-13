MODEL=$1
DIFFICULTY=$2
python train.py setup \
    -d data/training/$MODEL-$DIFFICULTY.db \
    -i data/mqnli/preprocessed/bert-$DIFFICULTY.pt