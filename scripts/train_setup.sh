MODEL=$1
DIFFICULTY=$2
python train.py setup \
    -d mqnli_models/$MODEL-$DIFFICULTY.db \
    -i mqnli_data/mqnli-$MODEL-$DIFFICULTY.pt