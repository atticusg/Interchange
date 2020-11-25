MODEL=$1
DIFFICULTY=$2
DATE=$3
python train.py setup \
    -d mqnli_models/$MODEL-$DIFFICULTY-$DATE.db \
    -i mqnli_data/mqnli-$MODEL-$DIFFICULTY.pt