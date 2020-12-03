MODEL=$1
DIFFICULTY=$2
DATE=$3
python experiment.py setup \
    -d experiment_data/$MODEL/$MODEL-$DIFFICULTY-$DATE.db \
    -m mqnli_models/$MODEL-$DIFFICULTY-best.pt \
    -i mqnli_data/mqnli-$MODEL-default.pt
