MODEL=$1
DIFFICULTY=$2
python experiment.py setup \
    -d experiment_data/$MODEL-$DIFFICULTY/$MODEL-$DIFFICULTY.db \
    -m mqnli_models/$MODEL-$DIFFICULTY-best.pt \
    -i mqnli_data/mqnli-$MODEL-default.pt
