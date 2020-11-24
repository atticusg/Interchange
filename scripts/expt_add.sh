MODEL=$1
DIFFICULTY=$2
python experiment.py add \
    -d experiment_data/$MODEL-$DIFFICULTY/$MODEL-$DIFFICULTY.db \
    -t $MODEL \
    -m mqnli_models/$MODEL-$DIFFICULTY-best.pt \
    -o experiment_data/$MODEL-$DIFFICULTY/ \
    -n 500
