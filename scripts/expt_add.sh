MODEL=$1
DIFFICULTY=$2
DATE=$3
python experiment.py add \
    -d experiment_data/$MODEL/$MODEL-$DIFFICULTY-$DATE.db \
    -t $MODEL \
    -m mqnli_models/$MODEL-$DIFFICULTY-best.pt \
    -o experiment_data/$MODEL/$DIFFICULTY-$DATE/ \
    -n 500
