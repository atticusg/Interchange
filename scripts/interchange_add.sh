MODEL=$1
shift
DIFFICULTY=$1
shift
python interchange.py add \
    -d experiment_data/$MODEL/$MODEL-$DIFFICULTY.db \
    -t $MODEL \
    -m mqnli_models/$MODEL-$DIFFICULTY-best.pt \
    -o experiment_data/$MODEL/$DIFFICULTY/ \
    -n 500 \
    "$@"
# -l bert_cls_only
