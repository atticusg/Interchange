MODEL=$1
shift
DIFFICULTY=$1
shift
python interchange_manager.py add \
    -d data/interchange/$MODEL-$DIFFICULTY.db \
    -t $MODEL \
    -m data/models/$MODEL-$DIFFICULTY-best.pt \
    -o data/interchange/$MODEL-$DIFFICULTY/ \
    -n 1000 \
    "$@"
# -l bert_cls_only
