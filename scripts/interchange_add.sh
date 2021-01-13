MODEL=$1
shift
DIFFICULTY=$1
shift
python interchange.py add \
    -d data/interchange/$MODEL/$MODEL-$DIFFICULTY.db \
    -t $MODEL \
    -m data/models/$MODEL-$DIFFICULTY-best.pt \
    -o data/interchange/$MODEL/$DIFFICULTY/ \
    -n 500 \
    "$@"
# -l bert_cls_only
