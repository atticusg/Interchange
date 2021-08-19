python interchange_manager.py add \
    -d data/interchange/rand-bert-hard.db \
    -t rand-bert \
    -m data/models/bert-hard-best.pt \
    -o data/interchange/rand-bert-hard/ \
    -n 1000 \
    "$@"
# -l bert_cls_only
