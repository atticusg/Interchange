MODEL=$1
DIFFICULTY=$2
python interchange_manager.py setup \
    -d data/interchange/$MODEL/$MODEL-$DIFFICULTY.db \
    -m data/models/$MODEL-$DIFFICULTY-best.pt \
    -i data/mqnli/preprocessed/bert-easy.pt
