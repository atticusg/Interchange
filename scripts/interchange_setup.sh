MODEL=$1
DIFFICULTY=$2
python interchange.py setup \
    -d experiment_data/$MODEL/$MODEL-$DIFFICULTY.db \
    -m data/models/$MODEL-$DIFFICULTY-best.pt \
    -i data/preprocessed/$MODEL-easy.pt
