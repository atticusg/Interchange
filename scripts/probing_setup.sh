MODEL=$1
DIFFICULTY=$2
python probe.py setup \
    --db_path "data/probing/probing-$MODEL-$DIFFICULTY.db" \
    --model_path "data/models/$MODEL-$DIFFICULTY-best.pt" \
    --data_path "data/mqnli/preprocessed/$MODEL-easy.pt"