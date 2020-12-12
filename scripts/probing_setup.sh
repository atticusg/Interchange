MODEL=$1
DIFFICULTY=$2
python probing.py setup \
    --db_path "probing_results/probing-$MODEL-$DIFFICULTY.db" \
    --model_path "mqnli_models/$MODEL-$DIFFICULTY-best.pt" \
    --data_path "mqnli_data/mqnli-$MODEL-default.pt"