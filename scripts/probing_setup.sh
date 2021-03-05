MODEL=$1
DIFFICULTY=$2
python probe_manager.py setup \
    --db_path "data/probing/$MODEL-$DIFFICULTY.db" \
    --model_path "data/models/$MODEL-$DIFFICULTY-best.pt" \
    --data_path "data/mqnli/preprocessed/bert-easy.pt"