MODEL=$1
DIFFICULTY=$2
python train.py add_grid_search \
    -d mqnli_models/$MODEL-$DIFFICULTY.db \
    -r 3 \
    -o mqnli_models/$MODEL-$DIFFICULTY/
