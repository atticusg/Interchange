MODEL=$1
DIFFICULTY=$2
DATE=$3
python train.py add_grid_search \
    -d mqnli_models/$MODEL-$DIFFICULTY-$DATE.db \
    -r 4 \
    -o mqnli_models/$MODEL-$DIFFICULTY-$DATE/