MODEL=$1
DIFFICULTY=$2
python train.py add_grid_search \
    -d data/training/$MODEL-$DIFFICULTY.db \
    -r 3 \
    -o data/training/$MODEL-$DIFFICULTY/
