MODEL=$1
DIFFICULTY=$2
python probing.py add_grid_search \
    --db_path "probing_results/probing-$MODEL-$DIFFICULTY.db"\
    --res_save_dir "probing_results/$MODEL-$DIFFICULTY/"
