MODEL=$1
shift
DIFFICULTY=$1
shift
python probe_manager.py add_grid_search \
    --db_path "data/probing/$MODEL-$DIFFICULTY.db"\
    --res_save_dir "data/probing/$MODEL-$DIFFICULTY/" \
    "$@"
