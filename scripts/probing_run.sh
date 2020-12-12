MODEL=$1
shift
DIFFICULTY=$1
shift
python probing.py run \
    --db_path "probing_results/probing-$MODEL-$DIFFICULTY.db" \
    "$@"
# -n <NUMBER> -s <STARTED_STATUS>

