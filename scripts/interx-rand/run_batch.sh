BATCH_SIZE=$1
shift
python interchange_manager.py run \
    -d "data/interchange/rand-bert-hard.db" \
    -x \
    -m "scripts/metascript.sh" \
    -b $BATCH_SIZE \
    -l "data/interchange/rand-bert-hard/batched_runs/" \
    "$@"
# -n <NUMBER> -s <STARTED_STATUS>

