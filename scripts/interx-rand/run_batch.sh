python interchange_manager.py run \
    -d "data/interchange/rand-bert-hard.db" \
    -x \
    -m "scripts/metascript.sh" \
    -l "data/interchange/rand-bert-hard/batched_runs/" \
    "$@"
#  -b <BATCH_SIZE> -n <NUMBER> -s <STARTED_STATUS>

