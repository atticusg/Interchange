MODEL=$1
shift
DIFFICULTY=$1
shift
python probe_manager.py run \
    --db_path "data/probing/$MODEL-$DIFFICULTY.db" \
    -x \
    -m "scripts/metascript.sh" \
    "$@"
# -n <NUMBER> -s <STARTED_STATUS>
