MODEL=$1
shift
DIFFICULTY=$1
shift
python probe.py run \
    --db_path "data/probing/probing-$MODEL-$DIFFICULTY.db" \
    -x \
    -m "scripts/metascript.sh" \
    "$@"
# -n <NUMBER> -s <STARTED_STATUS>
