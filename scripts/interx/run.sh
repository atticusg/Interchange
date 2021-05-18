MODEL=$1
shift
DIFFICULTY=$1
shift
python interchange_manager.py run \
    -d "data/interchange/${MODEL}-${DIFFICULTY}.db" \
    "$@"
# -n <NUMBER> -s <STARTED_STATUS>
# -m "scripts/metascript.sh" \

