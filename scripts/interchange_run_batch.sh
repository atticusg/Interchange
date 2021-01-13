MODEL=$1
shift
DIFFICULTY=$1
shift
python interchange.py run \
    -d "data/interchange/${MODEL}/${MODEL}-${DIFFICULTY}.db" \
    -x \
    -m "scripts/metascript.sh" \
    -b 7 \
    -l "data/interchange/${MODEL}/${DIFFICULTY}/batched_runs/" \
    "$@"
# -n <NUMBER> -s <STARTED_STATUS>

