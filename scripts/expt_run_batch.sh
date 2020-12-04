MODEL=$1
shift
DIFFICULTY=$1
shift
python experiment.py run \
    -d "experiment_data/${MODEL}/${MODEL}-${DIFFICULTY}.db" \
    -x \
    -m "scripts/metascript.sh" \
    -b 7 \
    -l "experiment_data/${MODEL}/${DIFFICULTY}/batched_runs/" \
    "$@"
# -n <NUMBER> -s <STARTED_STATUS>