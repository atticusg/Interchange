MODEL=$1
shift
DIFFICULTY=$1
shift
python experiment.py run \
    -d "experiment_data/${MODEL}/${MODEL}-${DIFFICULTY}.db" \
    "$@"
# -n <NUMBER> -s <STARTED_STATUS>
# -m "scripts/metascript.sh" \