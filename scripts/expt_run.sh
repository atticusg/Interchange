MODEL=$1
DIFFICULTY=$2
shift
python experiment.py run \
    -d "experiment_data/${MODEL}-${DIFFICULTY}/${MODEL}-${DIFFICULTY}.db" \
    -m "scripts/metascript.sh" \
    "$@"
# -n <NUMBER> -s <STARTED_STATUS>
