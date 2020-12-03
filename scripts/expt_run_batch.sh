MODEL=$1
shift
DIFFICULTY=$1
shift
DATE=$1
shift
python experiment.py \
    -d "experiment_data/${MODEL}/${MODEL}-${DIFFICULTY}-${DATE}.db" \
    -x \
    -m "scripts/metascript.sh" \
    -b 7 \
    -l "experiment_data/${MODEL}/${DIFFICULTY}-${DATE}/batched_runs/" \
    "$@"
