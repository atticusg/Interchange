MODEL=$1
shift
DIFFICULTY=$1
shift
DATE=$1
shift
python experiment.py query \
    -d "experiment_data/${MODEL}/${MODEL}-${DIFFICULTY}-${DATE}.db" \
    "$@"
