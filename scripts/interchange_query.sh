MODEL=$1
shift
DIFFICULTY=$1
shift
python interchange.py query \
    -d "experiment_data/${MODEL}/${MODEL}-${DIFFICULTY}.db" \
    "$@"
