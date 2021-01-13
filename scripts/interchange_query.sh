MODEL=$1
shift
DIFFICULTY=$1
shift
python interchange.py query \
    -d "data/interchange/${MODEL}/${MODEL}-${DIFFICULTY}.db" \
    "$@"
