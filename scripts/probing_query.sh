MODEL=$1
shift
DIFFICULTY=$1
shift
python probing.py query -d "probing_results/${MODEL}-${DIFFICULTY}.db" "$@"
