MODEL=$1
shift
DIFFICULTY=$1
shift
python probe.py query -d "probing_results/probing-${MODEL}-${DIFFICULTY}.db" "$@"
