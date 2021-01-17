MODEL=$1
shift
DIFFICULTY=$1
shift
python probe_manager.py query -d "data/probing/probing-${MODEL}-${DIFFICULTY}.db" "$@"
