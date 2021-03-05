MODEL=$1
shift
DIFFICULTY=$1
shift
python probe_manager.py query -d "data/probing/${MODEL}-${DIFFICULTY}.db" "$@"
