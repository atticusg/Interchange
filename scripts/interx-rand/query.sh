MODEL=$1
shift
DIFFICULTY=$1
shift
python interchange_manager.py query \
    -d "data/interchange/rand-bert-hard.db" \
    "$@"
