MODEL=$1
shift
DIFFICULTY=$1
shift
python train_manager.py query -d "data/training/${MODEL}-${DIFFICULTY}.db" "$@"
