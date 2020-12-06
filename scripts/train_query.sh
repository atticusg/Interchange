MODEL=$1
shift
DIFFICULTY=$1
shift
python train.py query -d "mqnli_models/${MODEL}-${DIFFICULTY}.db" "$@"
