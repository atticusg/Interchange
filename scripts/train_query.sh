MODEL=$1
shift
DIFFICULTY=$2
shift
DATE=$3
shift
python train.py query \
    -d "mqnli_models/${MODEL}-${DIFFICULTY}-${DATE}.db" "$@"
