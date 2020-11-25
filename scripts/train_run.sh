MODEL=$1
shift
DIFFICULTY=$1
shift
DATE=$1
shift
python train.py run \
    -d "mqnli_models/${MODEL}-${DIFFICULTY}-${DATE}.db" \
    -m "scripts/metascript.sh" \
    "$@"