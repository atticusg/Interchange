MODEL=$1
shift
DIFFICULTY=$1
shift
python train.py run \
    -d "mqnli_models/${MODEL}-${DIFFICULTY}.db" \
    -i "python train_${MODEL}.py" \
    -m "scripts/metascript.sh" \
    "$@"

