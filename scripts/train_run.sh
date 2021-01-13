MODEL=$1
shift
DIFFICULTY=$1
shift
python train.py run \
    -d "data/training/${MODEL}-${DIFFICULTY}.db" \
    -i "python train_${MODEL}.py" \
    -x \
    -m "scripts/metascript.sh" \
    "$@"
# -n <NUM> -s <STARTED STATUS>

