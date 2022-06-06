EXPERIMENT=$1
shift
python mocf_manager.py run \
    -e $EXPERIMENT \
    -i "python mocf_train.py" \
    -x \
    -m "scripts/metascript.sh" \
    "$@"
# -n <NUM> -s <STARTED STATUS>

