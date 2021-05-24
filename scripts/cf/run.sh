EXPERIMENT=$1
shift
python cf_manager.py run \
    -e $EXPERIMENT \
    -i "python cf_train.py" \
    -x \
    -m "scripts/metascript.sh" \
    "$@"
# -n <NUM> -s <STARTED STATUS>

