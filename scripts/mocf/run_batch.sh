EXPERIMENT=$1
shift
python mocf_manager.py run \
    -e $EXPERIMENT \
    -i "python mocf_train.py" \
    -x \
    -m "scripts/metascript.sh" \
    -l "data/mocf/batched_runs/" \
    "$@"
#  -b <BATCH_SIZE> -n <NUMBER> -s <STARTED_STATUS>

