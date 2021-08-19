MODEL=$1
shift
DIFFICULTY=$1
shift
python interchange_manager.py analyze \
    -d "data/interchange/${MODEL}-${DIFFICULTY}.db" \
    -i "python expt_interchange_analysis.py" \
    -x \
    -b 22 \
    -l "data/interchange/${MODEL}-${DIFFICULTY}/batched_runs/" \
    "$@"
# -n num_expts -r 2 -s -2
# -m <metascript>

