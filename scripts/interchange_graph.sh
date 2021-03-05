MODEL=$1
shift
DIFFICULTY=$1
shift
python interchange_manager.py analyze_graph \
    -d "data/interchange/${MODEL}-${DIFFICULTY}.db" \
    -i "python graph_analysis.py" \
    -r 1 \
    "$@"
# -n num_expts -s -2
# -x
# -m <metascript>  \