MODEL=$1
shift
DIFFICULTY=$1
shift
python interchange_manager.py analyze_graph \
    -d "data/interchange/${MODEL}-${DIFFICULTY}.db" \
    -i "python graph_analysis.py" \
    -x \
    -m "nlprun -q john -a hanson-intervention" \
    -r 1 \
    -b 7 \
    "$@"
# -n <NUMBER> -s <STARTED_STATUS>

