python interchange_manager.py analyze_graph \
    -d "data/interchange/rand-bert-hard.db" \
    -m "scripts/graph_metascript.sh" \
    -i "python graph_analysis.py" \
    -r 1 \
    "$@"
# -n num_expts -s -2
# -x
# -m <metascript>  \