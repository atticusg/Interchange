BATCH_SIZE=$1
shift
python interchange_manager.py analyze_graph \
    -d "data/interchange/rand-bert-hard.db" \
    -i "python graph_analysis.py" \
    -m "scripts/graph_metascript.sh" \
    -l "data/interchange/rand-bert-hard/graph_batched_runs/" \
    -r 1 \
    -b $BATCH_SIZE \
    "$@"
# -n <NUMBER> -s <STARTED_STATUS>
