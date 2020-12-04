MODEL=$1
shift
DIFFICULTY=$1
shift
DATE=$1
shift
python experiment.py analyze_graph \
    -d "experiment_data/${MODEL}/${MODEL}-${DIFFICULTY}-${DATE}.db" \
    -i "python expt_graph.py" \
    -r 1 \
    "$@"
# -n num_expts -s -2
# -x -m "nlprun -q john -a hanson-intervention"