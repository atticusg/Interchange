MODEL=$1
shift
DIFFICULTY=$1
shift
python experiment.py analyze_graph \
    -d "experiment_data/${MODEL}/${MODEL}-${DIFFICULTY}.db" \
    -i "python expt_graph.py" \
    -x \
    -m "nlprun -q john -a hanson-intervention" \
    -r 1 \
    "$@"
# -n num_expts -s -2
