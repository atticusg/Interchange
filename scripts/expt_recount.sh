MODEL=$1
shift
DIFFICULTY=$1
shift
python experiment.py analyze \
    -d "experiment_data/${MODEL}/${MODEL}-${DIFFICULTY}.db" \
    -i "python expt_interchange_analysis.py" \
    -x \
    -m "nlprun -q john -a hanson-intervention" \
    -b 22 \
    -l "experiment_data/${MODEL}/${DIFFICULTY}/batched_runs/" \
    "$@"


# -n num_expts -r 2 -s -2

