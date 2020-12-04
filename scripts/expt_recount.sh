MODEL=$1
shift
DIFFICULTY=$1
shift
DATE=$1
shift
python experiment.py analyze \
    -d "experiment_data/${MODEL}/${MODEL}-${DIFFICULTY}-${DATE}.db" \
    -x \
    -m "nlprun -q john -a hanson-intervention" \
    -i "python expt_interchange_analysis.py" \
    -b 22 \
    -l "experiment_data/${MODEL}/${DIFFICULTY}-${DATE}/batched_runs/" \
    "$@"


# -n num_expts -r 2 -s -2

