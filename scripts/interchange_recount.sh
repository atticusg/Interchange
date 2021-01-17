MODEL=$1
shift
DIFFICULTY=$1
shift
python expt_managers/interchange_manager.py analyze \
    -d "data/interchange/${MODEL}/${MODEL}-${DIFFICULTY}.db" \
    -i "python expt_interchange_analysis.py" \
    "$@"

# -n num_expts -r 2 -s -2

