DB=$1
shift
python experiment.py query -d "experiment_data/sep/${DB}.db" "$@"

