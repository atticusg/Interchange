MODEL=$1
DIFFICULTY=$2
shift
python experiment.py query -d "experiment_data/${MODEL}-${DIFFICULTY}/${MODEL}-${DIFFICULTY}.db" "$@"
