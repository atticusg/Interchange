EXPERIMENT=$1
shift
python mocf_manager.py query -e $EXPERIMENT "$@"
