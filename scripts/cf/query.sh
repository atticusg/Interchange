EXPERIMENT=$1
shift
python cf_manager.py query -e $EXPERIMENT "$@"
