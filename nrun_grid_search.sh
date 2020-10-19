NOW=$(date +"%Y%m%d-%H%M%S")
LAYERS=$1
nlprun \
    -a hanson-intervention \
    -g 1 \
    -o experiment_data/sep_gs_${NOW}.log \
    "python run_grid_search.py $LAYERS"
