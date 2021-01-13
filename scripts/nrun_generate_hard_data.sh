NOW=$(date +"%Y%m%d-%H%M%S")
nlprun -a hanson-intervention -q john -r 64G -o data/mqnli/raw/hard/output-$NOW.log \
    'python mqnli/generate_data.py 500000 data/mqnli/raw/hard mqnli/data'
