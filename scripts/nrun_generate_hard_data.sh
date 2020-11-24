NOW=$(date +"%Y%m%d-%H%M%S")
nlprun -a hanson-intervention -q john -r 64G -o mqnli_data/hard/output-$NOW.log \
    'python mqnli/generate_data.py 500000 mqnli_data/hard'
