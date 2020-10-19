#python experiment.py preprocess \
#    "mqnli_data/mqnli.train.txt" \
#    "mqnli_data/mqnli.dev.txt" \
#    "mqnli_data/mqnli.test.txt" \
#    -o "mqnli_data/mqnli_sep.pt"

#python experiment.py setup \
#    --db_path "experiment_data/sep/test.db" \
#    --model_path "mqnli_models/lstm_sep_best.pt" \
#    --data_path "mqnli_data/mqnli_sep.pt" \
#    --res_save_dir "experiment_data/sep"

#python experiment.py add \
#    -d "experiment_data/sep/test.db" \
#    -t "lstm" \
#    -m "mqnli_models/lstm_sep_best.pt" \
#    -n 20 \
#    -o "experiment_data/sep"

python experiment.py run \
    -d "experiment_data/sep/test.db" \
    -n 10