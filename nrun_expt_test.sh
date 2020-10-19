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