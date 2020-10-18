python expt_interchange.py \
     --data_path "mqnli_data/mqnli_sep.pt" \
     --model_path "mqnli_models/lstm_sep_best.pt" \
     --res_save_dir "experiment_data/sep" \
     --abstraction '["sentence_q", ["lstm_0"]]' \
     --num_inputs 20