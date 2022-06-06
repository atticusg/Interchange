python probe.py \
    --model_path "data/models/bert-hard-best.pt" \
    --data_path "data/mqnli/preprocessed/bert-easy.pt" \
    --model_type "bert" \
    --res_save_dir "probing_results/test_probe5/" \
    --probe_train_lr 0.01