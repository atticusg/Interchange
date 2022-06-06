python mocf_train.py \
  --data_path data/mqnli/preprocessed/bert-hard_abl.pt \
  --mo_base_weight 1.0 \
  --mo_cf_weight 1.0 \
  --mo_aug_weight 1.0 \
  --mo_probe_weight 1.0 \
  --cf_train_num_random_bases 640 \
  --cf_train_num_random_ivn_srcs 10 \
  --cf_eval_num_random_bases 160 \
  --cf_eval_num_random_ivn_srcs 10 \
  --mapping '{"vp": {"bert_layer_3": ":,10,:"}}'