python interchange.py \
     --data_path "data/mqnli/preprocessed/bert-easy.pt" \
     --model_path "data/models/bert-hard-best.pt" \
     --abstraction '["sentence_q", ["bert_layer_0"]]' \
     --num_inputs 200 \
     --model_type "bert" \
     --interchange_batch_size 800 \
     --graph_alpha 0 \
     --res_save_dir "data/interchange/test/bert1/" \
     --save_intermediate_results 1 \
     --loc_mapping_type "bert_cls_only"

# '["sentence_q", ["lstm_0"]]'
# '["sentence_q", ["bert_layer_0"]]'