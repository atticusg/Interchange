python graph_analysis.py \
     --save_path "data/interchange/test/lstm1/interx-res-sentence_q-0117-163916.pt" \
     --graph_alpha 100 \
     --res_save_dir "data/interchange/test/lstm1/"

# '["sentence_q", ["lstm_0"]]'
# '["sentence_q", ["bert_layer_0"]]'


# bert
# new version
# {
#   'max_clique_sizes': '[5, 25, 29, 5, 31, 29, 5]',
#   'avg_clique_sizes': '[3.3, 6.61, 7.27, 3.3, 7.43, 7.0, 3.3]',
#   'sum_clique_sizes': '[33, 119, 131, 33, 104, 119, 33]',
#   'clique_counts': '[10, 18, 18, 10, 14, 17, 10]',
#   'graph_save_path': 'data/causal_abstraction/test/bert1/graph-0120-104751.pkl'
#  }

# old version
# {
#   'max_clique_sizes': '[7, 31, 45, 7, 31, 29, 7]',
#   'avg_clique_sizes': '[3.46, 7.44, 9.0, 3.46, 7.42, 7.16, 3.46]',
#   'sum_clique_sizes': '[45, 134, 189, 45, 104, 129, 45]',
#   'clique_counts': '[13, 18, 21, 13, 14, 18, 13]',
#   'graph_save_path': 'data/causal_abstraction/test/bert1/graph-0120-105131.pkl'
# }

# lstm
# new version
# {
#   'max_clique_sizes': '[5, 5, 6, 4, 5, 5, 5]',
#   'avg_clique_sizes': '[3.1, 2.88, 3.19, 2.81, 2.84, 2.94, 2.875]',
#   'sum_clique_sizes': '[93, 98, 99, 90, 94, 100, 92]',
#   'clique_counts': '[30, 34, 31, 32, 33, 34, 32]',
#   'graph_save_path': 'data/causal_abstraction/test/lstm1/graph-0120-111313.pkl'}
#


# old version
# {
#   'max_clique_sizes': '[18, 22, 19, 21, 21, 20, 21]',
#   'avg_clique_sizes': '[3.57, 3.525, 3.62, 3.48, 3.38, 3.29, 3.64]',
#   'sum_clique_sizes': '[146, 141, 152, 146, 135, 148, 142]',
#   'clique_counts': '[41, 40, 42, 42, 40, 45, 39]',
#   'graph_save_path': 'data/causal_abstraction/test/lstm1/graph-0120-105819.pkl'
#   }