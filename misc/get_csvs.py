import os
import numpy as np
import json
from experiment import db_utils as db


cols = ["id", "max_clique_sizes", "visualize_save_paths"]
cond_dict = {"status": 4}
like_dict = {"abstraction": "%lstm_0%"}
rows = db.select("../experiment_data/sep/nov2-500.db", "results", cols=cols,
                 cond_dict=cond_dict, like=like_dict)

for row in rows:
    clq_sizes = json.loads(row["max_clique_sizes"])
    save_paths = json.loads(row["visualize_save_paths"])
    idx_expt_with_larget_clq = np.argmax(clq_sizes)
    save_path = save_paths[idx_expt_with_larget_clq]

    remote_path = os.path.join("Interchange", save_path)
    cmd = f"scp hansonlu@sc.stanford.edu:{remote_path} experiment_data/clqviz"
    print(cmd)
    os.system(cmd)
