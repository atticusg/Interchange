from experiment import db_utils as db


db_path = "experiment_data/bert/bert-easy.db"
date = "nov16"

rows = db.select(db_path, "results", cols=["id", "save_path","log_path", "res_save_dir", "graph_save_paths"])

for row in rows:
    id = row["id"]
    save_path = row["save_path"].replace(date, "easy")
    res_save_dir = row["res_save_dir"].replace(date, "easy")
    log_path = row["log_path"].replace(date, "easy")
    graph_save_paths = row["graph_save_paths"].replace(date, "easy")
    db.update(db_path, "results", {"save_path": save_path, "log_path": log_path, "res_save_dir": res_save_dir, "graph_save_paths":graph_save_paths}, id=id)
