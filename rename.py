from experiment import db_utils as db


db_path = "experiment_data/lstm/lstm-easy.db"
date = "-dec4"

rows = db.select(db_path, "results", cols=["id", "save_path", "res_save_dir"])

for row in rows:
    id = row["id"]
    save_path = row["save_path"].replace(date, "")
    res_save_dir = row["res_save_dir"].replace(date, "")
    print("id", id, "save_path", save_path, "res_save_dir", res_save_dir)
    db.update(db_path, "results", {"save_path": save_path, "res_save_dir": res_save_dir}, id=id)
