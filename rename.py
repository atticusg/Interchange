from experiment import db_utils as db
import sys

def rename_experiment(path):
    rows = db.select(path, "results")
    for row in rows:
        # rename data paths
        update_dict = {}
        data_path = row["data_path"]
        data_path = data_path.replace("mqnli_data/mqnli_bert.pt", "data/mqnli/preprocessed/bert-easy.pt")
        data_path = data_path.replace("mqnli_data/mqnli-", "data/mqnli/preprocessed/")
        data_path = data_path.replace("default", "easy")
        update_dict["data_path"] = data_path

        model_path = row["model_path"]
        model_path = model_path.replace("bert_best.pt", 'bert-easy-best.pt')
        model_path = model_path.replace("mqnli_models/", "data/models/")
        update_dict["model_path"] = model_path

        for col in ["log_path", "res_save_dir", "save_path", "graph_save_paths", "visualize_save_paths"]:
            if col not in row: continue
            p = row[col]
            if p:
                p = p.replace("experiment_data/", "data/interchange/")
                p = p.replace("hard-nov3", "hard")
                update_dict[col] = p

        db.update(path, "results", update_dict, id=row["id"])
    print(f"Updated {len(rows)} rows in db {path}")

def rename_training(path):
    rows = db.select(path, "results")
    for row in rows:
        # rename data paths
        update_dict = {}

        data_path = row["data_path"].replace("mqnli_data/mqnli-", "data/mqnli/preprocessed/")
        data_path = data_path.replace("mqnli_bert.pt", "bert-easy.pt")
        update_dict["data_path"] = data_path

        if "tokenizer_vocab_path" in row:
            update_dict["tokenizer_vocab_path"] = row["tokenizer_vocab_path"].replace("mqnli_data/", "data/tokenization/")

        for col in ["model_save_path", "res_save_dir", "log_path"]:
            p = row[col]
            p = p.replace("bert-hard-nov24", "bert-hard")
            p = p.replace("bert-nov7", "bert-easy")
            p = p.replace("mqnli_models/", "data/training/")
            update_dict[col] = p

        db.update(path, "results", update_dict, id=row["id"])
    print(f"Updated {len(rows)} rows in db {path}")

def rename_probing(path):
    rows = db.select(path, "results")
    for row in rows:
        # rename data paths
        update_dict = {}
        data_path = row["data_path"]
        data_path = data_path.replace("mqnli_data/mqnli_bert.pt", "data/mqnli/preprocessed/bert-easy.pt")
        data_path = data_path.replace("mqnli_data/mqnli-", "data/mqnli/preprocessed/")
        data_path = data_path.replace("default", "easy")
        update_dict["data_path"] = data_path

        model_path = row["model_path"]
        model_path = model_path.replace("bert_best.pt", 'bert-easy-best.pt')
        model_path = model_path.replace("mqnli_models/", "data/models/")
        update_dict["model_path"] = model_path

        for col in ["log_path", "res_save_dir", "res_save_path"]:
            if col not in row: continue
            p = row[col]
            if p:
                p = p.replace("probing_results/", "data/probing/")
                update_dict[col] = p

        db.update(path, "results", update_dict, id=row["id"])
    print(f"Updated {len(rows)} rows in db {path}")

def rename_checkpoint(path):
    import torch
    ckpt = torch.load(path)
    if "tokenizer_vocab_path" in ckpt["model_config"]:
        tok_vocab_path = ckpt["model_config"]["tokenizer_vocab_path"]
        tok_vocab_path = tok_vocab_path.replace("mqnli_data/", "data/tokenization/")
        ckpt["model_config"]["tokenizer_vocab_path"] = tok_vocab_path
    torch.save(ckpt, path)
    print(f"Renamed checkpoint {path}")




def main():
    paths = sys.argv[1:]
    for path in paths:
        if "training" in path:
            rename_training(path)
        if "data/interchange" in path:
            rename_experiment(path)
        if "data/probing" in path:
            rename_probing(path)
        if path.endswith(".pt"):
            rename_checkpoint(path)

if __name__ == '__main__':
    main()

