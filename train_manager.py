import os
import argparse

from experiment import db_utils as db
from experiment import ExperimentManager

DEFAULT_SCRIPT = "python train_bert.py"
HIGH_NODES = ["sentence_q", "subj_adj", "subj_noun",
              "neg", "v_adv", "v_verb", "vp_q",
              "obj_adj", "obj_noun", "obj", "vp", "v_bar", "negp", "subj"]
META_SCRIPT = "nlprun -a hanson-intervention -q john -r 100G"
REMAPPING_PATH="data/tokenization/bert-remapping.txt"
VOCAB_PATH = "data/tokenization/bert-vocab.txt"
RAW_BERT_VOCAB_PATH = "data/tokenization/raw-bert-vocab.txt"

DEFAULT_BERT_OPTS = {
    "data_path": "",
    "tokenizer_vocab_path": VOCAB_PATH,
    "non_pretrained": False,
    "num_hidden_layers": 6,
    "hidden_size": 768,
    "device": "cuda",
    "output_classes": 3,

    "batch_size": 32,
    "batch_first": True,
    "eval_batch_size": 64,

    "optimizer_type": "adamw",
    "lr": 0.,
    "lr_scheduler_type": "",
    "lr_warmup_ratio": 0.,
    "weight_norm": 0.,

    "max_epochs": 5,
    "run_steps": -1,
    "evals_per_epoch": 2,
    "patient_epochs": 400,

    'model_save_path': "finetuned_bert", #name of model
    'res_save_dir': "",
    "log_path": ""
}

DEFAULT_RAW_BERT_OPTS = {
    "data_path": "",
    "tokenizer_vocab_path": VOCAB_PATH,
    "num_hidden_layers": 6,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "device": "cuda",
    "output_classes": 3,

    "batch_size": 32,
    "batch_first": True,
    "eval_batch_size": 64,

    "optimizer_type": "adamw",
    "lr": 0.,
    "lr_scheduler_type": "",
    "lr_warmup_ratio": 0.,
    "weight_norm": 0.,

    "max_epochs": 20,
    "run_steps": -1,
    "evals_per_epoch": 4,
    "patient_epochs": 400,

    'model_save_path': "raw_bert", #name of model
    'res_save_dir': "",
    "log_path": ""
}

DEFAULT_LSTM_OPTS = {
    "data_path": "",
    "device": "cuda",

    "task": "mqnli",
    "output_classes": 3,
    "vocab_size": 0,
    "tokenizer_vocab_path": "",

    "embed_dim": 256,
    "lstm_hidden_dim": 128,
    "bidirectional": True,
    "num_lstm_layers": 4,
    "dropout": 0.,
    "embed_init_scaling": 0.1,
    "fix_embeddings": False,
    'batch_first': False,

    "batch_size": 64,
    "eval_batch_size": 2048,

    "optimizer_type": "adamw",
    "lr": 0.001,
    "lr_scheduler_type": "",
    "lr_warmup_ratio": 0.25,
    "lr_step_epochs": 2,
    "lr_step_decay_rate": 0.1,
    "weight_norm": 0.,

    "max_epochs": 200,
    "run_steps": -1,
    "evals_per_epoch": 5,
    "patient_epochs": 10,

    "model_save_path": "lstm",
    "res_save_dir": "",
    "log_path": ""
}

def preprocess(model_type, train, dev, test, data_path, variant):
    import torch
    from datasets.mqnli import MQNLIBertData, MQNLIData

    if model_type == "raw_bert":
        data = MQNLIBertData(train, dev, test, REMAPPING_PATH, vocab_file=RAW_BERT_VOCAB_PATH, variant=variant)
    elif model_type == "bert":
        data = MQNLIBertData(train, dev, test, REMAPPING_PATH, variant=variant)
    elif model_type == "lstm":
        data = MQNLIData(train, dev, test, variant=variant)
    else:
        raise NotImplementedError(f"Does not support model type {model_type}")
    torch.save(data, data_path)
    print(f"Saved preprocessed dataset to {data_path}")

def setup(db_path, data_path):
    import torch
    if "raw_bert" in db_path:
        default_opts = DEFAULT_RAW_BERT_OPTS.copy()
    elif "bert" in db_path:
        default_opts = DEFAULT_BERT_OPTS.copy()
    elif "lstm" in db_path:
        default_opts = DEFAULT_LSTM_OPTS.copy()
    else:
        raise ValueError(f"Cannot infer model type from database path {db_path}")

    default_opts["data_path"] = data_path
    if "hard" in data_path or "medium" in data_path:
        default_opts["output_classes"] = 10
    if "lstm" in db_path:
        if "raw_bert" in data_path:
            default_opts["tokenizer_vocab_path"] = RAW_BERT_VOCAB_PATH
        elif "bert" in data_path:
            default_opts["tokenizer_vocab_path"] = VOCAB_PATH
        else:
            data = torch.load(data_path)
            default_opts["vocab_size"] = data.vocab_size

    ExperimentManager(db_path, default_opts)

def add_one(db_path):
    manager = ExperimentManager(db_path)
    manager.insert({"lr": 5e-5, "max_epochs": 200, 'patient_epochs': 30})

def add_grid_search(db_path, repeat, res_save_dir):
    from datetime import datetime
    from itertools import product

    manager = ExperimentManager(db_path)

    if "raw_bert" in db_path:
        # Aug 6
        grid_dict = {
            "batch_size": [64],
            "hidden_size": [384, 768],
            "lr": [1e-6, 1e-5],
            "lr_scheduler_type": [""],
            "max_epochs": [20],
            "num_hidden_layers": [4, 8]
        }
    elif "bert" in db_path:
        # easy
        # grid_dict = {
        #     "batch_size": [32],
        #     "lr": [2e-5, 5e-5],
        #     "lr_scheduler_type": ["linear"],
        #     "lr_warmup_ratio": [0.5],
        #     "max_epochs": [4, 8, 20],
        # }
        # hard and medium
        # grid_dict = {
        #     "batch_size": [12],
        #     "lr": [2e-5, 5e-5],
        #     "max_epochs": [3,4],
        #     "lr_scheduler_type": ["linear"],
        #     "lr_warmup_ratio": [0.25],
        #     "evals_per_epoch": [8]
        # }
        # hard ablation
        grid_dict = {
            "batch_size": [32],
            "lr": [2e-5, 5e-5],
            "lr_scheduler_type": ["linear"],
            "lr_warmup_ratio": [0.5],
            "max_epochs": [3, 4],
        }
    elif "lstm" in db_path:
        if "easy" in db_path:
            grid_dict = {
                "batch_first": [True],
                "batch_size": [256],
                "lr": [0.001, 0.0001],
                "dropout": [0.1],
                "num_lstm_layers": [2, 4, 6],
                "lr_scheduler_type": [""],
                "evals_per_epoch": [8],
            }
        else:
            # hard
            grid_dict = {
                "batch_first": [True],
                "batch_size": [64],
                "lr": [0.001, 0.0001],
                "dropout": [0.1],
                "num_lstm_layers": [2, 4, 6],
                "lr_scheduler_type": [""],
                "evals_per_epoch": [8],
            }
    else:
        raise ValueError(f"Cannot infer model type from database path {db_path}")

    var_opt_names = list(grid_dict.keys())
    var_opt_values = list(v if isinstance(v, list) else list(v) \
                          for v in grid_dict.values())

    # treat elements in list as separate args to fxn
    for tup in product(*var_opt_values):
        update_dict = {}
        for name, val in zip(var_opt_names, tup):
            update_dict[name] = val
        for _ in range(repeat):
            id = manager.insert(update_dict)
            time_str = datetime.now().strftime("%m%d-%H%M%S")
            curr_save_dir = os.path.join(res_save_dir, f"expt-{id}-{time_str}")
            manager.update({"res_save_dir": curr_save_dir}, id)
            print("----inserted example into database:", update_dict)

    # if model_type == "lstm":
    #     manager = ExperimentManager(db_path, EXPT_OPTS)
    #     module, _ = load_model(LSTMModule, model_path, device=torch.device("cpu"))
    #     num_layers = module.num_lstm_layers
    #     time_str = datetime.now().strftime("%m%d-%H%M%S")
    #     for high_node in HIGH_NODES:
    #         for layer in range(num_layers):
    #             for n in num_inputs:
    #                 abstraction = f'["{high_node}",["lstm_{layer}"]]'
    #                 id = manager.insert({"abstraction": abstraction,
    #                                     "num_inputs": n})
    #                 res_save_dir = os.path.join(res_dir, f"expt-{id}-{time_str}")
    #                 manager.update({"model_path": model_path,
    #                                 "res_save_dir": res_save_dir}, id)
    # else:
    #     raise ValueError(f"Unsupported model type: {model_type}")


def run(db_path, script, n, detach, metascript, ready_status, started_status):
    if "lstm" in db_path:
        expt_opts = list(DEFAULT_LSTM_OPTS.keys())
    elif "raw_bert" in db_path:
        expt_opts = list(DEFAULT_RAW_BERT_OPTS.keys())
    elif "bert" in db_path:
        expt_opts = list(DEFAULT_BERT_OPTS.keys())
    else:
        raise ValueError(f"Cannot infer model type from database path {db_path}")

    manager = ExperimentManager(db_path, expt_opts)

    if os.path.exists(script):
        with open(script, "r") as f:
            script = f.read().strip()

    if metascript and os.path.exists(metascript):
        with open(metascript, "r") as f:
            metascript = f.read().strip()

    manager.run(launch_script=script, n=n, detach=detach, metascript=metascript,
                ready_status=ready_status, started_status=started_status)


def query(db_path, id=None, status=None, limit=None):
    manager = ExperimentManager(db_path)
    cols = ["id", "status", "batch_size", "lr", "res_save_dir", "model_save_path"]
    rows = manager.query(cols=cols, status=status, id=id, limit=limit)
    if len(rows) == 0:
        return "No data found"
    s = ", ".join(col for col in cols)
    print(s)
    print("-"*len(s))
    for row in rows:
        print(row)
        print("-------")

def update_status(db_path, ids, id_range, status):
    if id_range:
        ids = list(range(id_range[0], id_range[1] + 1))
    for i in ids:
        db.update(db_path, "results", {"status": status}, id=i)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")

    preprocess_parser = subparsers.add_parser("preprocess")
    preprocess_parser.add_argument("model_type", type=str, help="type of model")
    preprocess_parser.add_argument("train", type=str, help="Train set")
    preprocess_parser.add_argument("dev", type=str, help="Dev set")
    preprocess_parser.add_argument("test", type=str, help="Test set")
    preprocess_parser.add_argument("-o", "--data_path", type=str,
                                   help="Destination", required=True)
    preprocess_parser.add_argument("-v", "--variant", type=str, default="basic",
                                   help="Type of bert realignment variant")

    setup_parser = subparsers.add_parser("setup")
    setup_parser.add_argument("-d", "--db_path", type=str,
                              help="Experiment database path")
    setup_parser.add_argument("-i", "--data_path", type=str,
                              help="Path to pickled dataset")

    add_one_parser = subparsers.add_parser("add_one")
    add_one_parser.add_argument("-d", "--db_path", type=str, required=True,
                                help="Experiment database path")

    add_gs_parser = subparsers.add_parser("add_grid_search")
    add_gs_parser.add_argument("-d", "--db_path", type=str, required=True,
                               help="Experiment database path")
    add_gs_parser.add_argument("-r", "--repeat", type=int, default=1,
                               help="Repeat each grid search config for number "
                                    "of times")
    add_gs_parser.add_argument("-o", "--res_save_dir", type=str, required=True,
                               help="Directory to save stored results")


    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("-d", "--db_path", type=str, required=True)
    run_parser.add_argument("-i", "--script", type=str, default=DEFAULT_SCRIPT)
    run_parser.add_argument("-n", "--n", type=int, default=None)
    run_parser.add_argument("-x", "--detach", action="store_true")
    run_parser.add_argument("-m", "--metascript", type=str, default=None)
    run_parser.add_argument("-r", "--ready_status", type=int, default=0)
    run_parser.add_argument("-s", "--started_status", type=int, default=None)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("-d", "--db_path", type=str,
                              help="Experiment database path")
    query_parser.add_argument("-i", "--id", type=int)
    query_parser.add_argument("-s", "--status", type=int)
    query_parser.add_argument("-n", "--limit", type=int)

    update_parser = subparsers.add_parser("update_status")
    update_parser.add_argument("-d", "--db_path", type=str, required=True)
    update_parser.add_argument("-i", "--ids", type=int, nargs="*")
    update_parser.add_argument("-r", "--id_range", type=int, nargs=2)
    update_parser.add_argument("-s", "--status", type=int, required=True)

    kwargs = vars(parser.parse_args())
    globals()[kwargs.pop("subparser")](**kwargs)


if __name__ == "__main__":
    main()
