import os
import argparse
import torch
from datetime import datetime

from itertools import product
from experiment import db_utils as db
from experiment import ExperimentManager

from datasets.mqnli import MQNLIBertData
from train import load_model
from modeling.lstm import LSTMModule


EXPT_OPTS = ["data_path", "model_path", "res_save_dir", "abstraction", "num_inputs"]
DEFAULT_SCRIPT = "python expt_train_bert.py"
HIGH_NODES = ["sentence_q", "subj_adj", "subj_noun", "neg", "v_adv", "v_verb", "vp_q", "obj_adj", "obj_noun", "obj", "vp", "v_bar", "negp", "subj"]
META_SCRIPT = "nlprun -a hanson-intervention -q john -r 100G"
REMAPPING_PATH="mqnli_data/bert-remapping.txt"
VOCAB_PATH = "mqnli_data/bert-vocab.txt"

DEFAULT_OPTS = {
    "data_path": "",
    "tokenizer_vocab_path": VOCAB_PATH,
    "device": "cuda",

    "batch_size": 32,
    "use_collate": False,
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

    'model_save_path': "pretrained_bert", #name of model
    'res_save_dir': "",
    "log_path": ""
}

def preprocess(train, dev, test, data_path):
    data = MQNLIBertData(train, dev, test, REMAPPING_PATH)
    torch.save(data, data_path)

def setup(db_path, data_path):
    default_opts = DEFAULT_OPTS.copy()
    default_opts["data_path"] = data_path
    manager = ExperimentManager(db_path, default_opts)

def add_one(db_path):
    manager = ExperimentManager(db_path)
    manager.insert({"lr": 5e-5, "max_epochs": 200, 'patient_epochs': 30})

def add_grid_search(db_path, repeat, res_save_dir):
    manager = ExperimentManager(db_path)
    grid_dict = {"batch_size": [32],
                 "lr": [2e-5, 5e-5],
                 "max_epochs": [4,8,20],
                 "lr_scheduler_type": ["linear"],
                 "lr_warmup_ratio": [0.5],
                 "evals_per_epoch": [8]}
    var_opt_names = list(grid_dict.keys())
    var_opt_values = list(v if isinstance(v, list) else list(v) for v in grid_dict.values())

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


def run(db_path, script, n, detach, metascript, metascript_batch, metascript_log_dir,
        ready_status, started_status):
    expt_opts = list(DEFAULT_OPTS.keys())
    manager = ExperimentManager(db_path, expt_opts)

    if os.path.exists(script):
        with open(script, "r") as f:
            script = f.read().strip()

    if metascript and os.path.exists(metascript):
        with open(metascript, "r") as f:
            metascript = f.read().strip()

    manager.run(launch_script=script, n=n, detach=detach,
                metascript=metascript, metascript_batch=metascript_batch,
                metascript_log_dir=metascript_log_dir,
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

    compile_data_parser = subparsers.add_parser("preprocess")
    compile_data_parser.add_argument("train", type=str, help="Train set")
    compile_data_parser.add_argument("dev", type=str, help="Dev set")
    compile_data_parser.add_argument("test", type=str, help="Test set")
    compile_data_parser.add_argument("-o", "--data_path", type=str, help="Destination", required=True)

    setup_parser = subparsers.add_parser("setup")
    setup_parser.add_argument("-d", "--db_path", type=str, help="Experiment database path")
    setup_parser.add_argument("-i", "--data_path", type=str, help="Path to pickled dataset")

    add_one_parser = subparsers.add_parser("add_one")
    add_one_parser.add_argument("-d", "--db_path", type=str, required=True, help="Experiment database path")

    add_gs_parser = subparsers.add_parser("add_grid_search")
    add_gs_parser.add_argument("-d", "--db_path", type=str, required=True, help="Experiment database path")
    add_gs_parser.add_argument("-r", "--repeat", type=int, default=1, help="Repeat each grid search config for number of times")
    add_gs_parser.add_argument("-o", "--res_save_dir", type=str, required=True, help="Directory to save stored results")
    
    # add_parser = subparsers.add_parser("add")
    # add_parser.add_argument("-d", "--db_path", type=str, required=True, help="Pickled dataset file")
    # add_parser.add_argument("-t", "--model_type", type=str, required=True, help="Model type, currently only supports lstm")
    # add_parser.add_argument("-m", "--model_path", type=str, required=True, help="Trained torch.nn.module")
    # add_parser.add_argument("-o", "--res_dir", type=str, required=True, help="Directory to save stored results")
    # add_parser.add_argument("-n", "--num_inputs", type=int, nargs="+")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("-d", "--db_path", type=str, required=True)
    run_parser.add_argument("-i", "--script", type=str, default=DEFAULT_SCRIPT)
    run_parser.add_argument("-n", "--n", type=int, default=None)
    run_parser.add_argument("-x", "--detach", action="store_true")
    run_parser.add_argument("-m", "--metascript", type=str, default=None)
    run_parser.add_argument("-b", "--metascript_batch", action="store_true")
    run_parser.add_argument("-l", "--metascript_log_dir", type=str)
    run_parser.add_argument("-r", "--ready_status", type=int, default=0)
    run_parser.add_argument("-s", "--started_status", type=int, default=None)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("-d", "--db_path", type=str, help="Experiment database path")
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
