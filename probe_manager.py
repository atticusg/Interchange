import os
import argparse

from experiment import db_utils as db
from experiment import ExperimentManager

DEFAULT_SCRIPT = "python probe.py"

DEFAULT_PROBING_OPTS = {
    "model_path": "",
    "data_path": "",
    "model_type": "",

    "probe_max_rank": 24,
    "probe_dropout": 0.1,
    "probe_train_num_examples": 12800,
    "probe_train_num_dev_examples": 2000,
    "probe_correct_examples_only": True,

    "probe_train_batch_size": 512,
    "probe_train_eval_batch_size": 1024,
    "probe_train_weight_norm": 0.,
    "probe_train_max_epochs": 40,
    "probe_train_early_stopping_epochs": 4,
    "probe_train_lr": 0.001,
    "probe_train_lr_patience_epochs": 0,
    "probe_train_lr_anneal_factor": 0.5,
    "res_save_dir": "",
    "res_save_path": "",
    "log_path": ""
}

def setup(db_path, model_path, data_path):
    default_opts = DEFAULT_PROBING_OPTS.copy()
    default_opts["data_path"] = data_path
    default_opts["model_path"] = model_path
    default_opts["model_type"] = "bert" if "bert" in model_path else "lstm"
    ExperimentManager(db_path, default_opts)

def add_grid_search(db_path, res_save_dir):
    from datetime import datetime
    from itertools import product

    manager = ExperimentManager(db_path)

    grid_dict = {
        "probe_max_rank": [8, 32],
        "probe_train_lr": [0.001, 0.01],
        "probe_dropout": [0.1],
        "probe_train_weight_norm": [0.01, 0.1],
    }

    var_opt_names = list(grid_dict.keys())
    var_opt_values = list(v if isinstance(v, list) else list(v) \
                          for v in grid_dict.values())

    # treat elements in list as separate args to fxn
    for tup in product(*var_opt_values):
        update_dict = {}
        for name, val in zip(var_opt_names, tup):
            update_dict[name] = val

        id = manager.insert(update_dict)
        time_str = datetime.now().strftime("%m%d-%H%M%S")
        curr_save_dir = os.path.join(res_save_dir, f"probing-{id}-{time_str}")
        manager.update({"res_save_dir": curr_save_dir}, id)
        print("----inserted example into database:", update_dict)


def run(db_path, script, n, detach, metascript, ready_status, started_status):
    expt_opts = list(DEFAULT_PROBING_OPTS.keys())

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
    cols = ["id", "status", "probe_max_rank", "probe_train_num_examples", "res_save_dir"]
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

    setup_parser = subparsers.add_parser("setup")
    setup_parser.add_argument("-d", "--db_path", type=str,
                              help="Experiment database path")
    setup_parser.add_argument("-m", "--model_path", type=str,
                              help="Path to trained model")
    setup_parser.add_argument("-i", "--data_path", type=str,
                              help="Path to pickled dataset")


    add_gs_parser = subparsers.add_parser("add_grid_search")
    add_gs_parser.add_argument("-d", "--db_path", type=str, required=True,
                               help="Experiment database path")
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
