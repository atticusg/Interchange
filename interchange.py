import os
import argparse
from datetime import datetime
from experiment import ExperimentManager

DEFAULT_SCRIPT = "python expt_interchange.py"
HIGH_NODES = ["sentence_q", "subj_adj", "subj_noun", "neg", "v_adv", "v_verb",
              "vp_q", "obj_adj", "obj_noun", "obj", "vp", "v_bar", "negp",
              "subj"]
META_SCRIPT = "nlprun -a hanson-intervention -q john -r 100G"

INTERCHANGE_DEFAULT_OPTS = {
    "data_path": "",
    "model_path": "",
    "model_type": "",
    "log_path": "",
    "res_save_dir": "",
    "abstraction": "",
    "num_inputs": 500,
    "graph_alpha": 100,
    "interchange_batch_size": 128,
    "loc_mapping_type": "",
}


def setup(db_path, model_path, data_path):
    opts = INTERCHANGE_DEFAULT_OPTS.copy()
    opts["data_path"] = data_path,
    opts["model_path"] = model_path
    ExperimentManager(db_path, INTERCHANGE_DEFAULT_OPTS)


def add(db_path, model_type, model_path, res_dir, num_inputs, loc_mapping_type):
    import torch
    from trainer import load_model
    from modeling import get_module_class_by_name
    import experiment.db_utils as db

    model_class = get_module_class_by_name(model_type)

    manager = ExperimentManager(db_path)
    device = torch.device("cpu")
    module, _ = load_model(model_class, model_path, device=device)

    if model_type == "lstm":
        num_layers = module.num_lstm_layers - 1
        layer_name = "lstm"
        graph_alpha = 0
    elif model_type == "bert":
        num_layers = len(module.bert.encoder.layer) - 1
        layer_name = "bert_layer"
        graph_alpha = 100
        if loc_mapping_type:
            db.add_cols(db_path, "results", {"loc_mapping_type": ""})
    else:
        raise ValueError("Invalid model type")

    time_str = datetime.now().strftime("%m%d-%H%M%S")
    for high_node in HIGH_NODES:
        for layer in range(num_layers):
            for n in num_inputs:
                abstraction = f'["{high_node}",["{layer_name}_{layer}"]]'
                insert_dict = {"abstraction": abstraction,
                               "num_inputs": n,
                               "model_type": model_type,
                               "interchange_batch_size": 100,
                               "graph_alpha": graph_alpha}
                if loc_mapping_type:
                    insert_dict["loc_mapping_type"] = loc_mapping_type
                id = manager.insert(insert_dict)
                res_save_dir = os.path.join(res_dir, f"expt-{id}-{time_str}")
                manager.update({"model_path": model_path,
                                "res_save_dir": res_save_dir}, id)


def run(db_path, script, n, detach, metascript, metascript_batch_size,
        metascript_log_dir,
        ready_status, started_status):
    manager = ExperimentManager(db_path, INTERCHANGE_DEFAULT_OPTS)

    if os.path.exists(script):
        with open(script, "r") as f:
            script = f.read().strip()

    if metascript and os.path.exists(metascript):
        with open(metascript, "r") as f:
            metascript = f.read().strip()

    manager.run(launch_script=script, n=n, detach=detach,
                metascript=metascript,
                metascript_batch_size=metascript_batch_size,
                metascript_log_dir=metascript_log_dir,
                ready_status=ready_status, started_status=started_status)


def analyze(db_path, script, n, detach, metascript,
            metascript_batch_size, metascript_log_dir,
            ready_status, started_status):
    # expt_opts = ["data_path", "model_path", "save_path", "abstraction",
    #              "num_inputs", "res_save_dir"]
    expt_opts = ["save_path", "res_save_dir"]
    manager = ExperimentManager(db_path, expt_opts)

    if metascript and os.path.exists(metascript):
        with open(metascript, "r") as f:
            metascript = f.read().strip()

    manager.run(launch_script=script, n=n, detach=detach,
                metascript=metascript,
                metascript_batch_size=metascript_batch_size,
                metascript_log_dir=metascript_log_dir,
                ready_status=ready_status, started_status=started_status)


def add_graph(db_path, ids, alphas, all):
    from experiment import db_utils as db

    db.add_cols(db_path, "results", {"graph_alpha": 1})
    if all:
        assert len(alphas) == 1
        rows = db.select(db_path, "results", cols=["id"],
                         cond_dict={"status": 2})
        for row in rows:
            id = row["id"]
            db.update(db_path, "results",
                      {"graph_alpha": alphas[0], "status": 3}, id=id)

    elif ids:
        for id in ids:
            dup_ids = db.duplicate_rows(db_path, "results", id, len(alphas))
            for i, a in zip(dup_ids, alphas):
                db.update(db_path, "results", {"graph_alpha": a, "status": 3},
                          id=i)


def analyze_graph(db_path, script, n, detach, metascript, ready_status,
                  started_status):
    expt_opts = ["save_path", "graph_alpha", "res_save_dir"]
    manager = ExperimentManager(db_path, expt_opts)

    if metascript and os.path.exists(metascript):
        with open(metascript, "r") as f:
            metascript = f.read().strip()

    manager.run(launch_script=script, n=n, detach=detach,
                metascript=metascript, ready_status=ready_status,
                started_status=started_status)


def query(db_path, id=None, status=None, abstraction=None, limit=None):
    manager = ExperimentManager(db_path)
    cols = ["id", "log_path", "res_save_dir", "abstraction", "num_inputs",
            "status"]
    rows = manager.query(cols=cols, status=status, abstraction=abstraction,
                         id=id, limit=limit)
    if len(rows) == 0:
        return "No data found"
    s = ", ".join(col for col in cols)
    print(s)
    print("-" * len(s))
    for row in rows:
        print(row)
        print("-------")


def update_status(db_path, ids, id_range, status):
    from experiment import db_utils as db
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
                              help="Trained torch.nn.module")
    setup_parser.add_argument("-i", "--data_path", type=str,
                              help="Path to pickled dataset")

    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("-d", "--db_path", type=str, required=True,
                            help="Pickled dataset file")
    add_parser.add_argument("-t", "--model_type", type=str, required=True,
                            help="Model type, currently only supports lstm")
    add_parser.add_argument("-m", "--model_path", type=str, required=True,
                            help="Trained torch.nn.module")
    add_parser.add_argument("-o", "--res_dir", type=str, required=True,
                            help="Directory to save stored results")
    add_parser.add_argument("-n", "--num_inputs", type=int, nargs="+")
    add_parser.add_argument("-l", "--loc_mapping_type", type=str, default="",
                            help="Specify the set of hidden vector locations for intervention")

    add_graph_parser = subparsers.add_parser("add_graph")
    add_graph_parser.add_argument("-d", "--db_path", type=str, required=True,
                                  help="Pickled dataset file")
    add_graph_parser.add_argument("-i", "--ids", type=int, nargs="*")
    add_graph_parser.add_argument("-a", "--alphas", type=int, nargs="+")
    add_graph_parser.add_argument("--all", action="store_true")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("-d", "--db_path", type=str, required=True)
    run_parser.add_argument("-i", "--script", type=str, default=DEFAULT_SCRIPT)
    run_parser.add_argument("-n", "--n", type=int, default=None)
    run_parser.add_argument("-x", "--detach", action="store_true")
    run_parser.add_argument("-m", "--metascript", type=str, default=None)
    run_parser.add_argument("-b", "--metascript_batch_size", type=int,
                            default=0)
    run_parser.add_argument("-l", "--metascript_log_dir", type=str)
    run_parser.add_argument("-r", "--ready_status", type=int, default=0)
    run_parser.add_argument("-s", "--started_status", type=int, default=None)

    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("-d", "--db_path", type=str, required=True)
    analyze_parser.add_argument("-i", "--script", type=str,
                                default="python expt_interchange_analysis.py")
    analyze_parser.add_argument("-n", "--n", type=int, required=True)
    analyze_parser.add_argument("-x", "--detach", action="store_true")
    analyze_parser.add_argument("-m", "--metascript", type=str, default=None)
    analyze_parser.add_argument("-b", "--metascript_batch_size", type=int,
                                default=0)
    analyze_parser.add_argument("-l", "--metascript_log_dir", type=str)
    analyze_parser.add_argument("-r", "--ready_status", type=int, default=1)
    analyze_parser.add_argument("-s", "--started_status", type=int,
                                default=None)

    analyze_graph_parser = subparsers.add_parser("analyze_graph")
    analyze_graph_parser.add_argument("-d", "--db_path", type=str,
                                      required=True)
    analyze_graph_parser.add_argument("-i", "--script", type=str,
                                      default="python expt_viz_cliques.py")
    analyze_graph_parser.add_argument("-n", "--n", type=int, required=True)
    analyze_graph_parser.add_argument("-x", "--detach", action="store_true")
    analyze_graph_parser.add_argument("-m", "--metascript", type=str,
                                      default=None)
    analyze_graph_parser.add_argument("-r", "--ready_status", type=int,
                                      default=0)
    analyze_graph_parser.add_argument("-s", "--started_status", type=int,
                                      default=None)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("-d", "--db_path", type=str,
                              help="Experiment database path")
    query_parser.add_argument("-i", "--id", type=int)
    query_parser.add_argument("-s", "--status", type=int)
    query_parser.add_argument("-a", "--abstraction", type=str)
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