import os
import json
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

# [ <CLS> | not | every | bad  | singer | does | not | badly | sings | <e> | every | good | song  | <SEP> | ]
#  0      | 1   | 2     | 3    | 4      | 5    | 6   | 7     | 8     | 9   | 10    | 11   | 12    | 13    | -- BERT premise
#         | 14  | 15    | 16   | 17     | 18   | 19  | 20    | 21    | 22  | 23    | 24   | 25    | 26    | -- BERT hypothesis
# -----------------------------------------------------------------------------------------------------------
#         | sentence_q  | subj | subj   | neg        | v     | v     | vp_q        | obj  | obj   |
#         |             | _adj | _noun  |            | _adv  | _verb |             | _adj | _noun |
# -----------------------------------------------------------------------------------------------------------
#                       | ---- subj --- |            | --- v_bar --- |             | --- obj ---  |
#                                                    |    ---------------- vp -----------------   |
#                                       |   ------------------- negp -------------------------    |

# high_node_to_loc = {
#     "sentence_q": [0, 2],
#     "subj_adj": [0, 3],
#     "subj_noun": [0, 4],
#     "neg": [0, 6],
#     "v_adv": [0, 7],
#     "v_verb": [0, 8],
#     "vp_q": [0, 9, 10],
#     "obj_adj": [0, 11],
#     "obj_noun": [0, 12],
#     "obj": [0, 11, 12],
#     "vp": [0, 8, 10],
#     "v_bar": [0, 7, 8],
#     "negp": [0, 5, 6],
#     "subj": [0, 3, 4]
# }

high_node_to_loc = {
    "sentence_q": [0, 2, 15],
    "subj_adj": [0, 3, 16],
    "subj_noun": [0, 4, 17],
    "neg": [0, 6, 19],
    "v_adv": [0, 7, 20],
    "v_verb": [0, 8, 21],
    "vp_q": [0, 10, 23],
    "obj_adj": [0, 11, 24],
    "obj_noun": [0, 12, 25],
    "obj": [0, 12, 25],
    "vp": [0, 8, 10, 21, 23],
    "v_bar": [0, 8, 21],
    "negp": [0, 6, 19],
    "subj": [0, 4, 17]
}

cf_high_node_list = ["obj_noun", "obj_adj", "obj", "vp_q", "vp", "neg", "negp"]
bert_layer_idxs = [0, 2, 4, 6, 8, 10]

def preprocess(model_type, train, dev, test, data_path, variant):
    import torch
    from datasets.mqnli import MQNLIBertData, MQNLIData

    if model_type == "bert":
        data = MQNLIBertData(train, dev, test, REMAPPING_PATH, variant=variant)
    elif model_type == "lstm":
        data = MQNLIData(train, dev, test, variant=variant)
    else:
        raise NotImplementedError(f"Does not support model type {model_type}")
    torch.save(data, data_path)
    print(f"Saved preprocessed dataset to {data_path}")

def setup(experiment, data_path):
    from dataclasses import asdict
    from counterfactual import CounterfactualTrainingConfig

    db_path = os.path.join("data/cf-training/", f"{experiment}.db")

    default_opts = asdict(CounterfactualTrainingConfig())
    # if "bert" in db_path:
    #     default_opts = asdict(CounterfactualTrainingConfig())
    # elif "lstm" in db_path:
    #     default_opts = DEFAULT_LSTM_OPTS.copy()
    # else:
    #     raise ValueError(f"Cannot infer model type from database path {db_path}")

    default_opts["data_path"] = data_path
    # if "hard" in data_path or "medium" in data_path:
    #     default_opts["output_classes"] = 10
    # if "lstm" in db_path:
    #     if "bert" in data_path:
    #         default_opts["tokenizer_vocab_path"] = VOCAB_PATH
    #     else:
    #         data = torch.load(data_path)
    #         default_opts["vocab_size"] = data.vocab_size

    ExperimentManager(db_path, default_opts)

def add_one(db_path):
    manager = ExperimentManager(db_path)
    manager.insert({"lr": 5e-5, "max_epochs": 200, 'patient_epochs': 30})


def get_name_from_mapping(mapping_str):
    mapping = json.loads(mapping_str)
    high_node = list(mapping.keys())[0]
    low_node_to_loc = mapping[high_node]
    low_node = list(low_node_to_loc.keys())[0]
    low_layer = int(low_node[-1])
    loc_str = low_node_to_loc[low_node]
    loc = int(loc_str.split(",")[1])

    return f"{high_node}-layer_{low_layer}-loc_{loc}"

# TODO: test grid search
def add_grid_search(experiment, repeat):
    from datetime import datetime
    from itertools import product

    db_path = os.path.join("data/cf-training/", f"{experiment}.db")
    res_save_dir = os.path.join("data/cf-training", experiment)

    manager = ExperimentManager(db_path)
    base_dict = {
        "lr": 5e-5,
        "lr_scheduler_type": "linear",
        "scheduler_warmup_subepochs": 10,
        "eval_subepochs": 5,
        "max_subepochs": 80,
        "model_save_path": "",
        "interx_after_train": 1,
        "interx_save_results": 1
    }

    grid_dict = {
        k: [v] for k, v in base_dict.items()
    }

    if experiment == "ablation":
        grid_dict.update({
            "cf_type": ["augmented", "random_only"],
            "train_multitask_scheduler_type": ["fixed", "linear"],
            "mapping": ['{"vp": {"bert_layer_3": ":,10,:"}}'],
        })
    elif experiment == "impactful" or experiment == "impactful2":
        grid_dict.update({
            "cf_type": ["impactful"],
            "train_multitask_scheduler_type": ["fixed"],
            "mapping": ['{"vp": {"bert_layer_3": ":,10,:"}}'],
            "cf_impactful_ratio": [0.2, 0.5, 0.8]
        })
    elif experiment == "impactful_sanity_check":
        grid_dict.update({
            "cf_type": ["impactful"],
            "train_multitask_scheduler_type": ["fixed"],
            "mapping": ['{"vp": {"bert_layer_3": ":,10,:"}}'],
            "cf_impactful_ratio": [0.2, 0.5, 0.8],
            "interx_num_cf_training_pairs": [100000]
        })
    elif experiment == "cf_ratio":
        grid_dict.update({
            "mapping": ['{"vp": {"bert_layer_3": ":,10,:"}}'],
            "base_to_cf_ratio": [0.01, 0.1, 0.5, 1.0]
        })
    elif experiment == "cf_grid":
        mappings = []
        for high_node in cf_high_node_list:
            for loc in high_node_to_loc[high_node]:
                for layer_idx in bert_layer_idxs:
                    mappings.append(f'{{"{high_node}": {{"bert_layer_{layer_idx}": ":,{loc},:"}}}}')

        grid_dict.update({
            "train_multitask_scheduler_type": ["fixed"],
            "cf_type": ["random_only"],
            "mapping": mappings,
            "cf_impactful_ratio": [0.5],
        })
    elif experiment == "aug_grid":
        pass
    else:
        raise ValueError(f"Invalid experiment name")

    var_opt_names = list(grid_dict.keys())
    var_opt_values = list(v if isinstance(v, list) else list(v) \
                          for v in grid_dict.values())

    update_dicts = []
    for tup in product(*var_opt_values):
        update_dict = {}
        for name, val in zip(var_opt_names, tup):
            update_dict[name] = val
        update_dicts.append(update_dict)

    # add other experiments that are not part of the grid search to update_dicts
    if experiment.startswith("impactful"):
        new_expt = base_dict.copy()
        new_expt.update({
            "cf_type": "random_only",
            "train_multitask_scheduler_type": "fixed",
            "mapping": '{"vp": {"bert_layer_3": ":,10,:"}}',
            "model_save_path": "",
            "interx_after_train": 1,
            "interx_save_results": 1,
            "cf_impactful_ratio": 0.5
        })
        update_dicts.append(new_expt)

    # treat elements in list as separate args to fxn
    for _ in range(repeat):
        for update_dict in update_dicts:
            id = manager.insert(update_dict)
            # time_str = datetime.now().strftime("%m%d-%H%M%S")
            save_dir_name = f"expt-{id}"
            if experiment == "ablation":
                cf_type = update_dict["cf_type"]
                scheduler_type = update_dict["train_multitask_scheduler_type"]
                save_dir_name += f"-c_{cf_type}-s_{scheduler_type}"
            elif experiment == "cf_grid":
                save_dir_name += f"-{get_name_from_mapping(update_dict['mapping'])}"

            curr_save_dir = os.path.join(res_save_dir, save_dir_name)
            manager.update({"res_save_dir": curr_save_dir}, id)
            print("----inserted example into database:", update_dict)


def run(experiment, script, n, detach, metascript, ready_status, started_status):
    from dataclasses import asdict
    from counterfactual import CounterfactualTrainingConfig

    expt_opts = list(asdict(CounterfactualTrainingConfig()).keys())
    db_path = os.path.join("data/cf-training/", f"{experiment}.db")
    # if "lstm" in db_path:
    #     expt_opts = list(DEFAULT_LSTM_OPTS.keys())
    # elif "bert" in db_path:
    #     expt_opts = list(asdict(CounterfactualTrainingConfig()).keys())
    # else:
    #     raise ValueError(f"Cannot infer model type from database path {db_path}")

    manager = ExperimentManager(db_path, expt_opts)

    if os.path.exists(script):
        with open(script, "r") as f:
            script = f.read().strip()

    if metascript and os.path.exists(metascript):
        with open(metascript, "r") as f:
            metascript = f.read().strip()

    manager.run(launch_script=script, n=n, detach=detach, metascript=metascript,
                ready_status=ready_status, started_status=started_status)


def query(experiment, id=None, status=None, limit=None):
    db_path = os.path.join("data/cf-training/", f"{experiment}.db")
    manager = ExperimentManager(db_path)
    cols = ["id", "status", "res_save_dir", "model_save_path"]
    rows = manager.query(cols=cols, status=status, id=id, limit=limit)
    if len(rows) == 0:
        return "No data found"
    s = ", ".join(col for col in cols)
    print(s)
    print("-"*len(s))
    for row in rows:
        print(row)
        print("-------")

def update_status(experiment, ids, id_range, status):
    db_path = os.path.join("data/cf-training/", f"{experiment}.db")
    if id_range:
        ids = list(range(id_range[0], id_range[1] + 1))
    for i in ids:
        db.update(db_path, "results", {"status": status}, id=i)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")

    # preprocess_parser = subparsers.add_parser("preprocess")
    # preprocess_parser.add_argument("model_type", type=str, help="type of model")
    # preprocess_parser.add_argument("train", type=str, help="Train set")
    # preprocess_parser.add_argument("dev", type=str, help="Dev set")
    # preprocess_parser.add_argument("test", type=str, help="Test set")
    # preprocess_parser.add_argument("-o", "--data_path", type=str,
    #                                help="Destination", required=True)
    # preprocess_parser.add_argument("-v", "--variant", type=str, default="basic",
    #                                help="Type of bert realignment variant")

    setup_parser = subparsers.add_parser("setup")
    setup_parser.add_argument("-e", "--experiment", type=str,
                              help="name of the experiment")
    setup_parser.add_argument("-i", "--data_path", type=str,
                              default="data/mqnli/preprocessed/bert-hard_abl.pt",
                              help="Path to pickled dataset")

    add_gs_parser = subparsers.add_parser("add_grid_search")
    add_gs_parser.add_argument("-e", "--experiment", type=str, default="ablation",
                               help="name of the experiment")
    add_gs_parser.add_argument("-r", "--repeat", type=int, default=1,
                               help="Repeat each grid search config for number "
                                    "of times")



    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("-e", "--experiment", type=str, required=True)
    run_parser.add_argument("-i", "--script", type=str, default=DEFAULT_SCRIPT)
    run_parser.add_argument("-n", "--n", type=int, default=None)
    run_parser.add_argument("-x", "--detach", action="store_true")
    run_parser.add_argument("-m", "--metascript", type=str, default=None)
    run_parser.add_argument("-r", "--ready_status", type=int, default=0)
    run_parser.add_argument("-s", "--started_status", type=int, default=None)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("-e", "--experiment", type=str, required=True)
    query_parser.add_argument("-i", "--id", type=int)
    query_parser.add_argument("-s", "--status", type=int)
    query_parser.add_argument("-n", "--limit", type=int)

    update_parser = subparsers.add_parser("update_status")
    update_parser.add_argument("-e", "--experiment", type=str, required=True)
    update_parser.add_argument("-i", "--ids", type=int, nargs="*")
    update_parser.add_argument("-r", "--id_range", type=int, nargs=2)
    update_parser.add_argument("-s", "--status", type=int, required=True)

    kwargs = vars(parser.parse_args())
    globals()[kwargs.pop("subparser")](**kwargs)


if __name__ == "__main__":
    main()
