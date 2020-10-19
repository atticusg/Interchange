from experiment import ExperimentManager
from datasets.mqnli import MQNLIData
from train import load_model
from modeling.lstm import LSTMModule

from datetime import datetime
import argparse
import torch
import os

EXPT_OPTS = ["data_path", "model_path", "res_save_dir", "abstraction", "num_inputs"]
LAUNCH_SCRIPT = "python expt_interchange.py"
HIGH_NODES = ["sentence_q", "subj_adj", "subj_noun", "neg", "v_adv", "v_verb", "vp_q", "obj_adj", "obj_noun", "obj", "vp", "v_bar", "negp", "subj"]
META_SCRIPT = "nlprun -a hanson-intervention -q john -r 100G"

def setup(db_path, model_path, data_path):
    default_opts = {
        "data_path": data_path,
        "model_path": model_path,
        "log_path": "",
        "res_save_dir": "",
        "abstraction": "",
        "num_inputs": 20
    }
    manager = ExperimentManager(db_path, default_opts, LAUNCH_SCRIPT)

def run(db_path, n):
    manager = ExperimentManager(db_path, EXPT_OPTS, LAUNCH_SCRIPT, META_SCRIPT)
    manager.run(n)

def add(db_path, model_type, model_path, res_dir, num_inputs):
    if model_type == "lstm":
        manager = ExperimentManager(db_path, EXPT_OPTS, LAUNCH_SCRIPT)
        module, _ = load_model(LSTMModule, model_path, device=torch.device("cpu"))
        num_layers = module.num_lstm_layers
        time_str = datetime.now().strftime("%m%d-%H%M%S")
        for high_node in HIGH_NODES:
            for layer in range(num_layers):
                for n in num_inputs:
                    abstraction = f'["{high_node}",["lstm_{layer}"]]'
                    id = manager.insert({"abstraction": abstraction,
                                        "num_inputs": n})
                    res_save_dir = os.path.join(res_dir, f"expt-{id}-{time_str}")
                    manager.update({"res_save_dir": res_save_dir}, id)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def preprocess(train, dev, test, data_path, no_separator=False, for_transformer=False):
    data = MQNLIData(train, dev, test, for_transformer=for_transformer,
                     use_separator=(not no_separator))
    torch.save(data, data_path)


def query(db_path, id=None, status=None, abstraction=None, limit=None):
    manager = ExperimentManager(db_path, EXPT_OPTS, LAUNCH_SCRIPT)
    cols = ["id", "log_path", "res_save_dir", "abstraction", "num_inputs", "status"]
    rows = manager.query(cols=cols, status=status, abstraction=abstraction,
                         id=id, limit=limit)
    if len(rows) == 0:
        return "No data found"
    s = ", ".join(col for col in cols)
    print(s)
    print("-"*len(s))
    for row in rows:
        print(row)
        print("-------")

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")

    setup_parser = subparsers.add_parser("setup")
    setup_parser.add_argument("-d", "--db_path", type=str, help="Experiment database path")
    setup_parser.add_argument("-m", "--model_path", type=str, help="Trained torch.nn.module")
    setup_parser.add_argument("-i", "--data_path", type=str, help="Path to pickled dataset")

    compile_data_parser = subparsers.add_parser("preprocess")
    compile_data_parser.add_argument("train", type=str, help="Train set")
    compile_data_parser.add_argument("dev", type=str, help="Dev set")
    compile_data_parser.add_argument("test", type=str, help="Test set")
    compile_data_parser.add_argument("-o", "--data_path", type=str, help="Destination", required=True)
    compile_data_parser.add_argument("--no_separator", action="store_true")
    compile_data_parser.add_argument("--for_transformer", action="store_true")
    
    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("-d", "--db_path", type=str, required=True, help="Pickled dataset file")
    add_parser.add_argument("-t", "--model_type", type=str, required=True, help="Model type, currently only supports lstm")
    add_parser.add_argument("-m", "--model_path", type=str, required=True, help="Trained torch.nn.module")
    add_parser.add_argument("-o", "--res_dir", type=str, required=True, help="Directory to save stored results")
    add_parser.add_argument("-n", "--num_inputs", type=int, nargs="+")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("-d", "--db_path", type=str, required=True)
    run_parser.add_argument("-n", "--n", type=int, required=True)

    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("-d", "--db_path", type=str, help="Experiment database path")
    query_parser.add_argument("-i", "--id", type=int)
    query_parser.add_argument("-s", "--status", type=int)
    query_parser.add_argument("-a", "--abstraction", type=str)
    query_parser.add_argument("-n", "--limit", type=int)

    kwargs = vars(parser.parse_args())
    globals()[kwargs.pop("subparser")](**kwargs)


if __name__ == "__main__":
    main()
