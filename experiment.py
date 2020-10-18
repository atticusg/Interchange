from experiment import ExperimentManager
from datasets.mqnli import MQNLIData
from train import load_model
from modeling.lstm import LSTMModule
import argparse
import torch

EXPT_OPTS = ["data_path", "model_path", "res_save_dir", "abstraction", "num_inputs"]
LAUNCH_SCRIPT = "python expt_interchange.py"
HIGH_NODES = ["sentence_q", "subj_adj", "subj_noun", "neg", "v_adv", "v_verb", "vp_q", "obj_adj", "obj_noun", "obj", "vp", "v_bar", "negp", "subj"]

def setup(db_path, model_path, data_path, res_save_dir):
    default_opts = {
        "data_path": data_path,
        "model_path": model_path,
        "res_save_dir": res_save_dir,
        "abstraction": "",
        "num_inputs": 20
    }
    manager = ExperimentManager(db_path, default_opts, LAUNCH_SCRIPT)

def run(db_path, n):
    manager = ExperimentManager(db_path, EXPT_OPTS, LAUNCH_SCRIPT)
    manager.run(n)

def add(db_path, model_type, model_path, num_inputs):
    if model_type == "lstm":
        manager = ExperimentManager(db_path, EXPT_OPTS, LAUNCH_SCRIPT)
        module, _ = load_model(LSTMModule, model_path, device=torch.device("cpu"))
        num_layers = module.num_lstm_layers
        for high_node in HIGH_NODES:
            for layer in range(num_layers):
                for n in num_inputs:
                    abstraction = f'["{high_node}", ["lstm_{layer}"]]'
                    manager.insert({"abstraction": abstraction,
                                    "num_inputs": n})
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def preprocess(train, dev, test, data_path, no_separator=False, for_transformer=False):
    data = MQNLIData(train, dev, test, for_transformer=for_transformer,
                     use_separator=(not no_separator))
    torch.save(data, data_path)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subparser")

    setup_parser = subparsers.add_parser("setup")
    setup_parser.add_argument("-d", "--db_path", type=str, help="Experiment database path")
    setup_parser.add_argument("-m", "--model_path", type=str, help="Trained torch.nn.module")
    setup_parser.add_argument("-i", "--data_path", type=str, help="Path to pickled dataset")
    setup_parser.add_argument("-o", "--res_save_dir", type=str,help="Directory to save stored results")

    compile_data_parser = subparsers.add_parser("preprocess")
    compile_data_parser.add_argument("train", type=str, help="Train set")
    compile_data_parser.add_argument("dev", type=str, help="Dev set")
    compile_data_parser.add_argument("test", type=str, help="Test set")
    compile_data_parser.add_argument("-o", "--data_path", type=str, help="Destination", required=True)
    compile_data_parser.add_argument("--no_separator", action="store_true")
    compile_data_parser.add_argument("--for_transformer", action="store_true")
    
    add_parser = subparsers.add_parser("add")
    add_parser.add_argument("-d", "--db_path", type=str, required=True)
    add_parser.add_argument("-t", "--model_type", type=str, required=True)
    add_parser.add_argument("-m", "--model_path", type=str, required=True)
    add_parser.add_argument("-n", "--num_inputs", type=int, nargs="+")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("-d", "--db_path", type=str, required=True)
    run_parser.add_argument("-n", "--n", type=int, required=True)

    kwargs = vars(parser.parse_args())
    globals()[kwargs.pop("subparser")](**kwargs)


if __name__ == "__main__":
    main()