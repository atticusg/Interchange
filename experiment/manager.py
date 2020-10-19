import experiment.db_utils as db
import os

from datetime import datetime
from typing import Dict, List, Union

TABLE_NAME = "results"


class Experiment:
    def experiment(self, opts: Dict):
        raise NotImplementedError

    def run(self, opts: Dict) -> Dict:
        res_dict = self.experiment(opts)
        if "db_path" in opts and opts["db_path"]:
            self.record_results(res_dict, opts["db_path"], opts["id"])
        return res_dict

    def record_results(self, res_dict: Dict, db_path: str, xid: int):
        complete_dict = res_dict.copy()
        complete_dict["status"] = 1
        db.add_cols(db_path, TABLE_NAME, complete_dict)
        db.update(db_path, TABLE_NAME, complete_dict, xid)


class ExperimentManager:
    def __init__(self, db_path: str, expt_opts: Union[Dict, List],
                 launch_script: str, metascript: str=None):
        self.db_path = db_path
        self.launch_script = launch_script
        self.metascript = metascript

        if not os.path.exists(db_path):
            print("Creating new database for experiment manager")
            assert isinstance(expt_opts, Dict)
            self.expt_opts = list(expt_opts.keys())
            opts = expt_opts.copy()
            if "status" not in opts:
                opts["status"] = 0
            db.create_table(db_path, TABLE_NAME, opts)
        else:
            print("Using existing database for experiment manager")
            if isinstance(expt_opts, Dict):
                self.expt_opts = list(expt_opts.keys())
            elif isinstance(expt_opts, List):
                self.expt_opts = expt_opts
            else:
                raise ValueError

    def insert(self, opts):
        "insert a new experiment"
        opts["status"] = 0
        return db.update(self.db_path, TABLE_NAME, opts)

    def update(self, opts, id):
        return db.update(self.db_path, TABLE_NAME, opts, id=id)

    def insert_grid(self, opts_dict):
        "insert many experiments"
        pass

    def fetch(self, n=None):
        "Get all experiments that are not yet run from database"
        return db.fetch_new(self.db_path, TABLE_NAME, self.expt_opts, n=n)

    def query(self, cols=None, status=None, abstraction=None, id=None, limit=None):
        cond_dict = {}
        if status is not None: cond_dict["status"] = status
        if id: cond_dict["id"] = id
        like_dict = {}
        if abstraction:
            like_dict["abstraction"] = f"%{abstraction}%"
        return db.select(self.db_path, TABLE_NAME, cols=cols,
                         cond_dict=cond_dict, like=like_dict, limit=limit)


    def dispatch(self, opts):
        "launch an experiment by running bash script"
        update_dict = {"status": -1}

        if self.metascript:
            metascript = self.metascript
            if metascript.startswith("nlprun"):
                assert "-o" not in metascript
                time_str = datetime.now().strftime("%m%d-%H%M%S")
                log_path = os.path.join(opts["res_save_dir"], f"{time_str}.log")
                metascript += f" -o {log_path}"
                update_dict["log_path"] = log_path

        db.update(self.db_path, TABLE_NAME, update_dict, opts["id"])

        script_args = [self.launch_script]

        for opt_name, value in opts.items():
            script_args.append(f"--{opt_name}")
            script_args.append(repr(value))

        script_args += ["--db_path", f"{self.db_path}"]
        script = " ".join(script_args)

        if self.metascript:
            script = metascript + f'"{script}"'

        print("----running:\n", script)
        os.system(script)

    def run(self, n=None):
        expts = self.fetch(n)
        for expt_opts in expts:
            self.dispatch(expt_opts)
