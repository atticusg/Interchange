import experiment.db_utils as db
import os

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
                 launch_script: str):
        self.db_path = db_path
        self.launch_script = launch_script

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
        db.update(self.db_path, TABLE_NAME, opts)

    def insert_grid(self, opts_dict):
        "insert many experiments"
        pass

    def fetch(self, n=None):
        "Get all experiments that are not yet run from database"
        return db.fetch_new(self.db_path, TABLE_NAME, self.expt_opts, n=n)

    def dispatch(self, opts):
        "launch an experiment by running bash script"
        db.update(self.db_path, TABLE_NAME, {"status": -1}, opts["id"])
        script_args = [self.launch_script]

        for opt_name, value in opts.items():
            script_args.append(f"--{opt_name}")
            script_args.append(repr(value))

        script_args += ["--db_path", f"{self.db_path}"]
        script = " ".join(script_args)
        print("----running script", script)

        os.system(script)

    def run(self, n=None):
        expts = self.fetch(n)
        for expt_opts in expts:
            self.dispatch(expt_opts)
