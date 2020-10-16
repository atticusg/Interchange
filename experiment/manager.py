import experiment.db_utils as db
import os

TABLE_NAME = "results"


class Experiment:
    def experiment(self, opts: dict):
        raise NotImplementedError

    def run(self, opts: dict) -> dict:
        res_dict = self.experiment(opts)
        if "db_path" in opts and opts["db_path"]:
            self.record_results(res_dict, opts["db_path"], opts["id"])
        return res_dict

    def record_results(self, res_dict: dict, db_path: str, xid: int):
        complete_dict = res_dict.copy()
        complete_dict["status"] = 1
        db.add_cols(db_path, TABLE_NAME, complete_dict)
        db.insert(db_path, TABLE_NAME, complete_dict, xid)


class ExperimentManager:
    def __init__(self, db_path: str, default_opts: dict, launch_script: str):
        self.db_path = db_path
        self.launch_script = launch_script
        self.default_opts = default_opts
        opts = default_opts.copy()
        if "status" not in opts:
            opts["status"] = 0
        db.create_table(db_path, TABLE_NAME, opts)

    def insert(self, opts):
        "insert a new experiment"
        opts["status"] = 0
        db.insert(self.db_path, TABLE_NAME, opts)

    def insert_grid(self, opts_dict):
        "insert many experiments"
        pass

    def fetch(self, n=None):
        "Get all experiments that are not yet run from database"
        return db.fetch_new(self.db_path, TABLE_NAME,
                            list(self.default_opts.keys()), n=n)

    def dispatch(self, opts):
        "launch an experiment by running bash script"
        script = self.launch_script + " "
        script += " ".join(f"--{opt_name} {value}"
                           for opt_name, value in opts.items())
        script += f" --db_path {self.db_path}"
        print("running script", script)
        os.system(script)

    def run(self, n=None):
        expts = self.fetch(n)
        for expt_opts in expts:
            self.dispatch(expt_opts)
