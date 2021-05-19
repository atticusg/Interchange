from dataclasses import is_dataclass, asdict

import experiment.db_utils as db
import os
import stat
import shlex
import subprocess
import argparse

from datetime import datetime
from typing import Dict, List, Union, Optional

TABLE_NAME = "results"


class Experiment:
    def __init__(self, finished_status=1):
        self.finished_status = finished_status

    def experiment(self, opts: Dict):
        raise NotImplementedError

    def run(self, opts: Dict) -> Dict:
        res_dict = self.experiment(opts)

        if "db_path" in opts and opts["db_path"]:
            self.record_results(res_dict, opts["db_path"], opts["id"])
        return res_dict

    def record_results(self, res_dict: Dict, db_path: str, xid: int):
        complete_dict = res_dict.copy()
        if self.finished_status is not None:
            complete_dict["status"] = self.finished_status
        db.add_cols(db_path, TABLE_NAME, complete_dict)
        db.update(db_path, TABLE_NAME, complete_dict, xid)


class ExperimentManager:
    def __init__(self, db_path: str, expt_opts: Optional[Union[Dict, List]]=None):
        self.db_path = db_path

        if not os.path.exists(db_path):
            if not expt_opts:
                raise ValueError("Must provide experiment opts when initializing experiment!")
            print("Creating new database for experiment manager")
            assert isinstance(expt_opts, dict)
            self.expt_opts = list(expt_opts.keys())
            opts = expt_opts.copy()
            if "status" not in opts:
                opts["status"] = 0
            db.create_table(db_path, TABLE_NAME, opts)
        else:
            print("Using existing database for experiment manager")
            # if isinstance(expt_opts, Dict):
            #     self.expt_opts = list(expt_opts.keys())
            # elif isinstance(expt_opts, List):
            #     self.expt_opts = expt_opts
            # else:
            #     raise ValueError
            if expt_opts:
                if isinstance(expt_opts, dict): expt_opts = list(expt_opts.keys())
                self.expt_opts = expt_opts
            else:
                self.expt_opts = list(db.get_col_names(db_path, TABLE_NAME))

    def insert(self, opts):
        "insert a new experiment"
        opts["status"] = 0
        return db.update(self.db_path, TABLE_NAME, opts)

    def update(self, opts, id):
        return db.update(self.db_path, TABLE_NAME, opts, id=id)

    def insert_grid(self, opts_dict):
        "insert many experiments"
        pass

    def fetch(self, n=None, status=0):
        "Get all experiments that are not yet run from database"
        return db.fetch_new(self.db_path, TABLE_NAME, self.expt_opts, n=n,
                            status=status)

    def query(self, cols=None, status=None, abstraction=None, id=None, limit=None):
        cond_dict = {}
        if status is not None: cond_dict["status"] = status
        if id: cond_dict["id"] = id
        like_dict = {}
        if abstraction:
            like_dict["abstraction"] = f"%{abstraction}%"
        return db.select(self.db_path, TABLE_NAME, cols=cols,
                         cond_dict=cond_dict, like=like_dict, limit=limit)

    def get_script(self, opts, launch_script):
        script_args = [launch_script]

        for opt_name, value in opts.items():
            script_args.append(f"--{opt_name}")
            script_args.append(repr(value))

        script_args += ["--db_path", f"{self.db_path}"]
        script = " ".join(script_args)
        return script


    def dispatch(self, opts, launch_script, metascript=None, detach=False,
                 started_status=None):
        "launch an experiment by running bash script"
        update_dict = {}
        if started_status is not None:
            update_dict["status"] = started_status
        if metascript:
            # need result save dir to exist for logging purposes
            save_dir = opts["res_save_dir"]
            if not os.path.exists(save_dir):
                print("Creating directory to save results:", save_dir)
                os.makedirs(save_dir)
            time_str = datetime.now().strftime("%m%d-%H%M%S")

            if metascript.startswith("nlprun"):
                # manage logging output for nlprun
                assert "-o" not in metascript
                log_path = os.path.join(save_dir, f"{time_str}.log")
                metascript += f" -o {log_path} "
                update_dict["log_path"] = log_path

        if update_dict:
            db.update(self.db_path, TABLE_NAME, update_dict, opts["id"])

        # generate script to launch task
        script = self.get_script(opts, launch_script)

        if metascript:
            # save script to a file, and use this script file for metascript
            script_file_path = os.path.join(save_dir, f"script_{time_str}.sh")
            with open(script_file_path, "w") as f:
                f.write(script)

            # make executable
            st = os.stat(script_file_path)
            os.chmod(script_file_path, st.st_mode | stat.S_IXUSR)

            cmds = shlex.split(metascript)
            cmds.append(script_file_path)
            print(f"----running:\n{cmds}")
            print(f"----script:\n{script}")
            if detach:
                subprocess.Popen(cmds, start_new_session=detach)
            else:
                subprocess.run(cmds)
        else:
            cmds = shlex.split(script)
            print("----running:\n", cmds)
            if detach:
                subprocess.Popen(cmds, start_new_session=detach)
            else:
                subprocess.run(cmds)

    def run(self, launch_script, n=None, detach=False, metascript=None,
            metascript_batch_size=0, metascript_log_dir=None, ready_status=0,
            started_status=None):
        if metascript and metascript_batch_size:
            self.run_metascript_batch(launch_script, metascript, metascript_log_dir,
                                      metascript_batch_size,
                                      n=n, detach=detach, ready_status=ready_status,
                                      started_status=started_status)
        else:
            expts = self.fetch(n, status=ready_status)
            for expt_opts in expts:
                self.dispatch(expt_opts, launch_script=launch_script,
                              metascript=metascript, detach=detach,
                              started_status=started_status)



    def run_metascript_batch(self, launch_script, metascript, log_dir, batch_size, n=None,
                             detach=False, ready_status=1, started_status=None):
        """ Put a batch of scripts into one script file and run the whole batch
        with a given metascript """

        all_expts = self.fetch(n, status=ready_status)
        time_str = datetime.now().strftime("%m%d-%H%M%S")

        for i in range(0, len(all_expts), batch_size):
            expts = all_expts[i:i+batch_size]

            if not os.path.exists(log_dir):
                print("Creating directory to save results:", log_dir)
                os.makedirs(log_dir)

            assert "-o" not in metascript

            curr_log_path = os.path.join(log_dir, f"out-{time_str}-batch{i}.log")
            curr_metascript = metascript + f" -o {curr_log_path} "

            scripts = []
            for opts in expts:
                scripts.append(self.get_script(opts, launch_script))
                if started_status is not None:
                    update_dict = {"status": started_status}
                    db.update(self.db_path, TABLE_NAME, update_dict, opts["id"])


            script_file_path = os.path.join(log_dir, f"script-{time_str}-batch{i}.sh")

            with open(script_file_path, "w") as f:
                for script in scripts:
                    f.write(script + "\n")

            # make executable
            st = os.stat(script_file_path)
            os.chmod(script_file_path, st.st_mode | stat.S_IXUSR)

            cmds = shlex.split(curr_metascript)
            cmds.append(script_file_path)
            print(f"----running:\n{cmds}")

            if detach:
                subprocess.Popen(cmds, start_new_session=detach)
            else:
                subprocess.run(cmds)
        # subprocess.Popen(cmds, start_new_session=True)

def recover_boolean_args(opts: Dict, reference: Dict):
    """ Boolean arguments in opts are cast to integers (0, 1) by the sqlite
    database systems. Recover those boolean arguments given a reference."""

    for k in opts.keys():
        if k in reference and isinstance(reference[k], bool):
            opts[k] = bool(opts[k])

def parse_args(parser: argparse.ArgumentParser, default_opts):
    if is_dataclass(default_opts):
        default_dict = asdict(default_opts)
    else:
        assert isinstance(default_opts, dict)
        default_dict = default_opts

    for arg_name, default_val in default_dict.items():
        if is_dataclass(default_opts):
            arg_type = default_opts.__annotations__[arg_name]
        else:
            # determine type by default value
            if default_val is None:
                default_val = ""
            arg_type = type(default_val)
        arg_type = int if arg_type == bool else arg_type
        parser.add_argument(f"--{arg_name}", type=arg_type, default=default_val)

    if "id" not in default_dict:
        parser.add_argument("--id", type=int)
    if "db_path" not in default_dict:
        parser.add_argument("--db_path", type=str)

    args = parser.parse_args()
    args = vars(args)
    recover_boolean_args(args, default_dict)

    return args
