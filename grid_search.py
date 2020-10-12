import time, torch, sqlite3
from train import Trainer

from itertools import product
from datetime import datetime
import json

SAMPLE_RES_DICT = {
    'dev_acc': 0.,
    'loss': 0.,
    'epoch': 0,
    'step': 0
}

def db_connect(db_file):
    """ create a database connection to the SQLite database
    Args:
        db_file: database file
    Returns: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)

    return None


class GridSearch:
    def __init__(self, model_class, data, base_model_opts, base_train_opts, db_path, base_res_dict=None):
        """Initializes the GridSearch class.
        Assumes that parameters of the s0 models are not changed
        """
        self.model_class = model_class
        self.data = data
        self.base_model_opts = base_model_opts.copy()
        self.base_train_opts = base_train_opts.copy()
        if base_res_dict is None:
            base_res_dict = SAMPLE_RES_DICT

        all_opts = {}
        all_opts.update(self.base_model_opts)
        all_opts.update(self.base_train_opts)

        if "device" in all_opts:
            del all_opts["device"]
        self.db_path = db_path
        if self.db_path:
            conn = db_connect(self.db_path)
            cmd, opts_cols, res_cols = create_cmd(all_opts, base_res_dict)
            with conn:
                cur = conn.cursor()
                print(cmd)
                cur.execute(cmd)
            self.db_opts_cols = opts_cols
            self.db_res_cols = res_cols

    def execute(self, grid_dict):
        var_opt_names = list(grid_dict.keys())
        var_opt_values = list(v if isinstance(v, list) else list(v) for v in grid_dict.values())

        # treat elements in list as separate args to fxn
        for tup in product(*var_opt_values):
            model_opt_dict = {}
            train_opt_dict = {}
            for name, val in zip(var_opt_names, tup):
                if name in self.base_model_opts:
                    model_opt_dict[name] = val
                elif name in self.base_train_opts:
                    train_opt_dict[name] = val
                else:
                    raise ValueError("Invalid config parameter name: %s" % name)

            model_config = self.base_model_opts.copy()
            train_config = self.base_train_opts.copy()

            model_config.update(model_opt_dict)
            train_config.update(train_opt_dict)
            self.run_once(model_config, train_config)

    def run_once(self, model_config, train_config):
        model = self.model_class(**model_config).to(model_config["device"])

        date_str = datetime.now().strftime("%m%d_%H%M%S")
        model_save_path = '%s_%s.pt' % (train_config["model_save_path"], date_str)
        train_config["model_save_path"] = model_save_path
        print("model_config:", model_config)
        print("train_config:", train_config)

        trainer = Trainer(self.data, model, **train_config)
        best_model_ckpt = trainer.train()

        res_dict = {"dev_acc": best_model_ckpt["best_dev_acc"], "loss": best_model_ckpt["loss"],
                    "epoch": best_model_ckpt["epoch"], "step": best_model_ckpt["step"]}

        self.record_results(model_config, train_config, res_dict)

    def record_results(self, model_config, train_config, res_dict):
        if self.db_path:
            opts = {}
            opts.update(model_config)
            opts.update(train_config)
            self.db_insert(opts, res_dict)
            print('>>>> Added entry to database {}'.format(self.db_path))

    def db_insert(self, opts, res_dict):
        assert len(opts) >= len(self.db_opts_cols)
        assert len(res_dict) >= len(self.db_res_cols)

        all_cols = self.db_opts_cols + self.db_res_cols
        cmd = insert_cmd(all_cols)
        l = []
        for item in self.db_opts_cols:
            value = opts[item]
            if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, dict):
                value = json.dumps(value)
            l.append(value)
        for item in self.db_res_cols:
            value = res_dict[item]
            if isinstance(value, list) or isinstance(value,tuple) or isinstance(value, dict):
                value = json.dumps(value)
            l.append(value)

        cmd_tuple = tuple(l)

        conn = db_connect(self.db_path)
        with conn:
            cur = conn.cursor()
            cur.execute(cmd, cmd_tuple)
            new_id = cur.lastrowid
        return new_id


def insert_cmd(cols):
    cols_string = '(' + ','.join(cols) + ')'
    vals_string = '(' + ','.join(['?' for x in cols]) + ')'
    cmd = f'INSERT INTO results {cols_string} VALUES {vals_string}'

    return cmd


def create_cmd(opts, base_res_dict):
    cmd = "CREATE TABLE IF NOT EXISTS results(\nid integer PRIMARY KEY,\n"

    type2str = {float: 'real', int: 'integer', list: 'text', str: 'text',
                bool: 'bool', tuple: 'text', dict: 'text'}

    opts_cols = []

    for col, value in opts.items():
        opts_cols.append(col)
        type_str = type2str[type(value)]
        default = str(value)
        if type_str == 'text':
            default = '\"' + default + '\"'
        if default == '\"None\"':
            default = 'NULL'

        s = f'{col} {type_str} DEFAULT {default}'

        if not isinstance(value, list):
            s += ' NOT NULL'
        s += ',\n'
        cmd += s

    res_cols = []
    for k, value in base_res_dict.items():
        col = k
        res_cols.append(col)
        type_str = type2str[type(value)]
        s = f'{col} {type_str} NOT NULL'
        s += ',\n'
        cmd += s

    cmd = cmd.strip(',\n') + '\n);'

    return cmd, opts_cols, res_cols