import sqlite3
import json

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
        raise RuntimeError(e)

def get_col_names(db_path, table_name):
    conn = db_connect(db_path)
    with conn:
        c = conn.cursor()
        c.execute(f"SELECT * FROM {table_name}")
        col_names = set(d[0] for d in c.description)
    return col_names

type2str = {float: 'real', int: 'integer', list: 'text', str: 'text',
            bool: 'bool', tuple: 'text', dict: 'text'}

def create_table(db_path, table_name, default_opts):
    conn = db_connect(db_path)
    cmd = create_cmd(table_name, default_opts)
    with conn:
        cur = conn.cursor()
        print(cmd)
        cur.execute(cmd)

def create_cmd(table_name, opts):
    cmd = f"CREATE TABLE IF NOT EXISTS {table_name}(\nid integer PRIMARY KEY,\n"

    opts_cols = []

    for col, value in opts.items():
        opts_cols.append(col)
        type_str = type2str[type(value)]

        default = str(value)
        if type_str == 'text':
            default = '\"' + default + '\"'
        elif default == '\"None\"':
            default = 'NULL'

        cmd += f'{col} {type_str} DEFAULT {default},\n'

    cmd = cmd.strip(',\n') + '\n);'
    return cmd


def add_cols(db_file, table_name, res_dict):
    conn = db_connect(db_file)
    with conn:
        c = conn.cursor()
        c.execute(f"SELECT * FROM {table_name} LIMIT 1;")
        existing_cols = set(d[0] for d in c.description)
        new_cols = [k for k in res_dict.keys() if k not in existing_cols]

        if len(new_cols) == 0:
            return

        for col_name in new_cols:
            type_str = type2str[type(res_dict[col_name])]
            cmd_str = f'ALTER TABLE {table_name} ADD COLUMN {col_name} {type_str};'
            c.execute(cmd_str)


def update(db_path, table_name, insert_dict, id=None):
    all_cols = list(insert_dict.keys())
    cmd = update_cmd(table_name, all_cols, id)

    l = []
    for item in all_cols:
        value = insert_dict[item]
        if isinstance(value, list) or isinstance(value, tuple) or isinstance(
                value, dict):
            value = json.dumps(value)
        l.append(value)

    cmd_tuple = tuple(l)
    conn = db_connect(db_path)
    with conn:
        cur = conn.cursor()
        cur.execute(cmd, cmd_tuple)
        new_id = cur.lastrowid
    return new_id


def update_cmd(table_name, cols, id=None):
    if id:
        set_str = ", ".join([f"{c} = ?" for c in cols])
        cond_str = f"id = {id}"
        cmd = f"UPDATE {table_name} SET {set_str} WHERE {cond_str};"
    else:
        cols_string = '(' + ', '.join(cols) + ')'
        vals_string = '(' + ', '.join(['?' for _ in cols]) + ')'
        cmd = f'INSERT INTO {table_name} {cols_string} VALUES {vals_string};'
    return cmd


def fetch_new(db_path, table_name, opt_cols, n=None):
    cols = ["id"] + opt_cols
    cond_dict = {"status": 0}
    return select(db_path, table_name, cols=cols, cond_dict=cond_dict, limit=n)


def select(db_path, table_name, cols=None, cond_dict=None, like=None, limit=None):
    cmd = select_cmd(table_name, cols=cols, cond_dict=cond_dict, like=like,
                     limit=limit)

    conn = db_connect(db_path)
    with conn:
        cur = conn.cursor()
        cur.execute(cmd)
        rows = cur.fetchall()

    res = [{col_name: value for col_name, value in zip(cols, row)}
           for row in rows]
    return res


def select_cmd(table_name, cols=None, cond_dict=None, like=None, limit=None):
    cols_string = "*" if cols is None else ", ".join(cols)
    cmd = f"SELECT {cols_string} FROM {table_name}"
    if cond_dict:
        cmd += " WHERE "
        cmd += " AND ".join(f"{k} = {v}" for k, v in cond_dict.items())
    if like:
        cmd += " AND "
        cmd += " AND ".join(f"{k} LIKE {p}" for k, p in like.items())
    if limit:
        cmd += f" LIMIT {limit}"

    cmd += ";"
    return cmd
