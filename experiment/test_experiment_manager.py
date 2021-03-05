from experiment.manager import ExperimentManager
from experiment.db_utils import fetch_new

TABLE_NAME = "results"

def test_expt_manager_setup():
    db_path = "test_manager.db"
    default_opts = {"x": 0, "y": 0}
    launch_script = ""

    manager = ExperimentManager(db_path, default_opts, launch_script)


def test_fetch_new():
    res = fetch_new("test_manager.db", TABLE_NAME, ["x", "y"])
    print(res)


def test_run():
    db_path = "test_manager.db"
    default_opts = {"x": 0, "y": 0}
    launch_script = "python ../expt_example.py"

    manager = ExperimentManager(db_path, default_opts, launch_script)
    for x, y in [(6, 4), (-2, 100), (100, 5)]:
        manager.insert({"x": x, "y": y})

    manager.run()


