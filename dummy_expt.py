from experiment.manager import Experiment

import argparse

class TestExperiment(Experiment):
    def experiment(self, opts: dict):
        res_dict = {"xplusone": opts["x"] + 1,
                    "ysquared": opts["y"] ** 2}
        return res_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--x", type=int, required=True)
    parser.add_argument("--y", type=int, required=True)
    parser.add_argument("--id", type=int)
    parser.add_argument("--db_path", type=str)

    # data_path = "mqnli_data/mqnli.pt"
    # data = MQNLIData("mqnli_data/mqnli.train.txt",
    #                  "mqnli_data/mqnli.dev.txt",
    #                  "mqnli_data/mqnli.test.txt")
    # torch.save(data, data_path)
    # model_path = "mqnli_models/lstm_best.pt"
    # db_path = "experiment_data/runtime.db"

    args = parser.parse_args()
    e = TestExperiment()
    e.run(vars(args))