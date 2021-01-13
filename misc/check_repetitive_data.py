from datasets.mqnli import MQNLIData, MQNLIDataset

def check_repetitive_examples(data):
    print("preparing dev set")
    dev_set = set((tuple(example[0].tolist()), example[1]) for example in data.test)
    print("Finished preparing. Showing first few examples")
    printed = 0
    for dev_tuple in dev_set:
        if printed >= 3: break
        print(dev_tuple)
        printed += 1

    print("Checking samples in the train set")
    total = len(data.train)
    repeated = 0
    for train_i in range(len(data.train)):
        train_example = data.train[train_i]
        train_tuple = (tuple(train_example[0]), train_example[1])
        if train_tuple in dev_set:
            repeated += 1

    print("Finished checking %d examples, found %d repeating examples (%.2f%%)"
          % (total, repeated, repeated / total))


if __name__ == "__main__":
    data = MQNLIData("/home/hansonlu/intervention/Interchange/data/mqnli/raw/easy/train.txt",
              "/home/hansonlu/intervention/Interchange/data/mqnli/raw/easy/dev.txt",
              "/home/hansonlu/intervention/Interchange/data/mqnli/raw/easy/test.txt")
    check_repetitive_examples(data)