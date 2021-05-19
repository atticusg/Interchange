import math

class LinearCfTrainingSchedule:
    def __init__(
            self,
            base_dataset,
            batch_size,
            num_subepochs=20,
            warmup_subepochs=5,
            ratio_step_size=0.1,
    ):
        """
        |---------------------------------------| base dataset

        |---|---|---|---|---|---|---|---|---|---| divide into subepochs

        |---|---|---|---+|---+|---++|---++|---+++|---+++|---+++| ...
        | warmup    | ratio of cf examples increases linearly until 1:1  ...

        :param base_dataset:
        :param batch_size:
        :param num_subepochs:
        :param warmup_subepochs:
        """
        self.total_num_batches = math.ceil(len(base_dataset) / batch_size)
        subepoch_size = self.total_num_batches // num_subepochs
        self.subepoch_sizes = [subepoch_size for _ in range(num_subepochs)]
        # add the remainder of batches to the first subepoch
        self.subepoch_sizes[0] += self.total_num_batches - subepoch_size * num_subepochs
        self.warmup_subepochs = warmup_subepochs
        self.ratio_step_size = ratio_step_size

    def __call__(self, epoch):
        subepoch_idx = epoch % len(self.subepoch_sizes)
        num_base_examples = self.subepoch_sizes[subepoch_idx]
        if epoch < self.warmup_subepochs:
            return [num_base_examples, 0]
        else:
            cf_ratio = min(1.0, (epoch - self.warmup_subepochs) * self.ratio_step_size)
            num_cf_examples = math.ceil(cf_ratio * num_base_examples)
            return [num_base_examples, num_cf_examples]

class FixedRatioSchedule:
    def __init__(
        self,
        dataset_sizes,
        batch_size
    ):
        self.total_num_batches = [
            math.ceil(dataset_size / batch_size) for dataset_size in dataset_sizes
        ]

    def __call__(self, epoch):
        return self.total_num_batches
