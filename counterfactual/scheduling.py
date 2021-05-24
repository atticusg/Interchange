import math

def get_subepoch_sizes(base_dataset_size, batch_size, num_subepochs):
    total_num_batches = math.ceil(base_dataset_size / batch_size)
    subepoch_size = total_num_batches // num_subepochs
    subepoch_sizes = [subepoch_size for _ in range(num_subepochs)]
    # add the remainder of batches to the first subepoch
    subepoch_sizes[0] += total_num_batches - subepoch_size * num_subepochs

    return subepoch_sizes, total_num_batches

class LinearCfTrainingSchedule:
    def __init__(
            self,
            base_dataset,
            batch_size,
            num_subepochs=20,
            warmup_subepochs=5,
            ratio_step_size=0.1,
            final_ratio=1.0
    ):
        """
        |---------------------------------------| base dataset

        |---|---|---|---|---|---|---|---|---|---| divide into subepochs

        |---|---|---|---+|---+|---++|---++|---+++|---+++|---+++| ...
        | warmup    | ratio of cf examples increases linearly until final_ratio  ...

        :param base_dataset:
        :param batch_size:
        :param num_subepochs:
        :param warmup_subepochs:
        """
        self.subepoch_sizes, self.total_num_batches = \
            get_subepoch_sizes(len(base_dataset), batch_size, num_subepochs)

        self.warmup_subepochs = warmup_subepochs
        self.ratio_step_size = ratio_step_size
        self.final_ratio = final_ratio

    def __call__(self, epoch):
        subepoch_idx = epoch % len(self.subepoch_sizes)
        num_base_examples = self.subepoch_sizes[subepoch_idx]
        if epoch < self.warmup_subepochs:
            return [num_base_examples, 0]
        else:
            cf_ratio = min(self.final_ratio, (epoch - self.warmup_subepochs) * self.ratio_step_size)
            num_cf_examples = math.ceil(cf_ratio * num_base_examples)
            return [num_base_examples, num_cf_examples]

class FixedRatioSchedule:
    def __init__(
        self,
        dataset_sizes,
        batch_size,
        num_subepochs=None,
        ratio=None
    ):
        self.ratio = ratio
        self.total_num_batches = None
        self.subepoch_sizes = None

        if ratio is None:
            # No ratio given, directly return num batches in each dataset
            self.total_num_batches = [
                math.ceil(dataset_size / batch_size) for dataset_size in dataset_sizes
            ]
        else:
            # Ratio and num_subepoch, treat first dataset as base and return
            # fixed ratio of examples
            self.subepoch_sizes, self.total_num_batches = \
                    get_subepoch_sizes(dataset_sizes[0], batch_size, num_subepochs)


    def __call__(self, epoch):
        if self.ratio is None:
            return self.total_num_batches
        else:
            subepoch_idx = epoch % len(self.subepoch_sizes)
            num_base_examples = self.subepoch_sizes[subepoch_idx]
            num_cf_examples = math.ceil(self.ratio * num_base_examples)
            return [num_base_examples, num_cf_examples]
