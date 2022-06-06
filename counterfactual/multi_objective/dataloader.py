from typing import *
import torch

import itertools
import logging
from .dataset import MQNLIMultiObjectiveDataset
log = logging.getLogger(__name__)


class MultiObjectiveDataLoader:
    def __init__(
            self,
            dataset: MQNLIMultiObjectiveDataset,
            dataloader: Iterable,
            weight_fn: Optional[Callable[[int], Dict[str, float]]],
            weight_per_epoch: bool = True
    ):
        """ Randomly sample items from different iterable tasks at a given
        ratio that can vary given the current epoch number.

        If one task is not fully exhausted in one epoch, it will not
        be reset in the next epoch.

        Once a task is exhausted, it will be immediately reset.

        :param tasks: List of iterable objects, each being a task
        :param schedule_fn: Given an epoch index, returns the number of items to
            get from each task at that epoch
        """
        self.dataset = dataset
        self.weight_fn = weight_fn
        self.weight_per_epoch = weight_per_epoch
        self.dataloader = dataloader
        self.curr_step = None
        self.curr_epoch = None
        self.total_steps = None
        self.num_examples = dataset.num_examples

    def __iter__(self):
        # this will be called every time before a for loop starts
        # set counts
        self.dataloader_iterator = iter(self.dataloader)
        if self.curr_epoch is None:
            self.curr_epoch = 0
        else:
            self.curr_epoch += 1
        # print(f"Calling __iter__ at epoch {self.curr_epoch}")
        self.curr_step = 0
        return self

    def __next__(self):
        try:
            next_item = next(self.dataloader_iterator)
        except StopIteration:
            raise StopIteration

        if self.weight_fn:
            weight = self.weight_fn(self.curr_epoch) if self.weight_per_epoch else self.weight_fn(self.curr_step)
            res = (next_item, weight)
        else:
            res = (next_item, None)
        self.curr_step += 1
        return res

