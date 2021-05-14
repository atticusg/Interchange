from typing import *
import math
import torch
import torch.utils.data as torch_data

import itertools

import logging
log = logging.getLogger(__name__)


class MultiTaskDataLoader:
    def __init__(
            self,
            tasks: List[Iterable],
            schedule_fn: Callable[[int], List[int]],
            return_task_name: bool = False,
            task_names: Optional[List[str]] = None
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
        self.tasks = tasks
        self.schedule_fn = schedule_fn
        self.return_task_name = return_task_name
        self.task_names = task_names

        self.curr_step = None
        self.curr_epoch = None
        self.total_steps = None

        self.iterators = self.init_iterators()

    def init_iterators(self):
        iterators = []
        for dl in self.tasks:
            iterators.append(iter(dl))
        return iterators

    def __iter__(self):
        # this will be called every time before a for loop starts
        # set counts
        if self.curr_epoch is None:
            self.curr_epoch = 0
        else:
            self.curr_epoch += 1
        # print(f"Calling __iter__ at epoch {self.curr_epoch}")
        self.curr_step = 0

        # get number of items for each task from schedule function
        self.curr_schedule = self.schedule_fn(self.curr_epoch)
        # [3, 1, 2] --> [0, 0, 0, 1, 2, 2]
        task_idxs = torch.tensor([x for idx, n in enumerate(self.curr_schedule) for x in itertools.repeat(idx, n)])

        # shuffle tasks for this epoch
        shuffle = torch.randperm(len(task_idxs))
        self.curr_task_idxs = task_idxs[shuffle].tolist()
        return self

    def __next__(self):
        if self.curr_step >= len(self.curr_task_idxs):
            raise StopIteration

        curr_task_idx = self.curr_task_idxs[self.curr_step]
        curr_iter = self.iterators[curr_task_idx]

        try:
            res = next(curr_iter)
        except StopIteration:
            # if an iterator is exhausted initialize a new one
            new_iter = iter(self.tasks[curr_task_idx])
            self.iterators[curr_task_idx] = new_iter
            res = next(new_iter)

        self.curr_step += 1
        if self.return_task_name:
            curr_task_name = curr_task_idx if self.task_names is None else self.task_names[curr_task_idx]
            return (res, curr_task_name)
        else:
            return res
