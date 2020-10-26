import torch
import numpy as np
import copy

import torch
from typing import Union

def copy_helper(x):
    if isinstance(x, (list, tuple, str, dict, np.ndarray)):
        return copy.deepcopy(x)
    elif isinstance(x, torch.Tensor):
        return x.detach().clone()
    else:
        return x


def serialize(x):
    if isinstance(x, torch.Tensor):
        if len(x.shape) == 0:
            return x.item()
        elif len(x.shape) == 1:
            return tuple(x.tolist())
        elif len(x.shape) == 2:
            return tuple(tuple(d0) for d0 in x.tolist())
        elif len(x.shape) == 3:
            return tuple(tuple(tuple(d1) for d1 in d0) for d0 in x.tolist())
        elif len(x.shape) == 4:
            return tuple(tuple(tuple(tuple(d2) for d2 in d1) for d1 in d0) for d0 in x.tolist())
        else:
            raise NotImplementedError(f"cannot serialize x with {len(x.shape)} dimensions")
    elif isinstance(x, np.ndarray):
        return x.tostring()

def deserialize(x: tuple):
    return torch.tensor(x)
