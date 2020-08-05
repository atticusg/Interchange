import torch
import numpy as np
import copy

def copy_helper(x):
    if isinstance(x, (list, tuple, str, dict, np.ndarray)):
        return copy.deepcopy(x)
    elif isinstance(x, torch.Tensor):
        return x.detach().clone()
    else:
        return x