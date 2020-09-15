import torch.nn as nn

def modularize(fxn, name=None):
    if not name:
        name = fxn.__name__

    class SubModule(nn.Module):
        def __init__(self):
            super(SubModule, self).__init__()

    return SubModule()