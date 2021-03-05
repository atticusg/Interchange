import torch



def write_pickle(d, f):
    torch.save(d, f)


def read_pickle(f):
    return torch.load(f)
