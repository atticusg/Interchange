import torch


def my_collate(batch, batch_first=True):
    sorted_batch = sorted(batch, key=lambda pair: pair[0].shape[0], reverse=True)
    sequences = [x[0] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
    lengths = torch.tensor([len(x) for x in sequences])
    labels = torch.tensor([x[1] for x in sorted_batch])
    return (sequences_padded, labels, lengths)


def write_pickle(d, f):
    torch.save(d, f)


def read_pickle(f):
    return torch.load(f)


