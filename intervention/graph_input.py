import torch


class GraphInput:
    """ A hashable input object that stores a dict mapping names of nodes to
    values of arbitrary type.

    `GraphInput` objects are intended to be immutable, so that its hash value
    can have a one-to-one correspondence to the dict stored in it. """

    def __init__(self, values, device=None):
        if device:
            assert all(
                isinstance(t, torch.Tensor) for _, t in values.items())
            self._values = {k: v.to(device) for k, v in values.items()}
        else:
            self._values = values

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, value):
        raise RuntimeError("GraphInput objects are immutable!")

    def __getitem__(self, item):
        return self.values[item]

    def __contains__(self, item):
        """ Override the python `in` operator """
        return item in self.values

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        if self.values is None:
            return "GraphInput{}"
        else:
            s = ", ".join(
                ("'%s': %s" % (k, type(v))) for k, v in self.values.items())
            return "GraphInput{%s}" % s

    def to(self, device):
        """Move all data to a pytorch Device.

        This does NOT modify the original GraphInput object but returns a new
        one. """
        assert all(isinstance(t, torch.Tensor) for _, t in self._values.items())

        new_values = {k: v.to(device) for k, v in self._values.items()}
        return GraphInput(new_values)
