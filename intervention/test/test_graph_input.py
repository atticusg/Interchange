import pytest
import torch
from intervention import GraphInput

def test_init_device():
    d = {str(x): torch.randn(10) for x in range(5)}
    device = torch.device("cuda")
    i = GraphInput(d, device=device)

    assert all(t.is_cuda for t in i.values.values())
