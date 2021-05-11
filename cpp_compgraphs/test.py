import torch
from torch.utils import cpp_extension

print("loading compgraph_cpp")
compgraph_cpp = cpp_extension.load(name="compgraph_cpp", sources=["compgraph.cpp", "print.cpp"])


a = torch.tensor([.1,.2,.3,.4,.5], requires_grad=True)
b = torch.tensor([1.] * 5, requires_grad=True)
c = a + b
d = c * 2.
e = d.mean()

# IMPORTANT NOTE: requires_grad must be set to true for computational graph
# to be formed
compgraph_cpp.inspect(e)
