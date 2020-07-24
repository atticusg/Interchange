import torch
from torch.utils import cpp_extension

print("loading compgraph_cpp")
compgraph_cpp = cpp_extension.load(name="compgraph_cpp", sources=["compgraph.cpp", "print.cpp"])


a = torch.tensor([1,2,3,4,5])
b = torch.tensor([10] * 5)
c = a + b
d = c * 2
compgraph_cpp.inspect(d)
