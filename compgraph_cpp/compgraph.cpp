#include <torch/extension.h>
// the above includes all necessary PyTorch bits to write C++ extensions

#include <torch/csrc/autograd/variable.h>

#include <stdio.h> // C nostalgia
#include <string>
#include <list>

#include "print.h"

using torch::Tensor;
using torch::autograd::Variable;
using torch::autograd::Node;


void traverse_graph(const Node& n, const std::string& parent_name) {
  print("---- visiting " + n.name());
  const edge_list& next_edges = n.next_edges();
  print("    My parent is " + parent_name);
  printf("    I have %lu parents/children\n", next_edges.size());

  if (next_edges.size() > 0) {
    print("ckpt 1");
    for (auto& edge: next_edges) {
      print("ckpt 2");
      auto& next_node = *(edge.function);
      print("ckpt 3");
      traverse_graph(next_node, n.name());
      print("ckpt 4");
    }
  }
}

void inspect(Tensor input) {
  print("inspecting tensor");
  print(input);
  print("Obtaining Node");
  auto root_node = input.grad_fn();
  // see "/pytorch/aten/src/ATen/core/Tensor.cpp", grad_fn() returns a
  // std::sharedptr to a Node object
  print("Obtained Node");
  std::string parent_name = "root";
  traverse_graph(*root_node, parent_name);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inspect", &inspect, "Inspect tensor");
}
