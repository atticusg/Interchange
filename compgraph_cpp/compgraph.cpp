#include <torch/extension.h>
// the above includes all necessary PyTorch bits to write C++ extensions

#include <torch/csrc/autograd/variable.h>

#include "print.h"

using torch::Tensor;
using torch::autograd::Variable;

void inspect(Tensor input) {
  print("inspecting tensor");
  print(input);
  print("Treating input as variable");
  Variable v = make_variable(input);
  print("success");
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inspect", &inspect, "Inspect tensor");
}
