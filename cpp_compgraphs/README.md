# Experimental PyTorch C++ Extension for Automatically Building Computational Graphs

According to the official PyTorch documentation, we can access its C++ API by
building a C++ extension (see [here](https://pytorch.org/cppdocs/)).

I basically followed one of its [official tutorials](https://pytorch.org/tutorials/advanced/cpp_extension.html)
to build one.

Currently I wrote a very simple `inspect` function in C++, that takes in a `torch::Tensor`.
We assume that a series of calculations led to that tensor, and our goal is to
traverse the computational graph that formed these series of calculations.

## Contents

`compgraph.cpp` defines the `inspect` function and does the necessary binding
 to make it accessible in Python.

`print.cpp` and `print.h` define some helper functions to print stuff in C++ (I
find the `std::cout << ... << std::endl;` syntax very annoying)

`test.py` builds the extension dynamically (see below) and tests the `inspect` function
with some toy inputs.

`setup.py` is necessary for the static build (again, see below)

## How to build

There are two ways to incorporate the C++ code into python in the form of a
module. The first way is more static, by running a setup script which compiles
the C++ code and sets up the correct links and bindings. The second way is
more dynamic (and more suitable for testing), which is to run a function in
python and provide a path to the C++ source code. This function will return
a python module object, from which we can call functions that we defined in C++.

It is recommended that you do all the following via a Conda environment (because
I haven't tried how it would work in other cases, at least for now).

Also I only tried this on my local Linux platform, and I can't confirm whether
it will work on others.

### Requirements

See `requirements.txt`. It turns out `cmake` is crucial to make the build work
on my Linux platform. Without it I wasn't able to import the extension into python.

Check out [this link](https://anaconda.org/anaconda/cmake)
to install it via Conda.

### Static build

After installing all the requirements, simply run
```
$ python setup.py install
```
This script installs a new python module called `compgraph_cpp` in a similar
way as you would install any other module via pip. It will compile the C++
code and produce some files in the `compgraph` directory containing binary code
and other files crucial for it to function as a python module.


After installation, you can
```
import torch
import compgraph_cpp
```
in a python file in the same Conda environment. According to the official
documentation, you must `import torch` first before importing the extension
module.

Then you can call the `inspect` function by:
```
x = torch.tensor(...)
compgraph_cpp.inspect(x)
```

### Dynamic build

This line of code does the job:
```
compgraph_cpp = torch.utils.cpp_extension.load(
  name="compgraph_cpp",
  sources=["compgraph.cpp", "print.cpp"])
```
This will successfully build the `compgraph_cpp` python module and
we can use it in the exact same way as above, except it doesn't install anything
locally.
