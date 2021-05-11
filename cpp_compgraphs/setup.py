from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='compgraph_cpp',
      ext_modules=[cpp_extension.CppExtension('compgraph_cpp', ['compgraph.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
