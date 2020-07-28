import torch

import ast
import inspect
from textwrap import dedent

from intervention.computation_graph import GraphNode, CompGraphConstructor, GraphInput

class TorchEqualityModule(torch.nn.Module):
    def __init__(self,
                 input_size=20,
                 hidden_layer_size=100,
                 activation="relu"):
        super(TorchEqualityModule, self).__init__()
        self.linear = torch.nn.Linear(input_size,hidden_layer_size)
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        else:
            raise NotImplementedError("Activation method not implemented")
        self.output = torch.nn.Linear(hidden_layer_size, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        linear_out = self.linear(x)
        self.hidden_vec = self.activation(linear_out)
        logits = self.output(self.hidden_vec)
        return self.sigmoid(logits)

def size_to_str(size):
    return '(' + (', ').join(['%d' % v for v in size]) + ')'

def traverse_backprop_graph(res, params=None):
    """ traverse the compgraph constructed by autograd,
    inspired by pytorchviz """
    if params is not None:
        param_map = {id(v): k for k, v in params.items()}
    seen = set()

    def add_nodes(var):
        if var not in seen:
            if hasattr(var, "variable"):
                u = var.variable
                name = param_map[id(u)] if params is not None else ""
                node_name = "%s\n %s" % (name, size_to_str(u.size()))

                print("\n----visiting variable node %s", node_name)
            else:
                print("\n----visiting backprop node %s", str(type(var).__name__))
                print("    available methods:")
                print("    ", dir(var))
                print("    ", var.__class__.__bases__)


            seen.add(var)

            if hasattr(var, "next_functions"):
                print("    this node has %d children" % len(var.next_functions))
                for u in var.next_functions:
                    if u[0] is not None:
                        add_nodes(u[0])

    add_nodes(res.grad_fn)

def ast_traverse(f):
    # parse python code that defines the module to get AST tree
    filename = inspect.getsourcefile(f)
    sourcelines, file_lineno = inspect.getsourcelines(f)
    print("location: %s:%d" % (filename, file_lineno))
    print("sourcelines:", sourcelines)

    source = ''.join(sourcelines)
    dedent_src = dedent(source)
    tree_root = ast.parse(dedent_src)
    MyVisitor().visit(tree_root)
    # print(ast.dump(tree))

class MyVisitor(ast.NodeVisitor):
    def visit_FunctionDef(self, node):
        if node.name == "forward":
            self.traverse_body(node)

    def traverse_body(self, node):
        print("traversing the body of %s" % node.name)
        for n in node.body:
            print(n)


def make_graph_from_hooks(m, x):
    constructor = CompGraphConstructor(m)
    g = constructor.make_graph(x)

    return g


def main():
    m = TorchEqualityModule()
    x = torch.randn(20)

    # comp_graph = make_graph_from_hooks(m, x)

    comp_graph2 = CompGraphConstructor.construct(m, x)

    input = GraphInput({"linear": x})
    # print(comp_graph.compute(input))
    print(comp_graph2.compute(input))
    # res = m(x)
    # res.backward()
    # traverse_backprop_graph(res, params=dict(m.named_parameters()))

    # ast_traverse(TorchEqualityModule)


if __name__ == "__main__":
    main()
