import pytest

from intervention import ComputationGraph, GraphNode, GraphInput

@pytest.fixture
def arithmetic_graph():
    class ArithmeticGraph(ComputationGraph):
        def __init__(self):
            @GraphNode()
            def leaf1(a, b, c):
                print("leaf1 = a + b + c = %d" % (a + b + c))
                return a + b + c

            @GraphNode()
            def leaf2(d, e):
                print("leaf2 = (d + e) / 10 = %f" % ((d + e) / 10))
                return (d + e) / 10

            @GraphNode(leaf1)
            def child1(x):
                print("child1 = leaf1 * 2 = %d" % (x * 2))
                return x * 2

            @GraphNode(leaf1, leaf2)
            def child2(x, y):
                print("child2 = leaf1 - leaf2 = %f" % (x - y))
                return x - y

            @GraphNode(child1, child2)
            def root(w, z):
                print("root = child1 + child2 + 1 = %f" % (w + z + 1))
                return w + z + 1

            super().__init__(root)

    return ArithmeticGraph()

@pytest.fixture
def singleton_graph():
    return

