
class Experiment:
    def __init__(self, id, inputs, interventions=None):
        self.inputs = inputs
        self.interventions = interventions
        self.id = id

class ComputationGraph:
    def __init__(self, root):
        self.root = root

    def validate_graph(self):
        self.names_to_nodes = {}

        def add_node(node):
            if node.name in self.names_to_nodes:
                raise ValueError("Names of nodes must be unique!")
            else:
                self.names_to_nodes[node.name] = node
                for child in node.children:
                    add_node(child)

        add_node(self.root)

    def compute(self, expt):
        return self.root.compute(expt)

class GraphNode:
    def __init__(self, *args, name=None, forward=None):
        self.children = args
        self.expt_cache = {}
        self.name = name
        if forward:
            self.forward = forward
            self.name = forward

    def __call__(self, f):
        self.forward = f
        if self.name is None:
            self.name = f.__name__
        return self

    def __repr__(self):
        return "XGraphNode(\"%s\")" % self.name

    def get_cache(self, expt):
        return self.expt_cache.get(expt.id, None)

    def set_cache(self, expt, value):
        self.expt_cache[expt.id] = value

    def clear_cache(self):
        del self.expt_cache
        self.expt_cache = {}

    def compute(self, expt):
        # check cache first if calculation results exist in cache
        res = self.get_cache(expt)
        if res is not None:
            return res

        if len(self.children) == 0:
            # leaf
            values = expt.inputs[self.name]
            res = self.forward(*values)
        else:
            # non-leaf node
            children_res = [c.compute(expt) for c in self.children]

            print("computing node", self.name)
            res = self.forward(*children_res)

        self.set_cache(expt, res)
        return res


@GraphNode()
def leaf1(a, b, c):
    print("a + b + c = %d" % (a + b + c))
    return a + b + c

@GraphNode()
def leaf2(d, e):
    print("(d + e) / 10 = %f" % (d + e))
    return (d + e) / 10

@GraphNode(leaf1)
def child1(x):
    print("root * 2 = %d" % (x * 2))
    return x * 2

@GraphNode(leaf1, leaf2)
def child2(x, y):
    print("root - root2 = %f" % (x-y))
    return x - y

@GraphNode(child1, child2)
def final(w, z):
    print("child1 + child2 + 1 = %f" % (w + z+1))
    return w + z + 1

g = ComputationGraph(final)
expt = Experiment(id=1, inputs={"leaf1": (10, 20, 30), "leaf2": (2,3)})
res = g.compute(expt)

print(res)