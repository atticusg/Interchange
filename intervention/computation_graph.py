from abc import ABC
import re
import torch

class Experiment:
    def __init__(self, id, inputs, interventions=None):
        self.inputs = inputs
        self.interventions = interventions
        self.input_key = id

class Loc:
    def __init__(self, loc=None):
        self._loc = Loc.process(loc) if loc else None

    def __getitem__(cls, item):
        return item

    @classmethod
    def str_to_slice(cls, s):
        return slice(*map(lambda x: int(x.strip()) if x.strip() else None, s.split(':')))

    @classmethod
    def parse_dim(cls, s):
        return Ellipsis if s == "..." \
            else True if s == "True" \
            else False if s == "False" \
            else Loc.str_to_slice(s) if ":" in s \
            else int(s)

    @classmethod
    def parse_str(cls, s):
        return tuple(Loc.parse_dim(x.strip()) for x in s.split(","))

    @classmethod
    def process(cls, x):
        if isinstance(x, int) or isinstance(x, list) or isinstance(x, tuple) \
                or x is Ellipsis or isinstance(x, slice):
            return x
        elif isinstance(x, str):
            return Loc.parse_str(x)

class GraphInput:
    """ A hashable input object that stores a dict mapping names of nodes to
    values of arbitrary type.

    `GraphInput` objects are intended to be immutable, so that its hash value
    can have a one-to-one correspondence to the dict stored in it. """
    def __init__(self, values):
        self._values = values

    @property
    def values(self):
        return self._values

    def __getitem__(self, item):
        return self._values[item]

    def __contains__(self, item):
        """ Override the python `in` operator """
        return item in self._values

    def __len__(self):
        return len(self._values)

    def __repr__(self):
        if self.values is None:
            return "GraphInput{}"
        else:
            s = ", ".join(("'%s': %s" % (k, type(v))) for k, v in self._values.items())
            return "GraphInput{%s}" % s

class Intervention:
    """ A hashable intervention object """
    def __init__(self, base, inputs=None, locs=None):
        inputs = {} if inputs is None else inputs
        locs = {} if locs is None else locs
        self._setup(base, inputs, locs)
        self.affected_nodes = None

    def _setup(self, base=None, inputs=None, locs=None):
        if base is not None:
            if isinstance(base, dict):
                base = GraphInput(base)
            self._base = base

        if locs is not None:
            # specifying any value that is not None will overwrite self._locs
            locs = {name: Loc.process(loc) for name, loc in locs.items()}
        else:
            locs = self._locs

        if inputs is not None:
            if isinstance(inputs, GraphInput):
                inputs = inputs.values

            # extract indexing in names
            loc_pattern = re.compile("\[.*\]")
            to_delete = []
            to_add = {}
            for name, value in inputs.items():
                # find if there is a index-like expression in name
                loc_search = loc_pattern.search(name)
                if loc_search:
                    true_name = name.split("[")[0]
                    loc_str = loc_search.group().strip("[]")
                    loc = Loc.parse_str(loc_str)
                    to_delete.append(name)
                    to_add[true_name] = value
                    locs[true_name] = loc

            # remove indexing part in names
            for name in to_delete:
                inputs.pop(name)
            inputs.update(to_add)

            self._inputs = GraphInput(inputs)

        self._locs = locs

    @property
    def base(self):
        return self._base

    @property
    def inputs(self):
        return self._inputs
    
    @inputs.setter
    def inputs(self, values):
        self._setup(inputs=values, locs={})

    @property
    def locs(self):
        return self._locs

    @locs.setter
    def locs(self, values):
        self._setup(locs=values)

    def set_input(self, name, value):
        d = self._inputs.values if self._inputs is not None else {}
        d[name] = value
        self._setup(inputs=d, locs=None) # do not overwrite existing locs

    def set_loc(self, name, value):
        d = self._locs if self._locs is not None else {}
        d[name] = value
        self._setup(locs=d)

    def __getitem__(self, name):
        return self._inputs.values[name]

    def __setitem__(self, name, value):
        self.set_input(name, value)

    def find_affected_nodes(self, graph):
        """
        Finds the set of nodes affected by this intervention in a given computation graph.

        Stores its results by setting the `affected_nodes` attribute of the
        `Intervention` object.

        :param interv: intervention experiment in question
        :return: python `set` of nodes affected by this experiment
        """
        if self.inputs is None or len(self.inputs) == 0:
            return set()

        affected_nodes = set()
        def affected(node):
            # check if children are affected, use simple DFS
            is_affected = False
            for c in node.children:
                if affected(c): # we do not want short-circuiting here
                    affected_nodes.add(node.name)
                    is_affected = True
            if node.name in self.inputs:
                affected_nodes.add(node.name)
                is_affected = True
            return is_affected

        affected(graph.root)
        self.affected_nodes = affected_nodes
        return affected_nodes


class GraphNode:
    def __init__(self, *args, name=None, forward=None):
        """
        Construct a computation graph node, can be used as a decorator of a function.

        This constructor is invoked first when `@GraphNode()` decorates a function.
        When used as a decorator, the `*args` are the parameters of the decorator

        :param args: other GraphNode objects that are the children of this node
        :param name: the name of the node. If not given, this will be the name
            of the function that it decorates by default
        :param forward:
        """
        self.children = args
        self.base_cache = {} # stores results of non-intervened runs
        self.interv_cache = {} # stores results of intervened experiments
        self.name = name
        if forward:
            self.forward = forward
            if name is None:
                self.name = forward.__name__

    def __call__(self, f):
        """
        Invoked immediately after `__init__` when `@GraphNode()` decorates a function

        :param f: the function to which the decorator is attached
        :return: a new GraphNode object
        """
        self.forward = f
        if self.name is None:
            self.name = f.__name__
        return self # adding the decorator GraphNode on a function returns a GraphNode object

    def __repr__(self):
        return "XGraphNode(\"%s\")" % self.name

    def clear_caches(self):
        del self.base_cache
        del self.interv_cache
        self.base_cache = {}
        self.interv_cache = {}

    def compute(self, inputs):
        """
        Computes the output of a node
        :param inputs: Can be a GraphInput object or an Intervention object
        :return:
        """
        # check if intervention is happening in this run
        interv = None
        is_affected = False
        if isinstance(inputs, Intervention):
            interv = inputs
            inputs = interv.base
            if interv.affected_nodes is None:
                raise RuntimeError("Must find affected nodes with respect to a graph before intervening")
            is_affected = self.name in interv.affected_nodes

        # read/store values to intervened cache if this node is effected
        cache = self.interv_cache if is_affected else self.base_cache

        # check cache first if calculation results exist in cache
        res = cache.get(interv if is_affected else inputs, None)
        if res is None:
            if interv and self.name in interv.inputs:
                if self.name in interv.locs:
                    # intervene a specific location in a vector/tensor
                    res = self.base_cache.get(inputs, None)
                    if res is None:
                        raise RuntimeError("Must compute result without intervention once before intervening "
                            "(base: %s, intervention: %s)" % (interv.base, interv.inputs))
                    idx = interv.locs[self.name]
                    res[idx] = interv.inputs[self.name]
                else:
                    # replace the whole tensor
                    res = interv.inputs[self.name]
            else:
                if len(self.children) == 0:
                    # leaf
                    values = inputs[self.name]
                    if isinstance(values, list) or isinstance(values, tuple):
                        res = self.forward(*values)
                    else:
                        res = self.forward(values)
                else:
                    # non-leaf node
                    children_res = []
                    for child in self.children:
                        child_res = child.compute(inputs if interv is None else interv)
                        children_res.append(child_res)
                    res = self.forward(*children_res)

        if is_affected:
            cache[interv] = res
        else:
            cache[inputs] = res

        return res

class ComputationGraph(ABC):
    def __init__(self, root):
        self.root = root
        self.nodes = {}
        self.results_cache = {} # stores results of simple computations and interventions
        self.leaves = set()
        self.validate_graph()

    def validate_graph(self):
        """
        Validates the structure of the computational graph by doing a dfs starting from the root.
        :raise: `ValueError` if something goes wrong
        """
        # TODO: check for cycles
        def add_node(node):
            if node.name in self.nodes:
                if self.nodes[node.name] is not node:
                    raise ValueError("Two different nodes cannot have the same name!")
                else:
                    return
            self.nodes[node.name] = node
            if len(node.children) == 0:
                self.leaves.add(node)
            for child in node.children:
                add_node(child)

        add_node(self.root)

    def validate_inputs(self, inputs):
        for node in self.leaves:
            if node.name not in inputs:
                raise RuntimeError("input value not provided for leaf node %s" % node.name)

    def validate_interv(self, interv):
        """
        Validates an experiment relevant to this `ComputationGraph`
        :param interv:  intervention experiment in question
        :raise: `ValueError` if something goes wrong
        """
        self.validate_inputs(interv.base)
        for name in interv.inputs.values.keys():
            if name not in self.nodes:
                raise RuntimeError("Node in intervention experiment not found "
                                 "in computation graph: %s" % name)
            # TODO: compare compatibility between shape of value and node

    def compute(self, inputs, store_cache=True):
        res = self.results_cache.get(inputs, None)
        if not res:
            self.validate_inputs(inputs)
            res = self.root.compute(inputs)
        if store_cache:
            self.results_cache[inputs] = res
        return res

    def intervene(self, interv, store_cache=True):
        base_res = self.compute(interv.base)

        interv_res = self.results_cache.get(interv, None)
        if not interv_res:
            self.validate_interv(interv)
            interv.find_affected_nodes(self)
            interv_res = self.root.compute(interv)
            if store_cache:
                self.results_cache[interv] = interv_res

        return base_res, interv_res

    def clear_caches(self):
        def clear_cache(node):
            node.clear_caches()
            for c in node.children:
                clear_cache(c)

        clear_cache(self.root)
        del self.results_cache
        self.results_cache = {}

    def get_result(self, node_name, x):
        return self.nodes[node_name].base_cache[x]

if __name__ == "__main__":
    class MyCompGraph(ComputationGraph):
        def __init__(self):
            @GraphNode()
            def leaf1(a, b, c):
                print("leaf1 = a + b + c = %d" % (a + b + c))
                return a + b + c

            @GraphNode()
            def leaf2(d, e):
                print("leaf2 = (d + e) / 10 = %f" % ((d + e)/10))
                return (d + e) / 10

            @GraphNode(leaf1)
            def child1(x):
                print("child1 = leaf1 * 2 = %d" % (x * 2))
                return x * 2

            @GraphNode(leaf1, leaf2)
            def child2(x, y):
                print("child2 = leaf1 - leaf2 = %f" % (x-y))
                return x - y

            @GraphNode(child1, child2)
            def root(w, z):
                print("root = child1 + child2 + 1 = %f" % (w + z+1))
                return w + z + 1

            super().__init__(root)

    g = MyCompGraph()
    g.clear_caches()

    inputs = GraphInput({"leaf1": (10, 20, 30), "leaf2": (2,3)})
    in1 = Intervention(inputs, {"child1": 100})

    res = g.intervene(in1)
    print(res)

    class Graph2(ComputationGraph):
        def __init__(self):
            @GraphNode()
            def leaf1(x, y):
                return x + y

            @GraphNode(leaf1)
            def leaf2(z):
                return -1 * z

            @GraphNode(leaf2)
            def root(x):
                return x.sum()

            super().__init__(root)

    g2 = Graph2()
    g.clear_caches()

    i1 = GraphInput({"leaf1": (torch.tensor([10,20,30]), torch.tensor([1, 1, 1]))})
    in1 = Intervention(i1)
    in1["leaf2[:2]"] = torch.tensor([101, 201])

    before, after = g2.intervene(in1)
    print("Before:", before, "after:", after)

    inputs = {"leaf1": torch.tensor([300, 300]), "leaf2": torch.tensor([100])}
    locs = {"leaf1": ":2", "leaf2": 2}
    in2 = Intervention(i1, inputs=inputs, locs=locs)
    before, after = g2.intervene(in2)
    print("Before:", before, "after:", after)


