from abc import ABC


class Experiment:
    def __init__(self, id, inputs, interventions=None):
        self.inputs = inputs
        self.interventions = interventions
        self.input_key = id

class GraphInput:
    """ A hashable input object that stores a dict mapping name of nodes in
    string format to arbitrary type. The dict cannot be changed once defined"""
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
    def __init__(self, base_input, interv_input):
        if isinstance(base_input, dict):
            base_input = GraphInput(base_input)
        if isinstance(interv_input, dict):
            interv_input = GraphInput(interv_input)
        self._base_input = base_input
        self._interv_input = interv_input
        self.affected_nodes = None

    @property
    def base_input(self):
        return self._base_input

    @property
    def interv_input(self):
        return self._interv_input

    def find_affected_nodes(self, graph):
        """
        Finds the set of affected nodes for an experiment.
        Also sets the `affected_nodes` attribute of the experiment.

        :param interv: intervention experiment in question
        :return: python `set` of nodes affected by this experiment
        """
        if self.interv_input is None or len(self.interv_input) == 0:
            return None

        affected_nodes = set()
        def affected(node):
            # check if children are affected, use simple DFS
            is_affected = False
            for c in node.children:
                if affected(c): # we do not want short-circuiting here
                    affected_nodes.add(node.name)
                    is_affected = True
            if node.name in self.interv_input:
                affected_nodes.add(node.name)
                is_affected = True
            return is_affected

        affected(graph.root)
        self.affected_nodes = affected_nodes
        return affected_nodes


class GraphNode:
    def __init__(self, *args, name=None, forward=None):
        """
        Construct a computation graph node, can be used as a decorator of a
        function.

        This constructor is immediately invoked when the `@GraphNode()`
        decorator is attached to a function.
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
        This method is invoked immediately after the constructor when the `@GraphNode` decorator is attached to a function
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
            inputs = interv.base_input

            if interv.affected_nodes is None:
                raise RuntimeError("Must find affected nodes with respect to a graph before intervening")
            if inputs not in self.base_cache:
                raise RuntimeError("Must compute result without intervention once before intervening "
                                   "(base_input: %s)" % interv.base_input)
            is_affected = self.name in interv.affected_nodes

        # read/store values to intervened cache if this node is effected
        cache = self.interv_cache if is_affected else self.base_cache

        # check cache first if calculation results exist in cache
        res = cache.get(inputs, None)
        if res is None:
            if interv and self.name in interv.interv_input:
                # this node is intervened, set result directly to intervened value
                # TODO: intervene in specific indices of an array-like object
                res = interv.interv_input[self.name]
            else:
                if len(self.children) == 0:
                    # leaf
                    values = inputs[self.name]
                    res = self.forward(*values)
                else:
                    # non-leaf node
                    children_res = [c.compute(inputs if interv is None else interv)
                                    for c in self.children]
                    res = self.forward(*children_res)

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
        print(inputs)
        for node in self.leaves:
            if node.name not in inputs:
                raise RuntimeError("input value not provided for leaf node %s" % node.name)

    def validate_interv(self, interv):
        """
        Validates an experiment relevant to this `ComputationGraph`
        :param interv:  intervention experiment in question
        :raise: `ValueError` if something goes wrong
        """
        self.validate_inputs(interv.base_input)
        for name in interv.interv_input.values.keys():
            if name not in self.nodes:
                raise RuntimeError("Node in intervention experiment not found "
                                 "in computation graph: %s" % name)
            # TODO: compare compatibility between shape of value and node

    def compute(self, inputs, store_cache=True):
        res = self.results_cache.get(interv.base_input, None)
        if not res:
            self.validate_inputs(inputs)
            res = self.root.compute(inputs)
        if store_cache:
            self.results_cache[inputs] = res
        return res

    def intervene(self, interv, store_cache=True):
        base_res = self.compute(interv.base_input)

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
interv = Intervention(inputs, {"child1": 100})

res = g.intervene(interv)
print(res)

