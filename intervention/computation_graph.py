import re
import torch


class Loc:
    """A helper class to manage parsing of indices and slices"""

    def __init__(self, loc=None):
        self._loc = Loc.process(loc) if loc else None

    def __getitem__(cls, item):
        return item

    @classmethod
    def process(cls, x):
        if isinstance(x, int) or isinstance(x, list) or isinstance(x, tuple) \
                or isinstance(x, slice) or x is Ellipsis:
            return x
        elif isinstance(x, str):
            return Loc.parse_str(x)

    @classmethod
    def parse_str(cls, s):
        return tuple(Loc.parse_dim(x.strip()) for x in s.split(","))

    @classmethod
    def parse_dim(cls, s):
        return Ellipsis if s == "..." \
            else True if s == "True" \
            else False if s == "False" \
            else Loc.str_to_slice(s) if ":" in s \
            else int(s)

    @classmethod
    def str_to_slice(cls, s):
        return slice(
            *map(lambda x: int(x.strip()) if x.strip() else None, s.split(':')))


class GraphInput:
    """ A hashable input object that stores a dict mapping names of nodes to
    values of arbitrary type.

    `GraphInput` objects are intended to be immutable, so that its hash value
    can have a one-to-one correspondence to the dict stored in it. """

    def __init__(self, values, device=None):
        if device:
            assert all(
                isinstance(t, torch.Tensor) for _, t in values.items())
            self._values = {k: v.to(device) for k, v in values.items()}
        else:
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
            s = ", ".join(
                ("'%s': %s" % (k, type(v))) for k, v in self._values.items())
            return "GraphInput{%s}" % s

    def to(self, device):
        assert all(isinstance(t, torch.Tensor) for _, t in self._values.items())

        new_values = {k: v.to(device) for k, v in self._values.items()}
        return GraphInput(new_values)


class Intervention:
    """ A hashable intervention object """

    def __init__(self, base, intervention=None, locs=None, device=None):
        """ Construct an intervention experiment.

        :param base: `GraphInput` or `dict(str->Any)` containing the "base" input to a graph,
            where we intervene on the intermediate outputs of this input instance.
        :param intervention: `GraphInput` or `dict(str->Any)` denoting the node
            names and corresponding values that we want to set the nodes
        :param locs: `dict(str-><index>)` optional, indices of nodes to intervene
        :param device: cuda/cpu if intervention values are pytorch tensors
        """
        intervention = {} if intervention is None else intervention
        locs = {} if locs is None else locs
        self.device = device
        self._setup(base, intervention, locs)
        self.affected_nodes = None

    def _setup(self, base=None, intervention=None, locs=None):
        if base is not None:
            if isinstance(base, dict):
                base = GraphInput(base, self.device)
            self._base = base

        if locs is not None:
            # specifying any value that is not None will overwrite self._locs
            locs = {name: Loc.process(loc) for name, loc in locs.items()}
        else:
            locs = self._locs

        if intervention is not None:
            if isinstance(intervention, GraphInput):
                intervention = intervention.values

            # extract indexing in names
            loc_pattern = re.compile("\[.*\]")
            to_delete = []
            to_add = {}
            for name, value in intervention.items():
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
                intervention.pop(name)
            intervention.update(to_add)

            if self.device:
                intervention = {k: v.to(self.device) for k, v in
                                intervention.items()}

            self._intervention = GraphInput(intervention)

        self._locs = locs

    @property
    def base(self):
        return self._base

    @property
    def intervention(self):
        return self._intervention

    @intervention.setter
    def intervention(self, values):
        self._setup(intervention=values, locs={})

    @property
    def locs(self):
        return self._locs

    @locs.setter
    def locs(self, values):
        self._setup(locs=values)

    def set_intervention(self, name, value):
        d = self._intervention.values if self._intervention is not None else {}
        d[name] = value if not self.device else value.to(self.device)
        self._setup(intervention=d, locs=None)  # do not overwrite existing locs

    def set_loc(self, name, value):
        d = self._locs if self._locs is not None else {}
        d[name] = value
        self._setup(locs=d)

    def __getitem__(self, name):
        return self._intervention.values[name]

    def __setitem__(self, name, value):
        self.set_intervention(name, value)

    def find_affected_nodes(self, graph):
        """
        Finds the set of nodes affected by this intervention in a given computation graph.

        Stores its results by setting the `affected_nodes` attribute of the
        `Intervention` object.

        :param interv: intervention experiment in question
        :return: python `set` of nodes affected by this experiment
        """
        if self.intervention is None or len(self.intervention) == 0:
            return set()

        affected_nodes = set()

        def affected(node):
            # check if children are affected, use simple DFS
            is_affected = False
            for c in node.children:
                if affected(c):  # we do not want short-circuiting here
                    affected_nodes.add(node.name)
                    is_affected = True
            if node.name in self.intervention:
                affected_nodes.add(node.name)
                is_affected = True
            return is_affected

        affected(graph.root)
        self.affected_nodes = affected_nodes
        return affected_nodes


class GraphNode:
    def __init__(self, *args, name=None, forward=None):
        """Construct a computation graph node, can be used as function decorator

        This constructor is invoked when `@GraphNode()` decorates a function.
        When used as a decorator, the `*args` are the parameters of the decorator

        :param args: other GraphNode objects that are the children of this node
        :param name: the name of the node. If not given, this will be the name
            of the function that it decorates by default
        :param forward:
        """
        self.children = args
        self.base_cache = {}  # stores results of non-intervened runs
        self.interv_cache = {}  # stores results of intervened experiments
        self.name = name
        if forward:
            self.forward = forward
            if name is None:
                self.name = forward.__name__

    def __call__(self, f):
        """Invoked immediately after `__init__` when `@GraphNode()` decorates a function

        :param f: the function to which the decorator is attached
        :return: a new GraphNode object
        """
        self.forward = f
        if self.name is None:
            self.name = f.__name__
        # adding the decorator GraphNode on a function returns a GraphNode object
        return self

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
                raise RuntimeError(
                    "Must find affected nodes with respect to a graph before intervening")
            is_affected = self.name in interv.affected_nodes

        # read/store values to intervened cache if this node is effected
        cache = self.interv_cache if is_affected else self.base_cache

        # check cache first if calculation results exist in cache
        res = cache.get(interv if is_affected else inputs, None)

        if res is None:
            if interv and self.name in interv.intervention:
                if self.name in interv.locs:
                    # intervene a specific location in a vector/tensor
                    res = self.base_cache.get(inputs, None)
                    if res is None:
                        raise RuntimeError(
                            "Must compute result without intervention once "
                            "before intervening (base: %s, intervention: %s)"
                            % (interv.base, interv.intervention))
                    idx = interv.locs[self.name]
                    res[idx] = interv.intervention[self.name]
                else:
                    # replace the whole tensor
                    res = interv.intervention[self.name]
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
                        child_res = child.compute(
                            inputs if interv is None else interv)
                        children_res.append(child_res)
                    res = self.forward(*children_res)

        if is_affected:
            cache[interv] = res
        else:
            cache[inputs] = res

        return res


class ComputationGraph:
    def __init__(self, root):
        """
        Constructs a computation graph by traversing from a root
        :param root:
        """
        self.root = root
        self.nodes = {}
        self.results_cache = {}  # stores results of simple computations and interventions
        self.leaves = set()
        self.validate_graph()

    def validate_graph(self):
        """
        Validates the structure of the computational graph by doing a dfs starting from the root.
        :raise: `RuntimeError` if something goes wrong
        """

        # TODO: check for cycles
        def add_node(node):
            if node.name in self.nodes:
                if self.nodes[node.name] is not node:
                    raise RuntimeError(
                        "Two different nodes cannot have the same name!")
                else:
                    return
            self.nodes[node.name] = node
            if len(node.children) == 0:
                self.leaves.add(node)
            for child in node.children:
                add_node(child)

        add_node(self.root)

    def validate_inputs(self, inputs):
        """
        Check if an input is provided for each leaf node
        :raise: `RuntimeError` if something goes wrong
        """
        for node in self.leaves:
            if node.name not in inputs:
                raise RuntimeError(
                    "input value not provided for leaf node %s" % node.name)

    def validate_interv(self, interv):
        """
        Validates an experiment relevant to this `ComputationGraph`
        :param interv:  intervention experiment in question
        :raise: `RuntimeError` if something goes wrong
        """
        self.validate_inputs(interv.base)
        for name in interv.intervention.values.keys():
            if name not in self.nodes:
                raise RuntimeError("Node in intervention experiment not found "
                                   "in computation graph: %s" % name)
            # TODO: compare compatibility between shape of value and node

    def compute(self, inputs, store_cache=True):
        """
        Run forward pass through graph with a given set of inputs

        :param inputs:
        :param store_cache:
        :return:
        """
        res = self.results_cache.get(inputs, None)
        if not res:
            self.validate_inputs(inputs)
            print("check type of inputs in compute", type(inputs))
            res = self.root.compute(inputs)
        if store_cache:
            self.results_cache[inputs] = res
        return res

    def intervene(self, interv, store_cache=True):
        """
        Run intervention on computation graph.

        :param interv:
        :param store_cache:
        :return:
        """
        base_res = self.compute(interv.base)

        interv_res = self.results_cache.get(interv, None)
        self.validate_interv(interv)
        interv.find_affected_nodes(self)
        if not interv_res:
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
        node = self.nodes[node_name]
        if isinstance(x, GraphInput):
            if x not in node.base_cache:
                self.compute(x)
            return node.base_cache[x]
        elif isinstance(x, Intervention):
            if x.base not in node.base_cache:
                self.intervene(x)
            if node.name not in x.affected_nodes:
                return node.base_cache[x]
            else:
                return node.interv_cache[x]
        else:
            raise RuntimeError("get_result requires a GraphInput or Intervention "
                               "object!")

class CompGraphConstructor:
    """Automatically construct a `ComputationGraph` from a `torch.nn.Module`.

    Currently, the constructor will automatically treat the submodules of
    type `torch.nn.Module` of the provided module as nodes in the computation
    graph. It only considers one level of submoduling, and ignores nested
    submodules. It only works if the outputs of every submodule directly feeds
    into another one as is, without any intermediate steps outside the scope of a
    submodule's forward() function.
    """

    def __init__(self, module, node_modules=None):
        assert isinstance(module,
                          torch.nn.Module), "Must provide an instance of a nn.Module"

        self.module = module

        self.name_to_node = {}
        self.module_to_name = {}
        self.current_input = None

        node_modules = module.named_children() if not node_modules else node_modules

        for name, submodule in node_modules:
            # construct nodes based on children modules of module
            # no edges link these nodes yet, edges are created during construct()
            node = GraphNode(name=name, forward=submodule.forward)
            self.name_to_node[name] = node
            self.module_to_name[submodule] = name

            submodule.register_forward_pre_hook(self.pre_hook)
            submodule.register_forward_hook(self.post_hook)

    @classmethod
    def construct(cls, module, *args, device=None):
        """ Construct a computation graph given a torch.nn.Module

        We must provide an instance of an input to the torch.nn.Module to construct
        the computation graph. The intermediate output values of each node will
        be automatically stored in the nodes of the graph that is constructed.

        :param module: torch.nn.Module
        :param args: inputs to module.forward()
        :return: (ComputationGraph, GraphInput) where `g` is the constructed
            computation graph, `input_obj` is a GraphInput object based on args,
            which is required for further intervention experiments on the
            ComputationGraph.
        """
        constructor = cls(module)
        g, input_obj = constructor.make_graph(*args, device=device)
        return g, input_obj

    def pre_hook(self, module, input):
        """ Executed before the module's forward() and unpacks inputs

        We track how modules are connected by augmenting the outputs of a
        module with the its own name, which can be read when the outputs become
        the inputs of another module.

        :param module: torch.nn.Module, submodule of self.module
        :param input: tuple of (Tensor, str) tuples, Tensors are actual inputs
             to module.forward(), str denotes name of module that outputted Tensor
        :return: modified input that is actually passed to module.forward()
        """

        name = self.module_to_name[module]
        current_node = self.name_to_node[name]
        print("I am in module", self.module_to_name[module],
              "I have %d inputs" % len(input))

        if not all(isinstance(x, tuple) and len(x) == 2 for x in input):
            raise RuntimeError(
                "At least one input to \"%s\" is not an output of a named module!" % name)

        actual_inputs = tuple(t[0] for t in input)

        # get information about which modules do the inputs come from
        if any(x[1] is None for x in input):
            if not all(x[1] is None for x in input):
                raise NotImplementedError(
                    "Nodes currently don't support mixed leaf and non-leaf inputs")
            current_node.children = []
            if self.current_input is not None:
                raise NotImplementedError(
                    "Currently only supports one input leaf!")
            else:
                self.current_input = GraphInput({name: actual_inputs})
        else:
            current_node.children = [self.name_to_node[t[1]] for t in input]

        return actual_inputs

    def post_hook(self, module, input, output):
        """Executed after module.forward(), repackages outputs with name of current module

        :param module: torch.nn.Module, submodule of self.Module
        :param input: the inputs to module.forward()
        :param output: outputs from module.forward()
        :return: modified output of module, and may be passed on to next module
        """
        name = self.module_to_name[module]
        current_node = self.name_to_node[name]

        # store output info in cache
        current_node.base_cache[self.current_input] = output

        # package node name together with output
        return (output, name)

    def make_graph(self, *args, device=None):
        """ construct a computation graph given a nn.Module """
        if device:
            input = tuple((x.to(device), None) for x in args)
        else:
            input = tuple((x, None) for x in args)
        print("current_input in make_graph", self.current_input)
        res, root_name = self.module(*input)
        graph_input_obj = self.current_input
        self.current_input = None
        root = self.name_to_node[root_name]
        return ComputationGraph(root), graph_input_obj


if __name__ == "__main__":
    ##### Example 1 #####
    class MyCompGraph(ComputationGraph):
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


    print("----- Example 1  -----")

    g = MyCompGraph()
    g.clear_caches()

    interv = GraphInput({"leaf1": (10, 20, 30), "leaf2": (2, 3)})
    in1 = Intervention(interv, {"child1": 100})

    res = g.intervene(in1)
    print(res)


    ##### Example 2 ######
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


    print("----- Example 2  -----")

    g2 = Graph2()
    g.clear_caches()

    i1 = GraphInput(
        {"leaf1": (torch.tensor([10, 20, 30]), torch.tensor([1, 1, 1]))})
    in1 = Intervention(i1)
    in1["leaf2[:2]"] = torch.tensor([101, 201])

    before, after = g2.intervene(in1)
    print("Before:", before, "after:", after)

    interv = {"leaf1": torch.tensor([300, 300]), "leaf2": torch.tensor([100])}
    locs = {"leaf1": Loc()[:2], "leaf2": 2}
    in2 = Intervention(i1, intervention=interv, locs=locs)
    before, after = g2.intervene(in2)
    print("Before:", before, "after:", after)


    ##### Example 3 #####

    class TorchEqualityModule(torch.nn.Module):
        def __init__(self,
                     input_size=20,
                     hidden_layer_size=100,
                     activation="relu"):
            super(TorchEqualityModule, self).__init__()
            self.linear = torch.nn.Linear(input_size, hidden_layer_size)
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


    module = TorchEqualityModule()
    input = torch.randn(20)
    g3, in3 = CompGraphConstructor.construct(module, input)
    print("----- Example 3 -----")
    print("Nodes of graph:", ", ".join(g3.nodes))
    print("Name of root:", g3.root.name)
