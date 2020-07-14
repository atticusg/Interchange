from abc import ABC

class Experiment:
    def __init__(self, id, inputs, interventions=None):
        self.inputs = inputs
        self.interventions = interventions
        self.id = id

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
        self.default_cache = {} # stores results of non-intervened runs
        self.intervened_cache = {} # stores results of intervened experiments
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
        del self.default_cache
        del self.intervened_cache
        self.default_cache = {}
        self.intervened_cache = {}

    def compute(self, expt, intervene=False):
        """
        Computes the output of a node
        :param expt: experiment object
        :param intervene: whether we perform an intervention as defined by this experiment
        :return:
        """
        # check if intervention is happening in this run
        is_affected = (self.name in expt.affected_nodes) if intervene else False

        # read/store values to intervened cache if this node is effected
        cache = self.intervened_cache if is_affected else self.default_cache
        print("computing node %s, am I affected? %s" % (self.name, is_affected))
        # check cache first if calculation results exist in cache
        res = cache.get(expt.id, None)
        if res is None:
            if self.name in expt.interventions and intervene:
                # this node is intervened, set result directly to intervened value
                res = expt.interventions[self.name]
            else:
                if len(self.children) == 0:
                    # leaf
                    values = expt.inputs[self.name]
                    res = self.forward(*values)
                else:
                    # non-leaf node
                    children_res = [c.compute(expt, intervene) for c in self.children]
                    res = self.forward(*children_res)

        cache[expt.id] = res
        return res

class ComputationGraph(ABC):
    def __init__(self, root):
        self.root = root
        self.nodes = {}
        self.expt_cache = {}
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
            for child in node.children:
                add_node(child)

        add_node(self.root)

    def validate_experiment(self, expt):
        """
        Validates an experiment relevant to this `ComputationGraph`
        :param expt:  intervention experiment in question
        :raise: `ValueError` if something goes wrong
        """
        for name in expt.interventions.keys():
            if name not in self.nodes:
                raise ValueError("Node in intervention experiment not found "
                                 "in computation graph: %s" % name)
            # TODO: compare compatibility between shape of value and node

    def find_affected_nodes(self, expt):
        """
        Finds the set of affected nodes for an experiment.
        Also sets the `affected_nodes` attribute of the experiment.

        :param expt: intervention experiment in question
        :return: python `set` of nodes affected by this experiment
        """
        if expt.interventions is None or len(expt.interventions) == 0:
            return None

        affected_nodes = set()
        def affected(node):
            # check if children are affected, use simple DFS
            is_affected = False
            for c in node.children:
                if affected(c): # we do not want short-circuiting here
                    affected_nodes.add(node.name)
                    is_affected = True
            if node.name in expt.interventions:
                affected_nodes.add(node.name)
                is_affected = True
            return is_affected


        affected(self.root)
        expt.affected_nodes = affected_nodes
        return affected_nodes

    def get_expt_by_id(self, eid):
        return self.expt_cache.get(eid, 0)

    def get_or_add_expt(self, expt):
        if expt.id not in self.expt_cache:
            self.validate_experiment(expt)
            self.find_affected_nodes(expt)
            self.expt_cache[expt.id] = expt
            print("added experiment %d to expt cache" % expt.id)

        print("experiment has affected nodes:", expt.affected_nodes)
        return self.expt_cache[expt.id]

    def run_expt(self, expt, intervene=False):
        expt = self.get_or_add_expt(expt)
        return self.root.compute(expt, intervene=intervene)

    def clear_caches(self):

        def clear_cache(node):
            node.clear_caches()
            for c in node.children:
                clear_cache(c)

        clear_cache(self.root)

class MyCompGraph(ComputationGraph):
    def __init__(self):
        super(MyCompGraph, self).__init__(self.root)

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

g = MyCompGraph()
g.clear_caches()

inputs = {"leaf1": (10, 20, 30), "leaf2": (2,3)}
interventions = {"child1": 100}
expt = Experiment(id=1, inputs=inputs, interventions=interventions)
res = g.run_expt(expt, intervene=False)
print(res)

print("\n-----Running intervention-----")
ires = g.run_expt(expt, intervene=True)
print(ires)
