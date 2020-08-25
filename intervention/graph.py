from intervention.graph_input import GraphInput
from intervention.intervention import Intervention

from collections import deque

class ComputationGraph:
    def __init__(self, root):
        """
        Constructs a computation graph by traversing from a root
        :param root:
        """
        self.root = root
        self.nodes = {}
        self.results_cache = {}  # caches final results compute(), intervene()
        self.leaves = set()
        self.validate_graph()

    def validate_graph(self):
        """Validate the structure of the computational graph

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

    def validate_interv(self, intervention):
        """
        Validates an experiment relevant to this `ComputationGraph`
        :param intervention:  intervention experiment in question
        :raise: `RuntimeError` if something goes wrong
        """
        self.validate_inputs(intervention.base)

        if not intervention.intervention:
            raise RuntimeError("Must specify some kind of intervention!")

        for name in intervention.intervention.values.keys():
            if name not in self.nodes:
                raise RuntimeError("Node in intervention experiment not found "
                                   "in computation graph: %s" % name)
            # TODO: compare compatibility between shape of value and node

    def compute(self, inputs, store_cache=True, iterative=True):
        """
        Run forward pass through graph with a given set of inputs

        :param inputs:
        :param store_cache:
        :return:
        """
        result = self.results_cache.get(inputs, None)
        if not result:
            self.validate_inputs(inputs)
            if iterative:
                result = self._iterative_compute(inputs)
            else:
                result = self.root.compute(inputs)
        if store_cache:
            self.results_cache[inputs] = result
        return result

    def _iterative_compute(self, inputs):
        stack = deque()
        stack.append(self.root)

        while len(stack) > 0:
            curr_node = stack[-1]
            for c in curr_node.childern:
                stack.append(c)


    def intervene(self, intervention, store_cache=True):
        """
        Run intervention on computation graph.

        :param intervention:
        :param store_cache:
        :return:
        """
        base_res = self.compute(intervention.base)

        interv_res = self.results_cache.get(intervention, None)
        self.validate_interv(intervention)
        intervention.find_affected_nodes(self)
        if not interv_res:
            interv_res = self.root.compute(intervention)
            if store_cache:
                self.results_cache[intervention] = interv_res

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
                return node.base_cache[x.base]
            else:
                return node.interv_cache[x]
        else:
            raise RuntimeError(
                "get_result requires a GraphInput or Intervention "
                "object!")
