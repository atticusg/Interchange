from intervention.intervention import Intervention
from intervention.utils import copy_helper

class GraphNode:
    def __init__(self, *args, name=None, forward=None):
        """Construct a computation graph node, can be used as function decorator

        This constructor is invoked when `@GraphNode()` decorates a function.
        When used as a decorator, the `*args` become parameters of the decorator

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
        """Invoked immediately after `__init__` during `@GraphNode()` decoration

        :param f: the function to which the decorator is attached
        :return: a new GraphNode object
        """
        self.forward = f
        if self.name is None:
            self.name = f.__name__
        # adding the decorator GraphNode on a function returns GraphNode object
        return self

    @property
    def children(self):
        return self._children

    @children.setter
    def children(self, values):
        self._children = list(values)
        self._children_dict = {c.name: c for c in self._children}

    @property
    def children_dict(self):
        if hasattr(self, "_children_dict"):
            return self._children_dict
        else:
            self._children_dict = {c.name: c for c in self._children}
            return self._children_dict


    def __repr__(self):
        return "GraphNode(\"%s\")" % self.name

    def compute(self, inputs):
        """Compute the output of a node

        :param inputs: Can be a GraphInput object or an Intervention object
        :return:
        """
        # check if intervention is happening in this run
        intervention = None
        is_affected = False
        if isinstance(inputs, Intervention):
            intervention = inputs
            inputs = intervention.base
            if intervention.affected_nodes is None:
                raise RuntimeError(
                    "Must find affected nodes with respect to a graph "
                    "before intervening")
            is_affected = self.name in intervention.affected_nodes

        # read/store values to intervened cache if this node is effected
        cache = self.interv_cache if is_affected else self.base_cache

        # check cache first if calculation results exist in cache
        result = cache.get(intervention if is_affected else inputs, None)

        if result is None:
            if intervention and self.name in intervention.intervention:
                if self.name in intervention.location:
                    # intervene a specific location in a vector/tensor
                    result = self.base_cache.get(inputs, None)
                    if result is None:
                        raise RuntimeError(
                            "Must compute result without intervention once "
                            "before intervening (base: %s, intervention: %s)"
                            % (intervention.base, intervention.intervention))
                    result = copy_helper(result)
                    idx = intervention.location[self.name]
                    result[idx] = intervention.intervention[self.name]
                else:
                    # replace the whole tensor
                    result = intervention.intervention[self.name]
                if len(self.children) != 0:
                    for child in self.children:
                        child_res = child.compute(
                            inputs if intervention is None else intervention)
            else:
                if len(self.children) == 0:
                    # leaf
                    values = inputs[self.name]
                    if isinstance(values, list) or isinstance(values, tuple):
                        result = self.forward(*values)
                    else:
                        result = self.forward(values)
                else:
                    # non-leaf node
                    children_res = []
                    for child in self.children:
                        child_res = child.compute(
                            inputs if intervention is None else intervention)
                        children_res.append(child_res)
                    result = self.forward(*children_res)

        if is_affected:
            cache[intervention] = result
        else:
            cache[inputs] = result

        return result

    def clear_caches(self):
        del self.base_cache
        del self.interv_cache
        self.base_cache = {}
        self.interv_cache = {}
