from intervention import Intervention, GraphInput
from intervention.utils import copy_helper
import torch

# TODO: add type hints

class GraphNode:
    def __init__(self, *args, name=None, forward=None, cache_results=True):
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

        # keep track which devices the results are originally stored in
        self.base_output_devices = {}
        self.interv_output_devices = {}
        self.name = name
        self.cache_results = cache_results
        if forward:
            self.forward = forward
            if name is None:
                self.name = forward.__name__

        # Saving the results in their original devices by default
        self.cache_device = None

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

    def __repr__(self):
        return "GraphNode(\"%s\")" % self.name

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

    def get_from_cache(self, inputs, from_interv):
        if not self.cache_results:
            return None

        cache = self.interv_cache if from_interv else self.base_cache
        output_device_dict = self.interv_output_devices if from_interv \
            else self.base_output_devices
        assert from_interv and isinstance(inputs, Intervention) or \
               (not from_interv) and isinstance(inputs, GraphInput)

        result = cache.get(inputs, None)

        if self.cache_device is not None and isinstance(result, torch.Tensor):
            output_device = output_device_dict[inputs]
            if output_device != self.cache_device:
                return result.to(output_device)

        return result

    def save_to_cache(self, inputs, result, to_interv):
        if not self.cache_results:
            raise RuntimeError(f"self.cache_results=False -- cannot save to cache for node {self.name}")

        cache = self.interv_cache if to_interv else self.base_cache
        output_device_dict = self.interv_output_devices if to_interv \
            else self.base_output_devices

        result_for_cache = result
        if self.cache_device is not None and isinstance(result, torch.Tensor):
            if result.device != self.cache_device:
                result_for_cache = result.to(self.cache_device)
            output_device_dict[inputs] = result.device

        cache[inputs] = result_for_cache

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
                raise RuntimeError("Must find affected nodes with respect to "
                                   "a graph before intervening")
            is_affected = self.name in intervention.affected_nodes

        # check cache first if calculation results exist in cache
        # result = cache.get(intervention if is_affected else inputs, None)

        result = self.get_from_cache(intervention if is_affected else inputs,
                                    is_affected)

        if result is not None:
            return result
        else:
            if intervention and self.name in intervention.intervention:
                if self.name in intervention.location:
                    # intervene a specific location in a vector/tensor
                    # result = self.base_cache.get(inputs, None)
                    if not self.cache_results:
                        raise RuntimeError(f"Cannot intervene on node {self.name}"
                                           "because its results are not cached")

                    result = self.get_from_cache(inputs, from_interv=False)
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
                    # if isinstance(values, list) or isinstance(values, tuple):
                    #     result = self.forward(*values)
                    # else:
                    result = self.forward(values)
                else:
                    # non-leaf node
                    children_res = []
                    for child in self.children:
                        child_res = child.compute(
                            inputs if intervention is None else intervention)
                        children_res.append(child_res)
                    result = self.forward(*children_res)

            if self.cache_results:
                self.save_to_cache(intervention if is_affected else inputs,
                                   result, is_affected)

            return result

    def clear_caches(self):
        del self.base_cache
        del self.interv_cache
        del self.base_output_devices
        del self.interv_output_devices

        self.base_cache = {}
        self.interv_cache = {}
        self.base_output_devices = {}
        self.interv_output_devices = {}
