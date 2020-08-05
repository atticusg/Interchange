import re

from intervention.location import Location
from intervention.graph_input import GraphInput


class Intervention:
    """ A hashable intervention object """

    def __init__(self, base, intervention=None, location=None, device=None):
        """ Construct an intervention experiment.

        :param base: `GraphInput` or `dict(str->Any)` containing the "base"
            input to a graph, where we intervene on the intermediate outputs of
            this input instance.
        :param intervention: `GraphInput` or `dict(str->Any)` denoting the node
            names and corresponding values that we want to set the nodes
        :param location: `dict(str->str or int or Loc or tuple)` optional,
            indices to intervene on part of a tensor or array
        :param device: Moves values in `intervention` to a `torch.Device`. This
            does not change the device of the base input.
        """
        intervention = {} if intervention is None else intervention
        location = {} if location is None else location
        self.device = device
        self._setup(base, intervention, location)
        self.affected_nodes = None

    def _setup(self, base=None, intervention=None, location=None):
        if base is not None:
            if isinstance(base, dict):
                base = GraphInput(base, self.device)
            self._base = base

        if location is not None:
            # specifying any value that is not None will overwrite self._locs
            location = {name: Location.process(loc) for name, loc in
                        location.items()}
        else:
            location = self._location

        if intervention is not None:
            if isinstance(intervention, GraphInput):
                intervention = intervention.values

            # extract indexing in names
            loc_pattern = re.compile(r"\[.*]")
            to_delete = []
            to_add = {}
            for name, value in intervention.items():
                # parse any index-like expressions in name
                loc_search = loc_pattern.search(name)
                if loc_search:
                    true_name = name.split("[")[0]
                    loc_str = loc_search.group().strip("[]")
                    loc = Location.parse_str(loc_str)
                    to_delete.append(name)
                    to_add[true_name] = value
                    location[true_name] = loc

            # remove indexing part in names
            for name in to_delete:
                intervention.pop(name)
            intervention.update(to_add)

            self._intervention = GraphInput(intervention, self.device)

        self._location = location
        for loc_name in self.location:
            if loc_name not in self.intervention:
                raise RuntimeWarning(
                    " %s in locs does not have an intervention value")

    @property
    def base(self):
        return self._base

    @property
    def intervention(self):
        return self._intervention

    @intervention.setter
    def intervention(self, values):
        self._setup(intervention=values, location={})

    @property
    def location(self):
        return self._location

    @location.setter
    def location(self, values):
        self._setup(location=values)

    def set_intervention(self, name, value):
        d = self._intervention.values if self._intervention is not None else {}
        d[name] = value if not self.device else value.to(self.device)
        self._setup(intervention=d,
                    location=None)  # do not overwrite existing locations

    def set_location(self, name, value):
        d = self._location if self._location is not None else {}
        d[name] = value
        self._setup(location=d)

    def __getitem__(self, name):
        return self._intervention.values[name]

    def __setitem__(self, name, value):
        self.set_intervention(name, value)

    def find_affected_nodes(self, graph):
        """Find nodes affected by this intervention in a computation graph.

        Stores its results by setting the `affected_nodes` attribute of the
        `Intervention` object.

        :param graph: `ComputationGraph` object
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
