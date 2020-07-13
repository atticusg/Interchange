from abc import ABC, abstractmethod

class InterventionModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def update_vector_cache(self):
        """ Here the user defines how to map strings to acceses vectors in the model """
        pass

    @abstractmethod
    def get_from_cache(self, name):
        """ Obtain the current values of an internal vector in the model """
        pass

    @abstractmethod
    def set_to_cache(self, name, value):
        """ Set the values of an internal vector """
        pass

    @abstractmethod
    def run(self, inputs):
        """ Run the model on a set of inputs, we expect this to update *all*
        internal vectors of the model according to the new input"""
        pass

    @abstractmethod
    def fix_and_run(self, names, inputs):
        """ Run the model on a set of inputs, but hold a set of internal vectors
        constant. This set is given by the list of strings in `names`.

        Currently, the best way to implement this function for pytorch modules
        that inherit from the `nn.Module` class is to re-write the forward()
        function.
        """
        pass

    @abstractmethod
    def get_causal_ordering(self, name):
        """ Get a data structure of what downstream vectors are affected by
        one internal vector designated by `name`"""
        pass

class ComputationGraphNode(ABC):
    def __init__(self, name, inputs):
        self.name = name
        self.inputs = inputs
        self.cache = None
        pass

    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def forward_with_cache():
        pass
