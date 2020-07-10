from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def init_vector_map(self):
        """ Here the user defines how to map strings to acceses vectors in the model """
        pass

    @abstractmethod
    def get_vector(self, name):
        """ Obtain the current values of an internal vector in the model """
        pass

    @abstractmethod
    def set_vector(self, name, value):
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
        constant. This set is given by the list of strings in `names`"""
        pass
