from model import Model

class Sklearn_MLPClassifier_Model(Model):
    def __init__(self, mlp_classifier_model):
        self.model = mlp_classifier_model

        pass

    def init_vector_map(self):
        """ Here the user defines how to map strings to acceses vectors in the model """

        pass

    def get_vector(self, name):
        """ Obtain the current values of an internal vector in the model """
        pass

    def set_vector(self, name, value):
        """ Set the values of an internal vector """
        pass

    def run(self, inputs):
        """ Run the model on a set of inputs, we expect this to update *all*
        internal vectors of the model according to the new input"""
        pass

    def fix_and_run(self, names, inputs):
        """ Run the model on a set of inputs, but hold a set of internal vectors
        constant. This set is given by the list of strings in `names`"""
        pass

    def get_causal_ordering(self, name):
        """ Get a data structure of what downstream vectors are affected by
        one internal vector designated by `name`"""
        pass
