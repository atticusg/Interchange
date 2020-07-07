class Dynamical_System(object):
    # interventions is a map from strings in self.vectors to a mask and replacement values
    # Should have a bunch of statements of the form:
    # self.hidden_rep= torch.matmul(x,W)
    # if "hidden_rep" in interventions:
    #     mask, replacement_values = interventions["hidden_rep"]
    #     self.intervene(self.hidden_rep, mask, replacement_values)
    def __init__(self, vectors, vector_lengths, causal_ordering, model):
        self.vectors= vectors
        self.vector_lengths = vectors_lengths
        self.causal_ordering =causal_ordering
        self.model = model
        self.vector_cache = dict()
        self.output_cache = dict()

    def vector_length(self,vector):
        return self.vector_lengths[vector]

    def precedes(self,vector1,vector2):
        if (vector1,vector2) in causal_ordering:
            return True
        return False

    def get_later_vectors(self, vector):
        result = set()
        for vector2 in self.vectors:
            if self.precedes(vector, vector2)
                result.add(vector2)
        return result 

    def intervene(self,vector, mask, values):
        vector[mask] = values
        return vector

    def get_vector_value(self,vector, input, interventions=None):
        if (input, interventions) in self.vector_cache:
            return self.vector_cache[(input, interventions)][vector]
        else:
            vectors, output = self.model.get_vectors_and_output(input,interventions)
            self.vector_cache[(input,interventions)] =vectors
            self.output_cache[(input,interventions)] =output
            return self.vector_cache[( input, interventions)][vector]

    def get_output(self, input, interventions=None):
        if (input, interventions) in self.output_cache:
            return self.output_cache[(input, interventions)]
        else:
            vectors, output = self.model.get_vectors_and_output(input,interventions)
            self.vector_cache[(input,interventions)] =vectors
            self.output_cache[(input,interventions)] =output
            return self.output_cache[(input, interventions)]
        self.model.run()
