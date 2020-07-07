class Variable:
    def __init__(self, valueset, value=None):
        self.valueset = valueset
        self.used = False
        self.set_value(value)
        self.used = False

    def set_value(self,value):
        if value not in self.valueset and value is not None:
            raise ValueError('Invalid value for variable:', value)
        if self.used:
            raise ValueError('This variable has algety been used')
        self.value = value
        self.used = True

    def get_value(self):
        return self.value

class Tape:
    def __init__(self, valuesets,values=None):
        self.variable_list = []
        self.valuesets =  valuesets
        if values is not None and len(values) != len(valuesets):
            raise ValueError("Values and valuesets have different lengths")
        if values is None:
            values = [None]*len(valuesets)
        for valueset, value in zip(self.valuesets, values):
            self.variable_list.append(Variable(valueset,value))

    def get_domain(self):
        domain = set()
        for variable,value in enumerate(variable_list):
            if value is not None:
                domain.add(variable)
        return domain

    def set_value(self,variable_index, value, intervention=None):
        if variable_index > len(self.variable_list):
            raise ValueError('Variable index out of bounds.')
        if intervention is not None and variable_index in intervention:
            return
        self.variable_list[variable_index].set_value(value)

    def get_value(self,variable_index):
        if variable_index > len(self.variable_list):
            raise ValueError('Variable index out of bounds.')
        return self.variable_list[variable_index].get_value()

class CausalModel(object):
    def __init__(self, input_valuesets, variable_valuesets, output_valuesets):
        self.input_valuesets = input_valuesets
        self.output_valuesets = output_valuesets
        self.variable_valuesets = variable_valuesets
        self.causal_ordering_cache = dict()

    def initialize_tapes(self, input, intervention):
        input_tape = Tape(self.input_valuesets, input)
        variable_tape = Tape(self.variable_valuesets)
        output_tape = Tape(self.output_valuesets)
        for index in intervention:
            variable_tape.set_value(index, intervention[index])
        return input_tape, variable_tape, output_tape

    def run_model(self, input, intervention):
        pass

    def populate_cache(self, input,intervention):
        input_tape,variable_tape,output_tape,ordering =self.run_model(input,intervention)
        self.causal_ordering_cache[(input,intervention)] = ordering

    def precedes(self,input,intervention,variable1,variable2):
        if (input,intervention) not in self.causal_ordering_cache:
            self.populate_cache(input,intervention)
        if (variable1,variable2) in self.causal_ordering_cache[(input,intervention)]:
            return True
        return False

    def get_previous_variables(self, input,intervention,variable):
        result = set()
        for X in len(self.variable_valuesets):
            if self.precedes(X,variable):
                result.add(X)
        return result
