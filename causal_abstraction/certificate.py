import copy

class Link:
    def __init__(self,pair,variable, value,state):
        self.pair = pair
        self.variable = variable
        self.value = value
        self.state = state

class LocalState:
    def __init__(self, vector, mask, values):
        self.vector = vector
        self.mask = mask
        self.values = values

class Pair:
    def __init__(self,input, intervention,map):
        self.input = input
        self.intervention = intervention
        self.map = map

class Certificate:
    def __init__(self, intervention_map_pairs,interventions, links, dynamical_system, causal_model):
        self.intervention_map_pairs = intervention_map_pairs
        self.interventions = interventions
        self.links = links
        self.dynamical_system = dynamical_system
        self.causal_model = causal_model
        self.set_pairlist()


    def set_pairlist(self,pair):
        if len(self.links) == 0:
            link = self.links[0]
            newintervention = copy.copy(link.pair.intervention)
            newintervention[link.variable] = link.value
            newmap = copy.copy(link.pair.map)
            newmap[link.variable].state = link.state
            input_tape,variable_tape,output_tape,variables = self.causal_model.run_model(pair.input,pair.intervention)
            self.pairlist = backtrack(variables, newmap,intervention)
        else:
            intervention = self.interventions[0]
            map = dict()
            input_tape,variable_tape,output_tape,variables = self.causal_model.run_model(intervention["input"],intervention["intermediate"])
            self.pairlist = backtrack(variables, map,intervention)

    def accept(self,variables,map):
        for x in variables:
            if X not in map:
                return False
        return True

    def reject(self,variables,map):
        for x in variables:
            if X not in map:
                return False
        return True

    def next(self, variables,map,intervention,initialize=False):
        if initialize:
            self.variable_to_local_states= dict()
            self.variable_list = []
            for X in variables.diff(map.keys()):
                if X not in self.variable_to_local_states:
                    self.variable_list.append(X)
                    previous_variables = self.causal_model.get_previous_variables(intervention["input"], intervention["intermediate"],X)
                    local_states= self.get_local_states(self.dynamical_system.vectors)
                    for Y in previous_variables:
                        if Y in map:
                            new_local_states = self.get_local_state_set(self.dynamical_system.get_later_vectors(map[Y].vector))
                            local_states = local_states.intersection(new_local_states)
                    self.variable_to_local_states[X] =  list(local_states)
        newmap = copy.copy(map)
        newmap[self.variable_list[0]] = self.variable_to_local_states[self.variable_list[0]][0]
        


    def backtrack2(self, variables,map,intervention):
        if self.reject(variables,map):
            return
        if self.accept(variables,map):
            self.log_success(variables,map)
        new_map= self.next(variables,map,intervention,True)
        while new_map is not None:
            backtrack(variables, new_map,intervention)
            new_map = self.next(variables,map,intervention)


    def remove_link(self):
        self.links = self.links[1:]

    def remove_intervention(self):
        self.intervention_input_pairs= self.intervention_input_pair[1:]

    def add_pair(self,pair,remove_intervention=False):
        if remove_intervention:
            self.remove_intervention()
        else:
            self.remove_link()
        _,variable_tape,_,variables = self.causal_model.run_model(pair.input,pair.intervention)
        for X in variables:
            for pair2 in self.intervention_map_pairs:
                if X in pair2.intervention:
                    self.links.append(Link(pair2,X, variable_tape[X],pair.map[X].state))
        self.intervention_map_pairs.add(pair)
        self.set_pairlist(pair)
        if len(self.set_pairlist) == 0:
            return None
        return self

    def next_pair(self):
        result = self.pairlist[0]
        self.pairlist = self.pairlist[1:]
        return self.pairlist

class ImplementationExperiment:
    def __init__(self, input_space, interventions, causal_model, dynamical_system):
        self.causal_model = causal_model
        self.dynamical_system = dynamical_system
        self.interventions = [{"intermediate": intervention, "input":input, "solution":self.dynamical_system.runmodel(input, intervention)} for intervention in sorted(interventions,key=lambda x: len(x.keys())) for input in input_space]
        self.links = []
        self.set_maplist()

    def root(self):
        return Certificate(set(),dict(),self.interventions,[], self.dynamical_system, self.causal_model)

    def accept(self,certificate):
        if len(certificate.interventions) == 0 and len(certificate.links) == 0:
            return True
        return False

    def reject(self,certificate):
        return False

    def next(self, certificate):
        new_certificate = copy.deepcopy(certificate)
        new_certificate = new_certificate.add_pair()
        certificate.next_pair()
        return new_certificate

    def log_success(self,certificate):
        print(certificate)
        print("success!")


    def backtrack(self,certificate):
        if self.reject(certificate):
            return
        if self.accept(certificate):
            self.log_success(certificate)
        new_certificate= self.next(certificate)
        while new_certificate is not None:
            backtrack(new_certificate)
            new_certificate = self.next(certificate)
