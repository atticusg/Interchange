import copy
import numpy as np
from intervention.graph_input import GraphInput
from intervention.intervention import Intervention
from intervention.location import Location

def create_possible_mappings(low_model, high_model, fixed_assignments = dict()):
    class MappingCertificate:
        def __init__(self, partial_mapping, high_nodes,dependencies):
            self.partial_mapping = partial_mapping
            self.high_nodes = [x  for x in high_nodes if x not in fixed_assignments]
            self.dependencies = dependencies
            for high_node in fixed_assignments:
                partial_mapping[high_node] = fixed_assignments[high_node]

        def remove_high_node(self):
            self.high_nodes = self.high_nodes[1:]

        def compatible_splits(self,split1, split2):
            return False

        def compatible_location(self, location):
            for high_node in self.partial_mapping:
                for low_node in self.partial_mapping[high_node]:
                    if low_node in location:
                        if not self.compatible_splits(self.partial_mapping[high_node][low_node], location[low_node]):
                            return False
            return True

        def set_assignment_list(self):
            self.assignment_list = []
            if len(self.high_nodes) == 0:
                return
            #grab the next high-level node
            high_node = self.high_nodes[0]
            dependent_high_nodes = self.dependencies[high_node]
            #cycle through the potential locations in the low-level model we can map the high-level node to
            for location in low_model.get_locations([self.partial_mapping[x] for x in dependent_high_nodes]):
                    if self.compatible_location(location):
                        self.assignment_list.append((high_node, location))

        def add_assignment(self):
            #add a new assignment to the partially constructed mapping
            self.remove_high_node()
            high_node,low_location = self.assignment_list[0]
            self.partial_mapping[high_node] = low_location
            if len(self.high_nodes) != 0:
                self.set_assignment_list()

        def next_assignment(self):
            #move on to the next assignment
            self.assignment_list = self.assignment_list[1:]

    mappings = []

    def accept(certificate):
        if len(certificate.high_nodes) == 0:
            return True
        return False

    def next(certificate):
        if len(certificate.assignment_list) == 0:
            return None
        new_certificate = copy.deepcopy(certificate)
        #Add in a assignmen to the mapping
        new_certificate.add_assignment()
        #Cycle the original map to the next assignment that could have been added
        certificate.next_assignment()
        #return the partial map
        return new_certificate

    def root():
        high_nodes, dependencies = high_model.get_nodes_and_dependencies()
        certificate = MappingCertificate(dict(), high_nodes, dependencies)
        certificate.set_assignment_list()
        return certificate

    def backtrack(certificate):
        if accept(certificate):
            mappings.append(certificate.partial_mapping)
            return
        next_certificate = next(certificate)
        while next_certificate is not None:
            backtrack(next_certificate)
            next_certificate = next(certificate)
    backtrack(root())
    return mappings

def get_value(high_model, high_node, high_intervention):
    return np.matrix(high_model.get_result(high_node, high_intervention), dtype=np.float64)

def create_new_realizations(low_model, high_model, high_node, mapping, low_intervention, high_intervention):
    new_realizations = dict()
    def fill_new_realizations(high_node, mapping, low_intervention, high_intervention):
        high_value = get_value(high_model, high_node, high_intervention)
        low_value = None
        for low_node in mapping[high_node]:
            low_value = low_model.get_result(low_node,low_intervention)[mapping[high_node][low_node]]
        for child in high_model.nodes[high_node].children:
            fill_new_realizations(child.name, mapping, low_intervention, high_intervention)
        if high_node in high_intervention.intervention.values or high_model.nodes[high_node] in high_model.leaves or high_node == high_model.root.name:
            return
        new_realizations[(high_node, high_value.tostring())] = low_value.tostring()

    fill_new_realizations(high_node, mapping, low_intervention, high_intervention)
    return new_realizations


def get_potential_realizations(new_realizations, total_realizations, high_node, high_model, new_high_intervention):
    partial_realizations = [dict()]
    high_value = get_value(high_model, high_node, new_high_intervention).tostring()
    for high_node2 in high_model.nodes:
        high_value2 = get_value(high_model, high_node2, new_high_intervention).tostring()
        if high_model.nodes[high_node2] in high_model.leaves or high_node2 == high_model.root.name or high_value != high_value2:
            continue
        if high_node2 == high_node:
            new_partial_realizations = []
            for partial_realization in partial_realizations:
                if (high_node, high_value) not in new_realizations:
                    return []
                partial_realization[(high_node, high_value)] = new_realizations[(high_node, high_value)]
                new_partial_realizations.append(partial_realization)
            partial_realizations = new_partial_realizations
        else:
            new_partial_realizations = []
            for partial_realization in partial_realizations:
                if (high_node2,high_value2) in new_realizations:
                    partial_realization_copy = copy.deepcopy(partial_realization)
                    partial_realization_copy[(high_node2, high_value2)] = new_realizations[(high_node2, high_value2)]
                    new_partial_realizations.append(partial_realization_copy)
                else:
                    for low_value in total_realizations[(high_node2, high_value2)]:
                        partial_realization_copy = copy.deepcopy(partial_realization)
                        partial_realization_copy[(high_node2, high_value2)] = low_value
                        new_partial_realizations.append(partial_realization_copy)
            partial_realizations = new_partial_realizations
    return partial_realizations


def high_to_low(high_model, high_intervention,realization, mapping, input_mapping):
    intervention = dict()
    location = dict()
    base = dict()
    for high_node in high_model.leaves:
        for low_node in mapping[high_node.name]:
            base[low_node] = input_mapping(get_value(high_model, high_node.name, high_intervention))
    for high_node in high_intervention.intervention.values:
        high_value = get_value(high_model, high_node, high_intervention).tostring()
        for low_node in mapping[high_node]:
            intervention[low_node] = np.array(np.fromstring(realization[(high_node, high_value)]), dtype=np.float64)
            location[low_node] = mapping[high_node][low_node]
    return Intervention(base,intervention,location)

def test_mapping(low_model,high_model,high_inputs,total_high_interventions,mapping, input_mapping):
    low_and_high_interventions = [(high_to_low(high_model, high_intervention, dict(), mapping, input_mapping), high_intervention) for high_intervention in high_inputs]
    total_realizations = dict()
    result = dict()
    while len(low_and_high_interventions) !=0:
        curr_low_intervention,curr_high_intervention = low_and_high_interventions[0]
        low_and_high_interventions = low_and_high_interventions[1:]
        #store whether the output matches
        high_output = get_value(high_model, high_model.root.name, curr_high_intervention)
        for low_node in mapping[high_model.root.name]:
            low_output = low_model.get_result(low_node,curr_low_intervention)
        result[(curr_low_intervention, curr_high_intervention)] = low_output == high_output
        #update the realizations
        new_realizations = create_new_realizations(low_model, high_model,high_model.root.name, mapping, curr_low_intervention, curr_high_intervention)
        #add on the new interventions that need to be checked
        for new_high_intervention in total_high_interventions:
            for high_node, high_value in new_realizations:
                if high_node in new_high_intervention.intervention:
                    for realization in get_potential_realizations( new_realizations, total_realizations, high_node, high_model, new_high_intervention):
                        new_low_intervention = high_to_low(high_model,new_high_intervention, realization,mapping, input_mapping)
                        low_and_high_interventions.append((new_low_intervention, new_high_intervention))
        #merge the new_realizations into the total realizations
        for high_node,high_value in new_realizations:
            if (high_node, high_value) in total_realizations:
                total_realizations[(high_node, high_value)].append(new_realizations[(high_node, high_value)])
            else:
                total_realizations[(high_node, high_value)] = [new_realizations[(high_node, high_value)]]
            total_realizations[(high_node, high_value)] = list(set(total_realizations[(high_node, high_value)]))
    return result



def find_abstractions(low_model, high_model, high_inputs, total_high_interventions, fixed_assignments, input_mapping):
    result = []
    for mapping in create_possible_mappings(low_model, high_model, fixed_assignments):
        result.append((test_mapping(low_model, high_model, high_inputs,total_high_interventions, mapping, input_mapping),mapping))
    return result