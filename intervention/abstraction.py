import copy
def create_possible_mappings(low_model, high_model, fixed_assignments = None):
    class MappingCertificate:
        def __init__(self, partial_mapping, high_nodes,dependencies):
            self.partial_mapping = partial_mapping
            self.high_nodes = high_nodes
            self.dependencies = dependencies

        def remove_high_node(self):
            self.high_nodes = self.high_nodes[1:]

        def set_assignment_list(self):
            self.assignment_list = []
            if len(self.high_nodes) == 0:
                return
            #grab the next high-level node
            high_node = self.high_nodes[0]
            #check if this nodes assignment is fixed
            if fixed_assignments is not None and high_node in fixed_assignments:
                self.assignment_list = [(high_node,fixed_assignments[high_node])]
                return
            dependent_high_nodes = self.dependencies[high_node]
            #cycle through the potential locations in the low-level model we can map the high-level node to
            for location in low_model.get_locations([self.partial_mapping[x] for x in dependent_high_nodes]):
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


def find_abstraction(high_model, low_model, fixed_assignments):
    for mapping in create_possible_mappings(low_model, high_model, fixed_assignments):
        test_mapping(low_model, mapping)
