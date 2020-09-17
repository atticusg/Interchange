from intervention import ComputationGraph, GraphNode, GraphInput, Intervention, Location
from intervention.abstraction import create_possible_mappings, find_abstractions
import numpy as np

class BooleanLogicProgram(ComputationGraph):
    def __init__(self):
        @GraphNode()
        def leaf1(a):
            return a

        @GraphNode()
        def leaf2(b):
            return b

        @GraphNode()
        def leaf3(c):
            return c

        @GraphNode(leaf1,leaf2)
        def bool_intermediate(x,y):
            return np.array([float(bool(x[0]) and bool(y[0]))], dtype=np.float64)

        @GraphNode(bool_intermediate, leaf3)
        def root(w,v ):
            return np.array([float(bool(w[0]) and bool(v[0]))], dtype=np.float64)

        super().__init__(root)

class NeuralNetwork(ComputationGraph):
    def __init__(self):
        @GraphNode()
        def leaf1(a):
            return a

        @GraphNode()
        def leaf2(b):
            return b

        @GraphNode()
        def leaf3(c):
            return c

        @GraphNode(leaf1,leaf2, leaf3)
        def network_intermediate1(x,y,z):
            a = np.array([x[0],y[0],z[0]])
            a = np.squeeze(a,2)
            a = np.squeeze(a,1)
            h = np.matmul(a,np.array([[0.5,0.5,0.1],[0.5,0.5,0.1],[1,-1,1]])) + np.array([-1,-1,1])
            return h

        @GraphNode(network_intermediate1)
        def network_intermediate2(x):
            y = np.matmul(x, np.transpose(np.array([1,1,1]))) - 1.5
            return np.array([y], dtype = np.float64)

        @GraphNode(network_intermediate2)
        def root(x):
            if x[0] > 0:
                return np.array([1])
            else:
                return np.array([0])

        super().__init__(root)
def verify_intervention(mapping, low_intervention, high_intervention, result):
    intermediate_high = bool(high_intervention.base.values["leaf1"][0]) and bool(high_intervention.base.values["leaf2"][0])
    if "bool_intermediate" in high_intervention.intervention.values:
        intermediate_high = bool(high_intervention.intervention.values["bool_intermediate"][0])
    output_high = intermediate_high and bool(high_intervention.base.values["leaf3"][0])
    a = np.array([low_intervention.base.values["leaf1"][0],low_intervention.base.values["leaf2"][0], low_intervention.base.values["leaf3"][0]])
    a = np.squeeze(a,2)
    a = np.squeeze(a,1)
    h = np.matmul(a,np.array([[0.5,0.5,0.1],[0.5,0.5,0.1],[1,-1,1]])) + np.array([-1,-1,1])
    if "network_intermediate1" in low_intervention.intervention.values:
        h[mapping["bool_intermediate"]["network_intermediate1"]] = low_intervention.intervention.values["network_intermediate1"]
    y = np.matmul(h, np.transpose(np.array([2,1,1]))) - 1.5
    if "network_intermediate2" in low_intervention.intervention.values:
        y = low_intervention.intervention.values["network_intermediate2"]
    output_low = y > 0
    if (output_low == output_high) == result[0]:
        return
    print(output_low, output_high, result[0])
    print(fawefawe)

high_model= BooleanLogicProgram()
low_model = NeuralNetwork()
#for mapping in create_possible_mappings(low_model,high_model, fixed_assignments={x:{x:Location()[:]} for x in ["bool_root", "leaf1",  "leaf2", "leaf3", "leaf4"]}):
#    print(mapping)
#    print("done \n\n")

inputs = []
for x in [(np.array([a]),np.array([b]),np.array([c])) for a in [0, 1] for b in [0, 1] for c in [0, 1] ]:
    inputs.append(Intervention({"leaf1":x[0],"leaf2":x[1],"leaf3":x[2], }, dict()))
total_high_interventions = []
for x in [(np.array([a]),np.array([b]),np.array([c])) for a in [0, 1] for b in [0, 1] for c in [0, 1] ]:
    for y in [np.array([0]), np.array([1])]:
        total_high_interventions.append(Intervention({"leaf1":x[0],"leaf2":x[1],"leaf3":x[2] }, {"bool_intermediate":y}))

def input_mapping(x):
    if x[0] == 0:
        x[0] ==-1
    return x

high_model.get_result(high_model.root.name,inputs[0])
print({key:input_mapping(inputs[0].base[key]) for key in inputs[0].base.values})
low_input = Intervention({key:input_mapping(np.expand_dims(np.expand_dims(inputs[0].base[key], 1), 1)) for key in inputs[0].base.values}, dict())
print(low_input.intervention)
low_model.get_result(low_model.root.name, low_input)

fail_list = []
for result,mapping in find_abstractions(low_model, high_model, inputs,total_high_interventions,{x:{x:Location()[:]} for x in ["root", "leaf1",  "leaf2", "leaf3"]},input_mapping):
    fail = False
    for interventions in result:
        low_intervention, high_intervention = interventions
        print(mapping)
        print("low:",low_intervention.intervention.values)
        print("lowbase:", low_intervention.base.values)
        print("high:", high_intervention.intervention.values)
        print("highbase:", high_intervention.base.values)
        print("success:", result[interventions])
        print("\n\n")
        verify_intervention(mapping,low_intervention, high_intervention, result[interventions])
        if not result[interventions]:
            fail = True
            if "bool_intermediate1" in mapping["bool_intermediate"]:
                print(afwoeij)
    fail_list.append((fail,mapping))

for fail in fail_list:
    print(fail, mapping)
