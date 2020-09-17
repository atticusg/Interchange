from intervention import ComputationGraph, GraphNode, GraphInput, Intervention, Location
from intervention.abstraction import create_possible_mappings, find_abstractions
import numpy as np
from sklearn.neural_network import MLPClassifier

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
    def __init__(self, W, b, W2, b2):
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
            h = np.matmul(a,W) + b
            return h

        @GraphNode(network_intermediate1)
        def network_intermediate2(x):
            y = np.matmul(x, W2) - b2
            return np.array([y], dtype = np.float64)

        @GraphNode(network_intermediate2)
        def root(x):
            if x[0] > 0:
                return np.array([1])
            else:
                return np.array([0])

        super().__init__(root)

def verify_intervention(mapping, low_intervention, high_intervention, result, W1, b1, W2, b2):
    intermediate_high = bool(high_intervention.base.values["leaf1"][0]) and bool(high_intervention.base.values["leaf2"][0])
    if "bool_intermediate" in high_intervention.intervention.values:
        intermediate_high = bool(high_intervention.intervention.values["bool_intermediate"][0])
    output_high = intermediate_high and bool(high_intervention.base.values["leaf3"][0])
    a = np.array([low_intervention.base.values["leaf1"][0],low_intervention.base.values["leaf2"][0], low_intervention.base.values["leaf3"][0]])
    a = np.squeeze(a,2)
    a = np.squeeze(a,1)
    h = np.matmul(a,W1) + b1
    if "network_intermediate1" in low_intervention.intervention.values:
        h[mapping["bool_intermediate"]["network_intermediate1"]] = low_intervention.intervention.values["network_intermediate1"]
    y = np.matmul(h, W2) - b2
    if "network_intermediate2" in low_intervention.intervention.values:
        y = low_intervention.intervention.values["network_intermediate2"]
    output_low = y > 0
    if (output_low == output_high) == result[0]:
        return
    print(aeofija)

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


for _ in range(10):
    MLPX = []
    MLPY = []
    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                MLPX.append([a,b,c])
                if a + b + c > 0:
                    MLPY.append(1)
                else:
                    MLPY.append(0)
    MLP = MLPClassifier(hidden_layer_sizes = (2,), activation ='relu', batch_size=1)
    MLP.fit(MLPX,MLPY)
    print(MLP.score(MLPX,MLPY))
    W,W2 = MLP.coefs_
    b,b2 = MLP.intercepts_

    high_model= BooleanLogicProgram()
    low_model = NeuralNetwork(W,b,W2,b2)

    high_model.get_result(high_model.root.name,inputs[0])
    low_input = Intervention({key:input_mapping(np.expand_dims(np.expand_dims(inputs[0].base[key], 1), 1)) for key in inputs[0].base.values}, dict())
    low_model.get_result(low_model.root.name, low_input)

    fail_list = []
    for result,mapping in  find_abstractions(low_model, high_model, inputs,total_high_interventions,{x:{x:Location()[:]} for x in ["root", "leaf1",  "leaf2", "leaf3"]},input_mapping):
        fail = False
        for interventions in result:
            low_intervention, high_intervention = interventions
            if not result[interventions]:
                fail = True
                if "bool_intermediate1" in mapping["bool_intermediate"]:
                    print(afwoeij)
        fail_list.append((fail,mapping))

    all_fails = True
    for fail in fail_list:
        print(fail, mapping)
        if not fail:
            all_fails= False
    print(all_fails)
    print("\n\n\n")
