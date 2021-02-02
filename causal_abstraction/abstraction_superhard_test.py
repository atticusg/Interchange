from intervention import ComputationGraph, GraphNode, Intervention, Location
from causal_abstraction.abstraction import find_abstractions
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

        @GraphNode()
        def leaf4(d):
            return d

        @GraphNode(leaf1,leaf2)
        def bool_intermediate(x,y):
            return np.array([float(bool(x[0]) and bool(y[0]))], dtype=np.float64)

        @GraphNode(bool_intermediate,leaf3)
        def bool_intermediate2(x,y):
            return np.array([float(bool(x[0]) and bool(y[0]))], dtype=np.float64)

        @GraphNode(bool_intermediate2, leaf4)
        def root(w,v):
            return np.array([float(bool(w[0]) and bool(v[0]))], dtype=np.float64)

        super().__init__(root)

class NeuralNetwork(ComputationGraph):
    def __init__(self, W, b, W2, b2, W3, b3):
        @GraphNode()
        def leaf1(a):
            return a

        @GraphNode()
        def leaf2(b):
            return b

        @GraphNode()
        def leaf3(c):
            return c

        @GraphNode()
        def leaf4(d):
            return d

        @GraphNode(leaf1,leaf2, leaf3, leaf4)
        def network_intermediate1(w, x, y, z):
            a = np.array([w[0], x[0],y[0],z[0]])
            a = a.transpose()
            h = np.matmul(a,W) + b
            h = np.tanh(h)
            return h

        @GraphNode(network_intermediate1)
        def network_intermediate2(h1):
            h2 = np.matmul(h1,W2) + b2
            h2 = np.tanh(h2)
            return h2

        @GraphNode(network_intermediate2)
        def network_intermediate3(h2):
            y = np.matmul(h2, W3) + b3
            return np.array([y], dtype = np.float64)

        @GraphNode(network_intermediate3)
        def root(x):
            if x[0] > 0:
                return np.array([1])
            else:
                return np.array([0])

        super().__init__(root)

def verify_intervention(mapping, low_intervention, high_intervention, result, W1, b1, W2, b2, W3, b3):
    intermediate_high = bool(high_intervention.base.values["leaf1"][0]) and bool(high_intervention.base.values["leaf2"][0])
    if "bool_intermediate" in high_intervention.intervention.values:
        intermediate_high = bool(high_intervention.intervention.values["bool_intermediate"][0])
    intermediate_high2 = bool(intermediate_high and bool(high_intervention.base.values["leaf3"][0]))
    if "bool_intermediate2" in high_intervention.intervention.values:
        intermediate_high2 = bool(high_intervention.intervention.values["bool_intermediate2"][0])
    output_high = intermediate_high and bool(high_intervention.base.values["leaf3"][0])
    a = np.array([low_intervention.base.values["leaf1"][0],low_intervention.base.values["leaf2"][0], low_intervention.base.values["leaf3"][0]])
    h = np.matmul(a,W1) + b1
    if "network_intermediate1" in low_intervention.intervention.values:
        if "network_intermediate1" in mapping["bool_intermediate2"]:
            h[mapping["bool_intermediate2"]["network_intermediate1"]] = low_intervention.intervention.values["network_intermediate1"]
        elif "network_intermediate1" in mapping["bool_intermediate1"]:
            h[mapping["bool_intermediate2"]["network_intermediate1"]] = low_intervention.intervention.values["network_intermediate1"]
    h2 = np.matmul(h2,W2) + b2
    h2 = np.tanh(h2)
    if "network_intermediate2" in low_intervention.intervention.values:
        if "network_intermediate2" in mapping["bool_intermediate2"]:
            h[mapping["bool_intermediate2"]["network_intermediate2"]] = low_intervention.intervention.values["network_intermediate2"]
        elif "network_intermediate2" in mapping["bool_intermediate1"]:
            h[mapping["bool_intermediate2"]["network_intermediate2"]] = low_intervention.intervention.values["network_intermediate2"]
    y = np.matmul(h2, W3) - b3
    if "network_intermediate3" in low_intervention.intervention.values:
        y = low_intervention.intervention.values["network_intermediate3"]
    output_low = y > 0
    if (output_low == output_high) == result[0]:
        return
    print(aeofija)

inputs = []
for x in [(np.array([a]),np.array([b]),np.array([c]),np.array([d])) for a in [0, 1] for b in [0, 1] for c in [0, 1] for d in [0, 1]]:
    inputs.append(Intervention({"leaf1":x[0],"leaf2":x[1],"leaf3":x[2],"leaf4":x[3], }, dict()))
total_high_interventions = []
for x in [(np.array([a]),np.array([b]),np.array([c]),np.array([d])) for a in [0, 1] for b in [0, 1] for c in [0, 1] for d in [0, 1]]:
    for y in [np.array([0]), np.array([1])]:
        total_high_interventions.append(Intervention({"leaf1":x[0],"leaf2":x[1],"leaf3":x[2],"leaf4":x[3]}, {"bool_intermediate":y}))
for x in [(np.array([a]),np.array([b]),np.array([c]),np.array([d])) for a in [0, 1] for b in [0, 1] for c in [0, 1] for d in [0, 1]]:
    for z in [np.array([0]), np.array([1])]:
        total_high_interventions.append(Intervention({"leaf1":x[0],"leaf2":x[1],"leaf3":x[2],"leaf4":x[3]}, {"bool_intermediate2":z}))
for x in [(np.array([a]),np.array([b]),np.array([c]),np.array([d])) for a in [0, 1] for b in [0, 1] for c in [0, 1] for d in [0, 1]]:
    for y in [np.array([0]), np.array([1])]:
        for z in [np.array([0]), np.array([1])]:
            total_high_interventions.append(Intervention({"leaf1":x[0],"leaf2":x[1],"leaf3":x[2],"leaf4":x[3]}, {"bool_intermediate":y,"bool_intermediate2":z}))

def verify_mapping(mapping, result, inputs,low_model):
    if len(result.keys()) != 67280:
        print(fawedwfe)
    for key in mapping["bool_intermediate"]:
        low_node = key
    for key in mapping["bool_intermediate2"]:
        low_node2 = key
    low_indices = mapping["bool_intermediate"][low_node]
    low_indices2 = mapping["bool_intermediate2"][low_node2]
    realizations = ["null"]
    realizations2 = ["null"]
    for input in inputs:
        low_input = Intervention({key:input_mapping(np.expand_dims(np.expand_dims(inputs[0].base[key], 1), 1)) for key in input.base.values}, dict())
        realizations.append(low_model.get_result(low_node, low_input).flatten()[low_indices])
        realizations2.append(low_model.get_result(low_node2, low_input).flatten()[low_indices2])
    for input in inputs:
        for realization in realizations:
            low_input = Intervention({key:input_mapping(np.expand_dims(np.expand_dims(inputs[0].base[key], 1), 1)) for key in input.base.values}, {low_node:realization})
            realizations2.append(low_model.get_result(low_node2, low_input).flatten()[low_indices2])
    pairs_to_verify = []
    for input in inputs:
        for realization in realizations:
            for realization2 in realizations2:
                pairs_to_verify.append((input, realization, realization2))
    success = set()
    for interventions in result:
        low_intervention, high_intervention = interventions
        for i, j in enumerate(pairs_to_verify):
            input, realization, realization2 = j
            if ("network_intermediate2" not in low_intervention.intervention.values and "network_intermediate1" not in low_intervention.intervention.values and realization == "null") or  (realization != "null" and np.array_equal(low_model.get_result(low_node, low_intervention).flatten()[low_indices], realization.flatten())):
                if int(low_intervention.base["leaf1"][0]) == int(input.base["leaf1"][0]):
                    if int(low_intervention.base["leaf2"][0]) == int(input.base["leaf2"][0]):
                        if int(low_intervention.base["leaf3"][0]) == int(input.base["leaf3"][0]):
                            if int(low_intervention.base["leaf4"][0]) == int(input.base["leaf4"][0]):
                                success.add(i)
    for i in range(len(pairs_to_verify)):
        if i not in success:
            print(fawefawe)

def input_mapping(x):
    if int(x[0]) == 0:
        return np.array([-1])
    return np.array([1])

first = True
for _ in range(10):
    MLPX = []
    MLPY = []
    for a in [0,1]:
        for b in [0,1]:
            for c in [0,1]:
                for d in [0,1]:
                    MLPX.append([a,b,c, d])
                    if a + b + c + d == 4:
                        MLPY.append(1)
                        for i in range(15):
                            MLPX.append([a,b,c, d])
                            MLPY.append(1)
                            MLPX.append([a,b,c, d])
                            MLPY.append(1)
                            MLPX.append([a,b,c, d])
                            MLPY.append(1)
                    else:
                        MLPY.append(0)
    MLP = MLPClassifier(hidden_layer_sizes = (2,2), activation ='tanh', batch_size=1)
    MLP.fit(MLPX,MLPY)
    print(MLP.score(MLPX,MLPY))
    W,W2, W3 = MLP.coefs_
    b,b2, b3 = MLP.intercepts_
    if first and False:
        verify_mapping(mapping, result, inputs, low_model)
        for interventions in result:
            low_intervention, high_intervention = interventions
            verify_intervention(mapping,low_intervention, high_intervention, result[interventions], W, b, W2, b2, W3, b3)
            if not result[interventions]:
                fail = True
                if "bool_intermediate1" in mapping["bool_intermediate"]:
                    print(afwoeij)

    if False:
        W = np.array([[0.5,0,0],[0.5,0,0],[0,0,1],[0,1,0]])
        b = np.array([-1,0,0])
        W2 = np.array([[0.5,0,0],[0.5,0,0],[0,0,1]])
        b2 = np.array([-1,-1,0])
        W3 = np.array([1,1,1])
        b3 = -1.5

    high_model= BooleanLogicProgram()
    low_model = NeuralNetwork(W,b,W2,b2, W3, b3)

    high_model.get_result(high_model.root.name,inputs[0])
    low_input = Intervention({key:input_mapping(np.expand_dims(np.expand_dims(inputs[0].base[key], 1), 1)) for key in inputs[0].base.values}, dict())
    low_model.get_result(low_model.root.name, low_input)

    fail_list = []
    for result,mapping in find_abstractions(low_model, high_model, inputs,total_high_interventions,{x:{x:Location()[:]} for x in ["root", "leaf1",  "leaf2", "leaf3", "leaf4"]},input_mapping):
        fail = False
        verify_mapping(mapping, result, inputs, low_model)
        for interventions in result:
            low_intervention, high_intervention = interventions
            verify_intervention(mapping,low_intervention, high_intervention, result[interventions], W, b, W2, b2, W3, b3)
            if not result[interventions]:
                fail = True
                if "bool_intermediate1" in mapping["bool_intermediate"]:
                    print(afwoeij)
        fail_list.append((fail,mapping))

    all_fails = True
    print(len(fail_list))
    for fail in fail_list:
        print(fail, mapping)
        if not fail:
            all_fails= False
    print(all_fails)
    print("\n\n\n")
