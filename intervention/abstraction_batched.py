import torch

import intervention
from intervention.utils import serialize
from intervention.abstraction_torch import create_possible_mappings

from torch.utils.data import IterableDataset, DataLoader
from itertools import product

class InterchangeDataset(IterableDataset):
    def __init__(self, low_input_tuple_len, low_hidden_loc):
        super(InterchangeDataset, self).__init__()
        self.low_inputs = [[] for _ in range(low_input_tuple_len)]
        self.low_outputs = []
        self.low_interv_values = []
        self.high_inputs = []
        self.high_outputs = []
        self.high_interv_values = []

        self.idx_low_interv = len(self.low_inputs)
        self.idx_low_outputs = len(self.low_inputs) + 1
        self.idx_high_inputs = len(self.low_inputs) + 2
        self.idx_high_interv = len(self.low_inputs) + 3
        self.idx_high_outputs = len(self.low_inputs) + 4
        self.idx_base_i = len(self.low_inputs) + 5
        self.idx_interv_i = len(self.low_inputs) + 6

        self.low_hidden_loc = low_hidden_loc

    @property
    def num_examples(self):
        return len(self.low_outputs)

    def add_example(self, low_input_tuple, low_outputs, low_hidden_values,
                    high_inputs, high_outputs, high_hidden_values):
        for i, x in enumerate(low_input_tuple):
            self.low_inputs[i].extend(x)
        self.low_outputs.extend(low_outputs)
        self.low_interv_values.extend(low_hidden_values[self.low_hidden_loc].to(torch.device("cpu")))
        self.high_inputs.extend(high_inputs)
        self.high_outputs.extend(high_outputs)
        self.high_interv_values.extend(high_hidden_values)

    def get_intervention_tuple(self, base_i, interv_i):
        low_base_input = [x[base_i] for x in self.low_inputs]
        low_interv_value = self.low_interv_values[interv_i]
        low_base_output = self.low_outputs[base_i]

        high_base_input = self.high_inputs[base_i]
        high_interv_value = self.high_interv_values[interv_i]
        high_base_output = self.high_outputs[base_i]

        return tuple(low_base_input +
                     [low_interv_value, low_base_output, high_base_input,
                      high_interv_value, high_base_output, base_i, interv_i])

    def __iter__(self):
        for (base_i, interv_i) in product(range(self.num_examples), repeat=2):
            yield self.get_intervention_tuple(base_i, interv_i)




def test_mapping(low_model, high_model, low_model_type, dataset, num_inputs,
                 batch_size, mapping, unwanted_low_nodes):
    if low_model_type == "lstm":
        raise NotImplementedError
    elif low_model_type == "bert":
        if unwanted_low_nodes is None:
            unwanted_low_nodes = set()
        relevant_mappings = {node: loc for node, loc in mapping.items() \
                             if node not in unwanted_low_nodes}

        if len(relevant_mappings) > 1:
            raise NotImplementedError("Currently does not support more than one intermediate nodes")

        print("--- Testing mapping", relevant_mappings)

        high_node = list(relevant_mappings.keys())[0]
        low_node = list(relevant_mappings[high_node].keys())[0]
        low_loc = relevant_mappings[high_node][low_node]

        device = torch.device("cuda")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        count = 0

        icd = intervention.abstraction_batched.InterchangeDataset(5, low_loc)

        print("    Getting base outputs")
        for i, input_tuple in enumerate(dataloader):
            if count >= num_inputs: break
            high_base_key = [serialize(x) for x in input_tuple[-2]]
            high_input = intervention.GraphInput.batched(
                {"input": input_tuple[-2].T}, high_base_key)
            high_output = high_model.compute(high_input)
            high_hidden = high_model.get_result(high_node, high_input)

            low_key = [serialize(x) for x in input_tuple[0]]
            low_input = intervention.GraphInput.batched(
                {"input": [x.to(device) for x in input_tuple]}, low_key)
            low_output = low_model.compute(low_input)
            low_hidden = low_model.get_result(low_node, low_input)

            icd.add_example(low_input_tuple=input_tuple,
                            low_outputs=low_output.tolist(),
                            low_hidden_values=low_hidden,
                            high_inputs=input_tuple[-2],
                            high_outputs=high_output.tolist(),
                            high_hidden_values=high_hidden)

            count += len(high_base_key)

        res_dict = {"base_i": [], "interv_i": [],
                    "high_base_res": [], "low_base_res": [],
                    "high_interv_res": [], "low_interv_res": []}

        print("    Running interchange experiments")
        intervention_dataloader = DataLoader(icd, batch_size=16)
        count = 0
        for i, input_tuple in enumerate(intervention_dataloader):
            high_input = input_tuple[icd.idx_high_inputs]
            high_interv_value = input_tuple[icd.idx_high_interv]

            high_base_key = [serialize(x) for x in high_input]
            high_base = intervention.GraphInput.batched(
                {"input": high_input.T}, high_base_key, cache_results=False)
            high_interv_key = [(serialize(x), serialize(interv)) for x, interv in \
                               zip(high_input, high_interv_value)]
            high_intervention = intervention.Intervention.batched(
                high_base, high_interv_key,
                intervention={high_node: high_interv_value}
            )

            high_base_res, high_interv_res = \
                high_model.intervene(high_intervention, store_cache=False)

            low_input = input_tuple[0]
            low_interv_value = input_tuple[icd.idx_low_interv]
            low_base_key = [serialize(x) for x in low_input]
            low_base = intervention.GraphInput.batched(
                {"input": [x.to(device) for x in input_tuple[:5]]}, low_base_key,
                cache_results=False)
            low_interv_key = [(serialize(x), serialize(interv)) for x, interv in \
                              zip(low_input, low_interv_value)]
            low_intervention = intervention.Intervention.batched(
                low_base, low_interv_key,
                intervention={low_node: low_interv_value.to(device)},
                location={low_node: low_loc}
            )

            low_base_res, low_interv_res = \
                low_model.intervene(low_intervention, store_cache=False)

            # assert torch.all(low_base_res.to(torch.device("cpu")) == input_tuple[icd.idx_low_outputs])
            # assert torch.all(high_base_res == input_tuple[icd.idx_high_outputs])

            res_dict["base_i"].extend(input_tuple[icd.idx_base_i])
            res_dict["interv_i"].extend(input_tuple[icd.idx_interv_i])
            res_dict["low_base_res"].extend(low_base_res.tolist())
            res_dict["high_base_res"].extend(high_base_res.tolist())
            res_dict["low_interv_res"].extend(low_interv_res.tolist())
            res_dict["high_interv_res"].extend(high_interv_res.tolist())
            count += len(high_input)

        res_dict["interchange_dataset"] = icd
        res_dict["mapping"] = mapping

        return res_dict



def find_abstractions_batch(low_model, high_model, low_model_type, dataset, num_inputs, batch_size,
                            fixed_assignments, unwanted_low_nodes=None):
    print("creating possible mappings")
    mappings = create_possible_mappings(low_model, high_model, fixed_assignments,
                                        unwanted_low_nodes)

    result = []
    with torch.no_grad():
        for mapping in mappings:
            result.append(test_mapping(low_model, high_model, low_model_type,
                                       dataset, num_inputs, batch_size, mapping, unwanted_low_nodes))

    return result