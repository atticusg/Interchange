import pytest
import torch
import pickle
import time
import sys
from torch.utils.data import DataLoader, Subset, Dataset, IterableDataset

import intervention
from intervention.utils import serialize

import compgraphs
from modeling.pretrained_bert import PretrainedBertModule
from train import load_model

from itertools import product

from expt_interchange import get_target_locs

data_path = "mqnli_data/mqnli_bert.pt"
model_path = "mqnli_models/bert_best.pt"

@pytest.fixture
def mqnli_bert_data():
    print("Loading data")
    return torch.load(data_path)

@pytest.fixture
def mqnli_bert_model():
    print("Loading model")
    model, _ = load_model(PretrainedBertModule, model_path, device=torch.device("cuda"))
    model.eval()
    return model

class FunkyDataset(Dataset):
    def __init__(self, low_input_tuple_len):
        super(FunkyDataset, self).__init__()
        self.low_inputs = [[] for _ in range(low_input_tuple_len)]
        self.low_outputs = []
        self.low_hidden_values = []
        self.high_inputs = []
        self.high_outputs = []
        self.high_hidden_values = []

        self.idx_low_outputs = len(self.low_inputs)
        self.idx_low_hidden = len(self.low_inputs) + 1
        self.idx_high_inputs = len(self.low_inputs) + 2
        self.idx_high_outputs = len(self.low_inputs) + 3
        self.idx_high_hidden = len(self.low_inputs) + 4

    def __len__(self):
        return len(self.low_outputs)

    def __getitem__(self, i):
        return tuple(field[i] for field in self.low_inputs +
        [self.low_outputs, self.low_hidden_values,
         self.high_inputs, self.high_outputs,
         self.high_hidden_values])

    def print_all(self):
        for i in range(len(self.low_inputs)):
            print(f"self.low_inputs[{i}]", len(self.low_inputs[i]))
            one_input = self.low_inputs[i][0]
            if isinstance(one_input, torch.Tensor):
                print("    shape:", one_input.shape)
        print("self.low_outputs", len(self.low_outputs))
        print("self.low_hidden_values", len(self.low_hidden_values))
        print("self.high_inputs", len(self.high_inputs))
        print("self.high_outputs", len(self.high_outputs))

def test_base_run(mqnli_bert_data, mqnli_bert_model):
    num_inputs = 128
    batch_size = 16

    high_node = "subj_adj"
    low_node = "bert_layer_2"
    low_loc = intervention.LOC[:,3,:]


    print("Setting up models and data")
    high_model = compgraphs.MQNLI_Logic_CompGraph(mqnli_bert_data, [high_node])
    low_model = compgraphs.MQNLI_Bert_CompGraph(mqnli_bert_model)
    low_model = compgraphs.Abstr_MQNLI_Bert_CompGraph(low_model, [low_node])

    device = torch.device("cuda")
    dataloader = DataLoader(mqnli_bert_data.dev, batch_size=batch_size, shuffle=False)

    count = 0

    funky = FunkyDataset(5)

    with torch.no_grad():
        for i, input_tuple in enumerate(dataloader):
            if count >= num_inputs: break
            high_key = [serialize(x) for x in input_tuple[-2]]
            high_input = intervention.GraphInput.batched(
                {"input": input_tuple[-2].T}, high_key)
            high_output = high_model.compute(high_input)
            high_hidden = high_model.get_result(high_node, high_input)

            funky.high_inputs.extend(input_tuple[-2])
            funky.high_outputs.extend(high_output.tolist())
            funky.high_hidden_values.extend(high_hidden)

            low_key = [serialize(x) for x in input_tuple[0]]
            low_input = intervention.GraphInput.batched(
                {"input": [x.to(device) for x in input_tuple]}, low_key)
            low_output = low_model.compute(low_input)
            low_hidden = low_model.get_result(low_node, low_input)

            for i, x in enumerate(input_tuple):
                funky.low_inputs[i].extend(x)
            funky.low_outputs.extend(low_output.tolist())
            funky.low_hidden_values.extend(low_hidden)

            count += len(high_key)

    funky.print_all()
    print("len of icd", len(funky))
    funky_idxs = [1,1,2,3,5,8,13,21,22,24,25,26,30,39,55,67,100]
    funky_subset = Subset(funky, funky_idxs)
    funky_dataloader = DataLoader(funky_subset, batch_size=8, shuffle=False)

    with torch.no_grad():
        for i, input_tuple in enumerate(funky_dataloader):
            curr_high_key = [serialize(x) for x in input_tuple[3]]
            curr_high_input = input_tuple[funky.idx_high_inputs].T
            curr_high_input = intervention.GraphInput.batched(
                {"input": curr_high_input}, curr_high_key)
            curr_high_output = high_model.compute(curr_high_input)
            curr_high_hidden = high_model.get_result(high_node, curr_high_input)
            assert torch.all(curr_high_output == input_tuple[funky.idx_high_outputs])
            assert torch.all(curr_high_hidden == input_tuple[funky.idx_high_hidden])

            curr_low_key = [serialize(x) for x in input_tuple[0]]
            curr_low_input = intervention.GraphInput.batched(
                {"input": [x.to(device) for x in input_tuple[:5]]}, curr_low_key)
            curr_low_output = low_model.compute(curr_low_input).to(torch.device("cpu"))
            curr_low_hidden = low_model.get_result(low_node, curr_low_input)
            assert torch.all(curr_low_output == input_tuple[funky.idx_low_outputs])
            assert torch.all(curr_low_hidden == input_tuple[funky.idx_low_hidden])

            # for j, x in enumerate(input_tuple):
            #     print(f"input_tuple[{j}]-----------")
            #     if isinstance(x, torch.Tensor):
            #         print("   Tensor with shape", x.shape)
            #     else:
            #         print(x)

def test_intervention():
    high_node = "sentence_q"
    low_node = "bert_layer_2"
    get_and_save_results = False
    num_inputs = 100
    batch_size = 20

    save_path = "experiment_data/bert/test-batch-sentence_q.pkl"
    if get_and_save_results:
        interv_info = {
            "target_locs": get_target_locs(high_node, data_variant="bert")
        }
        mqnli_bert_data = torch.load("mqnli_data/mqnli_bert.pt")
        mqnli_bert_model, _ = load_model(PretrainedBertModule,
                                         "mqnli_models/bert_best.pt",
                                         device=torch.device("cuda"))
        mqnli_bert_model.eval()
        high_model = compgraphs.MQNLI_Logic_CompGraph(mqnli_bert_data,
                                                      [high_node])
        low_model = compgraphs.MQNLI_Bert_CompGraph(mqnli_bert_model)
        low_model = compgraphs.Abstr_MQNLI_Bert_CompGraph(low_model, [low_node],
                                                          interv_info=interv_info)
        start_time = time.time()
        batch_results = intervention.abstraction_batched.find_abstractions_batch(
            low_model=low_model,
            high_model=high_model,
            low_model_type="bert",
            dataset=mqnli_bert_data.dev,
            num_inputs=num_inputs,
            batch_size=batch_size,
            fixed_assignments={x: {x: intervention.LOC[:]} for x in
                               ["root", "input"]},
            unwanted_low_nodes={"root", "input"}
        )
        duration = time.time() - start_time
        print(f"Interchange experiment took {duration:.2f}s")
        print(f"Saving to pickled file {save_path}")
        torch.save(batch_results, save_path)
        # with open(save_path, "wb") as f:
        #     pickle.dump(batch_results, f)
    else:
        print("Loading batch results from pickled file")
        # with open(save_path, "rb") as f:
        #     batch_results = pickle.load(f)
        batch_results = torch.load(save_path)

    assert len(batch_results) == 4
    for results in batch_results:
        for k in results.keys():
            if k not in {"mapping", "interchange_dataset"}:
                assert len(results[k]) == num_inputs ** 2

    # verify data in res dict
    print("Loading old result pickle to verify")
    data_path = "experiment_data/bert/expt-3-1110-143523/res-id3-sentence_q-1110-153617.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    print("verify between old and new")
    for (((results, realizations_to_inputs), mapping), res_dict) in zip(data, batch_results):
        print(len(realizations_to_inputs))
        verify_one_mapping(results, realizations_to_inputs, mapping, res_dict, low_node)


def verify_one_mapping(results, realizations_to_inputs, mapping, res_dict, low_node):
    icd = res_dict["interchange_dataset"]
    low_interv_to_idx, low_input_to_interv_value = get_dicts(icd)
    interv_mapping = get_local_interv_values(realizations_to_inputs, low_input_to_interv_value)

    equal_count = 0
    total_count = 0
    for (low_interv, high_interv), value in results.items():
        if len(low_interv.intervention.values) == 0 or len(high_interv.intervention.values) == 0:
            continue
        low_input = low_interv.base["input"][0].squeeze()
        low_interv_value = serialize(low_interv.intervention[low_node])
        low_interv_value = interv_mapping[low_interv_value]
        local_idx = low_interv_to_idx[(serialize(low_input), serialize(low_interv_value))]
        local_eq = res_dict["high_interv_res"][local_idx] == res_dict["low_interv_res"][local_idx]
        if local_eq == value:
            equal_count += 1
        total_count += 1

    print(f"{equal_count}/{total_count}={equal_count/total_count:.2%} of interventions are equal")



def get_local_interv_values(realizations_to_inputs, low_input_to_interv_value):
    interv_mapping = {}
    for serialized_interv_val, high_node in realizations_to_inputs:
        key = (serialized_interv_val, high_node)
        low_base_input = realizations_to_inputs[key].base["input"][0].squeeze()
        local_interv_value = low_input_to_interv_value[serialize(low_base_input)]
        interv_mapping[serialized_interv_val] = local_interv_value
    return interv_mapping

def get_dicts(icd):
    interv_key_to_idx = {}
    for i, (base_i, interv_i) in enumerate(product(range(icd.num_examples), repeat=2)):
        low_inputs = icd.low_inputs[0][base_i]
        low_interv_values = icd.low_interv_values[interv_i]
        key = (serialize(low_inputs), serialize(low_interv_values))
        interv_key_to_idx[key] = i

    low_input_to_interv_value = {}
    for base_i in range(icd.num_examples):
        low_input_to_interv_value[serialize(icd.low_inputs[0][base_i])] \
            = icd.low_interv_values[base_i]

    return interv_key_to_idx, low_input_to_interv_value

def test_pickle_size_ablation():
    load_path = "experiment_data/bert/test-batch-sentence_q.pkl"
    with open(load_path, "rb") as f:
        batch_results = pickle.load(f)

    keep_fields = {"interchange_dataset", "low_base_res", "high_base_res", "low_interv_res", "high_interv_res"}
    new_res = []
    for results in batch_results:
        d = {}
        for k in results:
            if k in keep_fields:
                d[k] = results[k]
        new_res.append(d)
    save_path = "experiment_data/bert/test-batch-size-ablation.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(new_res, f)

def test_check_size():
    load_path = "experiment_data/bert/test-batch-sentence_q.pkl"
    with open(load_path, "rb") as f:
        batch_results = pickle.load(f)

    for i, d in enumerate(batch_results):
        size_d, total = size_of_res_dict(d)
        print(f"--- Mapping {i}")
        print(f"    total: {total / 1024 :.2f} kb")
        print(f"    breakdown: {size_d}")

def size_of_res_dict(d):
    size_dict = {}
    total = 0
    for k, v in d.items():
        if k == "interchange_dataset":
            icd = v
            subtotal = 0
            for l in icd.low_inputs + [icd.low_interv_values, icd.low_outputs,
                                       icd.high_interv_values, icd.high_inputs,
                                       icd.high_outputs, ]:
                subtotal += size_of_list_of_tensors(l)
        else:
            subtotal = sys.getsizeof(v)

        size_dict[k] = subtotal / 1024
        total += subtotal
    return size_dict, total


def size_of_list_of_tensors(l):
    size = sys.getsizeof(l)
    for t in l:
        if isinstance(t, torch.Tensor):
            size += t.element_size() * t.nelement()
    return size


class ToyInterventionDataset(IterableDataset):
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        super(ToyInterventionDataset, self).__init__()

    def __iter__(self):
        for (i1, i2) in product(range(len(self.data1)), repeat=2):
            yield self.data1[i1] + self.data2[i2]


def test_toy_iterator():
    data1 = ["A", "B", "C"]
    data2 = ["1", "2", "3"]

    tid = ToyInterventionDataset(data1, data2)
    dataloader = DataLoader(tid)

    for x in dataloader:
        print(x)