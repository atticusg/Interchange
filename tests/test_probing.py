import time

import torch
from torch.utils.data import DataLoader
from modeling.utils import load_model
from modeling.pretrained_bert import PretrainedBertModule

from compgraphs import MQNLI_Logic_CompGraph
from compgraphs import MQNLI_Bert_CompGraph, Abstr_MQNLI_Bert_CompGraph

from interchange.probing import Probe
from interchange.probing import ProbeTrainer
from interchange.probing import ProbingData, ProbingDataset



def test_probe_dataset():
    print("\nloading model")
    model_path = "data/models/bert-easy-best.pt"
    data_path = "data/mqnli/preprocessed/bert-easy.pt"
    lo_node_name = "bert_layer_0"
    low_model_type="bert"
    num_examples = 6400

    model, _ = load_model(PretrainedBertModule, model_path)
    device = torch.device("cuda")

    model = model.to(device)
    model.eval()

    print("loading data")
    data = torch.load(data_path)

    print("constructing compgraph")
    base_graph = MQNLI_Bert_CompGraph(model)
    intermediate_nodes = [lo_node_name]
    hi_compgraph = MQNLI_Logic_CompGraph(data)
    lo_compgraph = Abstr_MQNLI_Bert_CompGraph(base_graph, intermediate_nodes)
    lo_compgraph.set_cache_device(torch.device("cpu"))

    print("Constructing dataset for probing")
    probing_dataset = ProbingDataset(
        data.train, num_examples, hi_compgraph, lo_compgraph, lo_node_name,
        low_model_type,
        preprocessing_device=device,
        probe_correct_examples_only=True
    )

    sentence_q_dict = probing_dataset.ctrl_mapping["sentence_q"]
    print(sentence_q_dict)
    for k, v in sentence_q_dict.items():
        k_str = data.decode(list(k), return_str=True)
        print(f"{k_str}: {v.item()}")

    print(f"num_inputs {len(probing_dataset.inputs)}, num labels {len(probing_dataset.labels['sentence_q'])} "
          f"num_ctrl_labels {len(probing_dataset.ctrl_labels['sentence_q'])}")

    batch_size = 40
    collate_fn = probing_dataset.get_collate_fn("sentence_q", lo_node_name, 0, is_control=False)
    dataloader = DataLoader(probing_dataset, batch_size=batch_size, collate_fn=collate_fn)
    ctrl_collate_fn = probing_dataset.get_collate_fn("sentence_q", lo_node_name, 0, is_control=True)
    ctrl_dataloader = DataLoader(probing_dataset, batch_size=batch_size, collate_fn=ctrl_collate_fn, shuffle=False)

    ctrl_to_orig = [None] * 16
    proj_sig_remapping = torch.tensor(
        [14, 10, 15, 1, 9, 6, 13, 3, 2, 11, 8, 7, 12, 5, 4, 0])
    for i, (input_tuple, ctrl_input_tuple) in enumerate(zip(dataloader, ctrl_dataloader)):
        hidden, labels = input_tuple
        ctrl_hidden, ctrl_labels = ctrl_input_tuple
        assert hidden.shape[1] == 768
        assert torch.all(ctrl_hidden == hidden)

        for (label, ctrl_label) in zip(labels, ctrl_labels):
            label, ctrl_label = label.item(), ctrl_label.item()
            if ctrl_to_orig[label] is None:
                ctrl_to_orig[label] = ctrl_label
            assert proj_sig_remapping[label] == ctrl_label
            assert ctrl_to_orig[label] == ctrl_label

    print("ctrl_to_orig label mapping", ctrl_to_orig)
    assert len(set(ctrl_to_orig)) == len(ctrl_to_orig)

def test_train():
    print("loading model")
    model_path = "data/models/bert-easy-best.pt"
    data_path = "data/mqnli/preprocessed/bert-easy.pt"

    high_node = "obj"
    low_node = "bert_layer_2"
    low_model_type = "bert"
    low_loc = 25
    num_examples = 12800
    probe_save_dir = ""

    """
    12800
    train_accs [0.85796875, 0.85203125, 0.84890625]
    ctrl_train_accs [0.27984375, 0.287890625, 0.2803125]
    
    train_accs [0.855703125, 0.855625, 0.846328125]
    ctrl_train_accs [0.29125, 0.2865625, 0.28078125]

    
    25600
    train_accs [0.8595703125, 0.8561328125, 0.8481640625]
    ctrl_train_accs [0.2907421875, 0.2721484375, 0.2758203125]
    
    51200
    train_accs [0.85380859375, 0.8512109375, 0.85486328125]
    ctrl_train_accs [0.2816796875, 0.264609375, 0.2628125]

    """

    torch.manual_seed(39)

    model, _ = load_model(PretrainedBertModule, model_path)
    device = torch.device("cuda")

    model = model.to(device)
    model.eval()

    print("loading data")
    data = torch.load(data_path)

    print("constructing compgraph")
    base_graph = MQNLI_Bert_CompGraph(model)
    intermediate_nodes = [low_node]
    high_compgraph = MQNLI_Logic_CompGraph(data)
    low_compgraph = Abstr_MQNLI_Bert_CompGraph(base_graph, intermediate_nodes)
    low_compgraph.set_cache_device(torch.device("cpu"))

    print("obtaining hidden variables")
    probe_data = ProbingData(data, high_compgraph, low_compgraph,
                             low_node, low_model_type,
                             probe_train_num_examples=num_examples,
                             probe_train_num_dev_examples=5000)

    print("training basic probe")
    probe_input_dim = 768
    probe_dropout = 0.1

    probe_max_ranks = [12, 64, 256]
    train_accs = []
    ctrl_train_accs = []
    for probe_max_rank in probe_max_ranks:
        probe = Probe(high_node, low_node, low_loc, is_control=False,
                      probe_output_classes=interchange.probing.utils.get_num_classes(high_node),
                      probe_input_dim=probe_input_dim, probe_max_rank=probe_max_rank,
                      probe_dropout=probe_dropout)
        trainer = ProbeTrainer(probe_data, probe,
                               probe_train_batch_size=128,
                               res_save_dir=probe_save_dir,
                               probe_train_lr_patience_epochs=10,
                               device=device)
        start_time = time.time()
        ckpt = trainer.train()
        duration = time.time() - start_time
        print(f"Training took {duration:.2f}s")
        train_accs.append(ckpt['train_acc'])

        print("training control probe")

        ctrl_probe = Probe(high_node, low_node, low_loc, is_control=True,
                           probe_output_classes=interchange.probing.utils.get_num_classes(high_node),
                           probe_input_dim=probe_input_dim,
                           probe_max_rank=probe_max_rank,
                           probe_dropout=probe_dropout)
        ctrl_trainer = ProbeTrainer(probe_data, ctrl_probe,
                               probe_train_batch_size=128,
                               res_save_dir=probe_save_dir,
                               probe_train_lr_patience_epochs=10,
                               device=device)
        start_time = time.time()
        ckpt = ctrl_trainer.train()
        duration = time.time() - start_time
        print(f"Training took {duration:.2f}s")
        ctrl_train_accs.append(ckpt["train_acc"])

    print("train_accs", train_accs)
    print("ctrl_train_accs", ctrl_train_accs)


def test_load_probe():
    save_path = "probing_results/test_train/subj-bert_layer_0-3_x-1212_103748.pt"
    model_path = "data/models/bert-easy-best.pt"
    data_path = "data/mqnli/preprocessed/bert-easy.pt"

    probe = Probe.from_checkpoint(save_path)

    model, _ = load_model(PretrainedBertModule, model_path)
    device = torch.device("cuda")

    model = model.to(device)
    model.eval()

    print("loading data")
    data = torch.load(data_path)

    print("constructing compgraph")
    base_graph = MQNLI_Bert_CompGraph(model)
    intermediate_nodes = [probe.low_node]
    hi_compgraph = MQNLI_Logic_CompGraph(data)
    lo_compgraph = Abstr_MQNLI_Bert_CompGraph(base_graph, intermediate_nodes)
    lo_compgraph.set_cache_device(torch.device("cpu"))

    print("obtaining hidden variables")
    probe_data = ProbingData(data, hi_compgraph, lo_compgraph, probe.low_node,
                             probe_train_num_examples=1600,
                             probe_train_num_dev_examples=6400)

    trainer = ProbeTrainer(probe_data, probe,
                           probe_train_batch_size=128,
                           res_save_dir="",
                           probe_train_lr_patience_epochs=10,
                           device=device)

    acc, loss = trainer.eval()
    print(f"Got acc {acc:.2%} loss {loss:.4}")