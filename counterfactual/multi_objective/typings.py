from dataclasses import dataclass
from typing import Tuple, Sequence
from typing import TypedDict

import torch
import antra
from antra.interchange.mapping import AbstractionMapping
from counterfactual.multi_objective import VOCAB_PATH


@dataclass
class MultiObjectiveTrainingConfig:
    data_path: str = ""
    tokenizer_vocab_path: str = VOCAB_PATH
    output_classes: int = 3
    device: str = "cuda"

    train_batch_size: int = 32
    eval_batch_size: int = 64

    optimizer_type: str = "adamw"
    lr: float = 5e-5
    lr_scheduler_type: str = "linear"
    lr_warmup_epoch_ratio: float = 0.5   # percentage/ratio of steps in an epoch to perform lr warmup
    weight_norm: float = 0.
    early_stopping_metric: str = "eval_weighted_avg_acc"

    mo_weight_type: str = "fixed"
    mo_base_weight: float = 1.0
    mo_cf_weight: float = 0.0
    mo_aug_weight: float = 0.0
    mo_probe_weight: float = 0.0

    cf_train_num_random_bases: int = 50000
    cf_train_num_random_ivn_srcs: int = 20
    cf_eval_num_random_bases: int = 1000
    cf_eval_num_random_ivn_srcs: int = 10
    cf_impactful_ratio: float = 0.5

    probe_max_rank: int = 32
    probe_dropout: float = 0.0001

    eval_only: bool = False

    max_epochs: int = 16
    eval_epochs: int = 1
    patient_epochs: int = 5
    run_steps: int = -1

    interx_after_train: bool = True
    interx_num_cf_training_pairs: int = 0
    interx_num_inputs: int = 1000
    interx_batch_size: int = 128
    interx_save_results: bool = True
    seed: int = 39

    model_save_path: str = ""
    res_save_dir: str = ""
    log_path: str = ""
    mapping: str = ""

    id: int = -1
    db_path: str = ""
