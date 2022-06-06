from dataclasses import dataclass

from counterfactual import VOCAB_PATH


@dataclass
class CounterfactualTrainingConfig:
    data_path: str = ""
    tokenizer_vocab_path: str = VOCAB_PATH
    output_classes: int = 3
    device: str = "cuda"

    train_batch_size: int = 32
    eval_batch_size: int = 64

    optimizer_type: str = "adamw"
    lr: float = 5e-5
    lr_scheduler_type: str = "linear"
    lr_warmup_subepochs: int = 10
    weight_norm: float = 0.
    primary_metric: str = "eval_avg_acc"

    train_multitask_scheduler_type: str = "fixed"
    base_to_cf_ratio: float = 1.0
    num_subepochs_per_epoch: int = 20
    scheduler_warmup_subepochs: int = 10
    scheduler_warmup_step_size: float = 0.1

    cf_type: str = "random_only"
    cf_train_num_random_bases: int = 50000
    cf_train_num_random_ivn_srcs: int = 20
    cf_eval_num_random_bases: int = 1000
    cf_eval_num_random_ivn_srcs: int = 10
    cf_impactful_ratio: float = 0.5

    eval_only: bool = False

    max_subepochs: int = 80 # changed
    run_steps: int = -1
    eval_subepochs: int = 5 # changed
    patient_subepochs: int = 20 # changed

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