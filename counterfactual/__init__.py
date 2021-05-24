from dataclasses import dataclass

REMAPPING_PATH="data/tokenization/bert-remapping.txt"
VOCAB_PATH = "data/tokenization/bert-vocab.txt"

@dataclass
class CounterfactualTrainingConfig:
    data_path: str = ""
    tokenizer_vocab_path: str = VOCAB_PATH
    output_classes: int = 3
    device: str = "cuda"

    train_batch_size: int = 32
    eval_batch_size: int = 64

    optimizer_type: str = "adamw"
    lr: float = 0.01
    lr_scheduler_type: str = ""
    lr_warmup_subepochs: int = 5 # changed
    weight_norm: float = 0.

    train_multitask_scheduler_type: str = "fixed"
    base_to_cf_ratio: float = 1.0
    num_subepochs_per_epoch: int = 20
    scheduler_warmup_subepochs: int = 8
    scheduler_warmup_step_size: float = 0.1

    cf_type: str = "random_only"
    cf_train_num_random_bases: int = 50000
    cf_train_num_random_ivn_srcs: int = 20
    cf_eval_num_random_bases: int = 1000
    cf_eval_num_random_ivn_srcs: int = 10

    eval_only: bool = False

    max_subepochs: int = 200 # changed
    run_steps: int = -1
    eval_subepochs: int = 5 # changed
    patient_subepochs: int = 20 # changed

    model_save_path: str = "cf_bert"
    res_save_dir: str = ""
    log_path: str = ""
    mapping: str = ""

    id: int = -1
    db_path: str = ""
