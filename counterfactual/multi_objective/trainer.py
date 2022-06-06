# trainer.py
import math
from typing import *
from collections import Counter
import shutil
from pprint import pprint
from tqdm import tqdm
import os
import time
from datetime import datetime
from dataclasses import asdict

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

import antra

from counterfactual.multi_objective.dataloader import MultiObjectiveDataLoader
from .typings import MultiObjectiveTrainingConfig

from transformers import get_linear_schedule_with_warmup, \
    get_constant_schedule_with_warmup


def log_stats(epoch, loss, primary_metric, metric_name, better=True, lr=None):
    if lr:
        print(f"Epoch {epoch}, lr={lr:.4}, loss={loss:.4}, "
              f"{metric_name}={primary_metric:.2%}{', BETTER' if better else ''}")
    else:
        print(f"Epoch {epoch}, loss={loss:.4}, "
              f"{metric_name}={primary_metric:.2%}{', BETTER' if better else ''}")

class MultiObjectiveTrainer:
    def __init__(
            self,
            train_dataloader: MultiObjectiveDataLoader,
            eval_dataloader: MultiObjectiveDataLoader,
            low_model: antra.ComputationGraph,
            high_model: antra.ComputationGraph,
            probe_model: Optional[torch.nn.Module],
            configs: MultiObjectiveTrainingConfig,
            low_base_model=None,
            store_cf_pairs: int = 0
    ):
        """ Counterfactual training.

        :param train_dataloader:
        :param eval_dataloader:
        :param low_model:
        :param high_model:
        :param configs: Contains all the training hyperparameters
        :param low_base_model: A torch.nn.Module that is to be trained.
            If not given, will use `low_model.model` as base model by default.
        :param store_cf_pairs: If given a positive value, the trainer will return
            a list of index pairs that denote the indices of counterfactual
            examples encountered during training.
        """
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.high_model = high_model
        self.low_model = low_model
        self.low_base_model = getattr(low_model, "model", low_base_model)
        self.probe_model = probe_model
        if self.low_base_model is None:
            raise ValueError("Must provide reference to low level nn module: "
                             "specify `low_base_model` argument or set `<low_model>.model`")

        self.configs: MultiObjectiveTrainingConfig = configs
        self.device = torch.device(configs.device)

        self.store_cf_pairs = store_cf_pairs
        self.cf_training_pairs = []

        self.model_save_path = None
        if configs.model_save_path:
            model_save_path = configs.model_save_path
            if configs.res_save_dir:
                if not os.path.exists(configs.res_save_dir):
                    os.mkdir(configs.res_save_dir)

                if model_save_path.endswith(".pt"):
                    model_save_path = model_save_path[:-len(".pt")]
                self.model_save_path = os.path.join(
                    configs.res_save_dir, f"{model_save_path}.pt")

        self.eval_only = configs.eval_only
        if self.eval_only:
            assert self.model_save_path
            # load base model
            checkpoint = torch.load(self.model_save_path)
            self.low_base_model.load_state_dict(checkpoint["model_state_dict"])

        # setup for early stopping
        # steps_per_epoch = math.ceil(data.train.num_examples / self.batch_size)
        # self.eval_steps = steps_per_epoch // self.evals_per_epoch
        # self.patient_threshold = self.patient_epochs * self.evals_per_epoch \
        #                          * self.eval_steps

        if configs.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.low_base_model.parameters(), lr=configs.lr,
                weight_decay=configs.weight_norm)
        elif configs.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.low_base_model.parameters(), lr=configs.lr,
                weight_decay=configs.weight_norm)
        else:
            raise ValueError(f"Unsupported optimizer type: {configs.optimizer_type}")

        self.lr_scheduler = None
        self.train_steps_per_epoch = math.ceil(train_dataloader.num_examples / configs.train_batch_size)
        self.eval_steps = math.ceil(eval_dataloader.num_examples / configs.eval_batch_size)
        if configs.lr_scheduler_type:
            num_warmup_steps = self.train_steps_per_epoch * configs.lr_warmup_epoch_ratio
            total_steps = self.train_steps_per_epoch * configs.max_epochs
            if configs.lr_scheduler_type == "linear":
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps, total_steps)
            elif configs.lr_scheduler_type == "constant":
                self.lr_scheduler = get_constant_schedule_with_warmup(
                    self.optimizer, num_warmup_steps, total_steps)
            # elif configs.lr_scheduler_type == "step":
            #     self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            #         self.optimizer, configs.lr_step_epochs * steps_per_epoch,
            #         gamma=configs.lr_step_decay_rate)
            else:
                raise ValueError(f"Unsupported lr scheduler type: "
                                 f"{configs.lr_scheduler_type}")

        # if "subphrase" in getattr(self.base_data.train, "variant", ""):
        #     print("Using weighted loss function")
        #     self.loss_fxn = nn.CrossEntropyLoss(reduction='none')
        # else:
        self.loss_fxn = nn.CrossEntropyLoss(reduction='mean')



    def inference_step(self, batch, is_train=False) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], int]:
        """ Do basic forward pass or intervention to get logits and labels.

        This will be called in both evaluation and training, so any setup such
        as model.zero_grad(), model.train() or model.eval(), and loss computation
        will be done outside this function

        :param batch: batch of inputs directly yielded from the dataloader
        :return: (logits, labels, task_name)
        """
        inputs, weights = batch
        high_device = torch.device("cpu")
        low_device = self.low_base_model.device

        high_ivn = inputs["high_intervention"].to(high_device)
        low_base_input = inputs["low_base_input"].to(low_device)
        low_aug_input = inputs["low_aug_input"].to(low_device)
        aug_labels = inputs["aug_labels"].to(low_device)
        base_labels = inputs["base_labels"].to(low_device)
        low_ivn_src = inputs["low_intervention_source"].to(low_device)
        mapping = inputs["mapping"]

        # record indices of counterfactual examples
        if is_train and len(self.cf_training_pairs) < self.store_cf_pairs:
            self.cf_training_pairs.extend(
                zip(inputs["base_idxs"], inputs["ivn_src_idxs"]))

        high_ivn_nodes = list(high_ivn.intervention.values.keys())
        assert len(high_ivn_nodes) == 1 # assume intervene at only one high variable for now
        high_ivn_node = high_ivn_nodes[0]

        low_node_to_loc = mapping[high_ivn_node]

        loss_dict = {}
        correct_count_dict = {}
        num_exs = len(inputs["base_labels"])

        def record_results(objective: str, logits: torch.Tensor, labels: torch.Tensor):
            loss_dict[objective] = self.loss_fxn(logits, labels)
            pred = torch.argmax(logits, dim=1)
            correct_count_dict[objective] = torch.sum(torch.eq(pred, labels)).item()

        if weights['cf'] > 0 or weights['probe'] > 0 or not is_train:
            low_node_to_val = {}
            for low_node, low_loc in low_node_to_loc.items():
                low_val = self.low_model.compute_node(low_node, low_ivn_src)
                low_node_to_val[low_node] = low_val[low_loc]

            low_ivn = antra.Intervention.batched(
                low_base_input, low_node_to_val, low_node_to_loc,
                batch_dim=0, cache_results=False
            )

            if weights['cf'] > 0 or not is_train:
                _, cf_logits = self.low_model.intervene(low_ivn)
                _, cf_labels = self.high_model.intervene(high_ivn)
                cf_labels = cf_labels.to(low_device)
                record_results('cf', cf_logits, cf_labels)

            if weights['probe'] > 0 or not is_train:
                if len(low_node_to_val) > 1 or len(low_node_to_loc) > 1:
                    raise NotImplementedError('Does not support multiple low node locations for probe loss')
                low_node = list(low_node_to_val.keys())[0]
                probe_inputs = low_node_to_val[low_node]
                probe_logits = self.probe_model(probe_inputs)
                probe_labels = high_ivn.intervention[high_ivn_node].to(low_device)
                record_results('probe', probe_logits, probe_labels)

        if weights['base'] > 0 or not is_train:
            base_logits = self.low_model.compute(low_base_input)
            record_results('base', base_logits, base_labels)

        if weights['aug'] > 0 or not is_train:
            aug_logits = self.low_model.compute(low_aug_input)
            record_results('aug', aug_logits, aug_labels)

        total_loss = torch.tensor(0., device=low_device)
        for k in loss_dict:
            total_loss += weights[k] * loss_dict[k]

        loss_dict['total'] = total_loss
        return loss_dict, correct_count_dict, num_exs


    def train(self):
        """ Train the model """
        early_stopping = False
        best_dev_acc = 0.
        best_primary_metric = 0.
        epochs_without_increase = 0
        best_model_checkpoint = {}
        conf = self.configs
        print("using configs---")
        pprint(asdict(conf))

        # setup summary writer
        writer_dir = os.path.join(conf.res_save_dir, "tensorboard")
        if os.path.exists(writer_dir):
            shutil.rmtree(writer_dir)

        writer = SummaryWriter(log_dir=writer_dir)


        print("========= Start training =========")
        total_step = 0
        for epoch in range(conf.max_epochs):
            epoch_start_time = time.time()

            epoch_total_loss = 0.
            epoch_step = 0

            for step, batch in enumerate(tqdm(self.train_dataloader, total=self.train_steps_per_epoch, desc=f"Epoch {epoch}")):
                self.low_base_model.train()
                self.low_base_model.zero_grad()

                loss_dict, _, _= self.inference_step(batch, is_train=True)
                loss_dict['total'].backward()

                epoch_total_loss += loss_dict['total'].item()

                for k in loss_dict:
                    writer.add_scalar(f"Train/{k}_loss", loss_dict[k].item(), total_step)

                self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # run only a few steps for testing
                epoch_step += 1
                total_step += 1
                if conf.run_steps > 0 and step == conf.run_steps - 1:
                    early_stopping = True
                    break

            epoch_avg_loss = epoch_total_loss / epoch_step

            if (epoch + 1) % conf.eval_epochs == 0:
                eval_res_dict = self.evaluate_and_predict()

                for metric, value in eval_res_dict.items():
                    writer.add_scalar(f"Eval/{metric}", value, epoch)

                primary_metric = eval_res_dict[conf.early_stopping_metric]
                if primary_metric > best_primary_metric:
                    best_primary_metric = primary_metric
                    epochs_without_increase = 0

                    best_model_checkpoint = {
                        'model_name': self.low_base_model.__class__,
                        'epoch': epoch,
                        'model_state_dict': self.low_base_model.state_dict(),
                        'train_config': self.configs,
                        'model_config': self.low_base_model.config(),
                        'metrics': eval_res_dict,
                        'primary_metric': conf.early_stopping_metric
                    }
                    if self.model_save_path:
                        torch.save(best_model_checkpoint, self.model_save_path)

                    log_stats(epoch, epoch_avg_loss, primary_metric, conf.early_stopping_metric,
                              lr=self.lr_scheduler.get_lr()[0] if self.lr_scheduler else None)

                    if primary_metric >= 0.99:
                        print('Performance is already good enough, (early stopping metric > 99%), trigger early stopping')
                        break

                else:
                    log_stats(epoch, epoch_avg_loss, primary_metric, conf.early_stopping_metric, better=False,
                              lr=self.lr_scheduler.get_lr()[0] if self.lr_scheduler else None)

                    epochs_without_increase += conf.eval_epochs
                    if epochs_without_increase >= conf.patient_epochs:
                        print('Ran', epochs_without_increase,
                              'steps without an increase in accuracy,',
                              'trigger early stopping.')
                        break

            duration = time.time() - epoch_start_time
            # print(f"---- Finished epoch {subepoch} in {duration:.2f} seconds, "
            #       f"best accuracy: {best_dev_acc:.2%} ----")
            if early_stopping:
                break

        if conf.run_steps <= 0:
            # checkpoint = torch.load(save_path)
            # best_model.load_state_dict(checkpoint['model_state_dict'])
            # corr, total, _ = evaluate_and_predict(mydataset.dev, best_model)
            print(f"Finished training. Final {conf.early_stopping_metric}={best_model_checkpoint['metrics'][conf.early_stopping_metric]:.2%}")
            print(f"Best model was obtained after {epoch + 1} epochs.")
            if self.model_save_path:
                print(f"Best model checkpoint saved in {self.model_save_path}")

        writer.close()
        return best_model_checkpoint, self.model_save_path

    def evaluate_and_predict(self):
        """ Evaluate the current low model """
        self.low_base_model.eval()
        correct_counter = Counter()
        total_counter = Counter()
        weighted_counter = Counter()

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluate", total=self.eval_steps):
                sum_weights = sum(batch[1].values())
                weights = {objective: weight / sum_weights for objective, weight in batch[1].items()}
                loss_dict, correct_count_dict, num_exs = self.inference_step(batch, is_train=False)

                for objective in correct_count_dict.keys():
                    correct_counter[objective] += correct_count_dict[objective]
                    if weights[objective] > 0:
                        weighted_counter[objective] += correct_count_dict[objective] * weights[objective]
                    total_counter[objective] += num_exs

        res_dict = {}
        weighted_res_dict = {}
        for objective in correct_counter.keys():
            res_dict[f"eval_{objective}_acc"] = correct_counter[objective] / total_counter[objective]
            weighted_res_dict[f'eval_{objective}_acc'] = weighted_counter[objective] / total_counter[objective]

        res_dict["eval_avg_acc"] = sum(res_dict.values()) / len(res_dict)
        res_dict["eval_weighted_avg_acc"] = sum(weighted_res_dict.values())

        return res_dict


