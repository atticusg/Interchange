# trainer.py
from typing import *
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

from counterfactual.multidataloader import MultiTaskDataLoader
from counterfactual import CounterfactualTrainingConfig

from transformers import get_linear_schedule_with_warmup, \
    get_constant_schedule_with_warmup


def log_stats(epoch, loss, acc, better=True, lr=None):
    if lr:
        print(f"Now at subepoch {epoch}, lr {lr:.4}, loss {loss:.4}, "
              f"got accuracy of {acc:.2%}{', BETTER' if better else ''}")
    else:
        print(f"Now at subepoch {epoch}, loss {loss:.4}, "
              f"got accuracy of {acc:.2%}{', BETTER' if better else ''}")

class CounterfactualTrainer:
    def __init__(
            self,
            train_dataloader: MultiTaskDataLoader,
            eval_dataloader: MultiTaskDataLoader,
            low_model: antra.ComputationGraph,
            high_model: antra.ComputationGraph,
            configs: CounterfactualTrainingConfig,
            low_base_model=None,
    ):
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.high_model: antra.ComputationGraph = high_model
        self.low_model: antra.ComputationGraph = low_model
        self.low_base_model = getattr(low_model, "model", low_base_model)
        if self.low_base_model is None:
            raise ValueError("Must provide reference to low level nn module: "
                             "specify `low_base_model` argument or set `<low_model>.model`")

        self.configs: CounterfactualTrainingConfig = configs
        self.device = torch.device(configs.device)

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
        if configs.lr_scheduler_type:
            tr_schedule = train_dataloader.schedule_fn
            num_warmup_steps = sum(sum(tr_schedule(i)) for i in range(configs.lr_warmup_subepochs))
            total_steps = sum(sum(tr_schedule(i)) for i in range(configs.max_subepochs))
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



    def inference_step(self, batch) -> Tuple:
        """ Do basic forward pass or intervention to get logits and labels.

        This will be called in both evaluation and training, so any setup such
        as model.zero_grad(), model.train() or model.eval(), and loss computation
        will be done outside this function

        :param batch: batch of inputs directly yielded from the dataloader
        :return:
        """
        inputs, task_name = batch
        high_device = torch.device("cpu")
        low_device = self.low_base_model.device

        if task_name.startswith("cf"):
            high_ivn = inputs["high_intervention"].to(high_device)
            low_base_input = inputs["low_base_input"].to(low_device)
            low_ivn_src = inputs["low_intervention_source"].to(low_device)
            mapping = inputs["mapping"]

            high_ivn_nodes = list(high_ivn.intervention.values.keys())
            assert len(high_ivn_nodes) == 1 # assume intervene at only one high variable for now
            high_ivn_node = high_ivn_nodes[0]

            low_node_to_loc = mapping[high_ivn_node]

            # TODO: modify compgraph interface to use results dict
            low_node_to_val = {}
            for low_node, low_loc in low_node_to_loc.items():
                low_val = self.low_model.compute_node(low_node, low_base_input)
                low_node_to_val[low_node] = low_val[low_loc]

            low_ivn = antra.Intervention.batched(
                low_ivn_src, low_node_to_val, low_node_to_loc,
                batch_dim=0, cache_results=False
            )

            _, logits = self.low_model.intervene(low_ivn)
            _, labels = self.high_model.intervene(high_ivn)
            labels = labels.to(low_device)
        else:
            model_inputs = inputs["inputs"].to(low_device)
            labels = inputs["labels"].to(low_device)
            logits = self.low_model.compute(model_inputs)

        return logits, labels, task_name


    def train(self):
        """ Train the model """
        early_stopping = False
        best_dev_acc = 0.
        best_dev_total_acc = 0.
        epochs_without_increase = 0
        best_model_checkpoint = {}
        model_save_path = None
        conf = self.configs
        print("using configs---")
        pprint(asdict(conf))

        if conf.model_save_path or conf.res_save_dir:
            train_start_time_str = datetime.now().strftime("%m%d_%H%M%S")
            model_save_path = conf.model_save_path
            if conf.res_save_dir:
                if not os.path.exists(conf.res_save_dir):
                    os.mkdir(conf.res_save_dir)

                if model_save_path.endswith(".pt"):
                    model_save_path = model_save_path[:-len(".pt")]
                model_save_path = os.path.join(conf.res_save_dir,
                                         f"{model_save_path}_{train_start_time_str}.pt")

        # setup summary writer
        writer_dir = os.path.join(conf.res_save_dir, "tensorboard")
        if os.path.exists(writer_dir):
            shutil.rmtree(writer_dir)

        writer = SummaryWriter(log_dir=writer_dir)

        train_start_time = time.time()

        print("========= Start training =========")
        total_step = 0
        for subepoch in range(conf.max_subepochs):
            epoch_start_time = time.time()

            total_loss = 0.
            subepoch_step = 0

            curr_schedule = self.train_dataloader.schedule_fn(subepoch)
            schedule_str = ",".join(f"{task}={n}" for task, n in zip(self.train_dataloader.task_names, curr_schedule))
            print(f"Train epoch {subepoch} schedule {schedule_str}")
            curr_epoch_len = sum(curr_schedule)
            for step, batch in enumerate(tqdm(self.train_dataloader, total=curr_epoch_len, desc=f"Epoch {subepoch}")):
                self.low_base_model.train()
                self.low_base_model.zero_grad()

                logits, labels, task_name = self.inference_step(batch)

                loss = self.loss_fxn(logits, labels)
                loss.backward()
                total_loss += loss.item()

                writer.add_scalar("Train/loss", loss.item(), total_step)
                writer.add_scalar(f"Train/{task_name}/loss", loss.item(), total_step)

                self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # run only a few steps for testing
                subepoch_step += 1
                total_step += 1
                if conf.run_steps > 0 and step == conf.run_steps - 1:
                    early_stopping = True
                    break

            avg_loss = total_loss / subepoch_step
            writer.add_scalar("Train/avg_loss", avg_loss, subepoch)

            if (subepoch + 1) % conf.eval_subepochs == 0:
                eval_res_dict = self.evaluate_and_predict()

                total_acc = eval_res_dict["total_correct_cnt"] / eval_res_dict["total_cnt"]
                base_acc = eval_res_dict["base_correct_cnt"] / eval_res_dict["base_cnt"]
                writer.add_scalar("Eval/total_acc", total_acc, subepoch)
                writer.add_scalar("Eval/base_acc", base_acc, subepoch)

                best_dev_total_acc = max(best_dev_total_acc, total_acc)
                if base_acc > best_dev_acc:
                    best_dev_acc = base_acc
                    epochs_without_increase = 0
                    best_model_duration = time.time() - train_start_time

                    best_model_checkpoint = {
                        'model_name': self.low_base_model.__class__,
                        'epoch': subepoch,
                        'duration': best_model_duration,
                        'model_state_dict': self.low_base_model.state_dict(),
                        'avg_train_loss': avg_loss,
                        'best_dev_acc': best_dev_acc,
                        'dev_total_acc': total_acc,
                        'train_config': self.configs,
                        'model_config': self.low_base_model.config(),
                    }
                    if model_save_path:
                        torch.save(best_model_checkpoint, model_save_path)

                    log_stats(subepoch, avg_loss, base_acc,
                              lr=self.lr_scheduler.get_lr()[0] if self.lr_scheduler else None)

                else:
                    log_stats(subepoch, avg_loss, base_acc, better=False,
                              lr=self.lr_scheduler.get_lr()[0] if self.lr_scheduler else None)

                    epochs_without_increase += conf.eval_subepochs
                    if epochs_without_increase >= conf.patient_subepochs:
                        print('Ran', epochs_without_increase,
                              'steps without an increase in accuracy,',
                              'trigger early stopping.')
                        early_stopping = True
                        break

            duration = time.time() - epoch_start_time
            # print(f"---- Finished epoch {subepoch} in {duration:.2f} seconds, "
            #       f"best accuracy: {best_dev_acc:.2%} ----")
            if early_stopping:
                break

        if conf.run_steps <= 0:
            train_duration = time.time() - train_start_time
            # checkpoint = torch.load(save_path)
            # best_model.load_state_dict(checkpoint['model_state_dict'])
            # corr, total, _ = evaluate_and_predict(mydataset.dev, best_model)
            print(f"Finished training. Total time is {train_duration:.1}s. Got final acc of {best_dev_acc:.2%}")
            print(f"Best model was obtained after {subepoch + 1} epochs, in {best_model_duration:.1} seconds")
            if conf.model_save_path:
                print(f"Best model saved in {model_save_path}")

        best_model_checkpoint["best_dev_total_acc"] = best_dev_total_acc
        writer.close()
        return best_model_checkpoint, model_save_path

    def evaluate_and_predict(self):
        """ Evaluate the current low model """
        self.low_base_model.eval()
        with torch.no_grad():
            total_preds = 0
            correct_preds = 0

            total_base_preds = 0
            correct_base_preds = 0

            curr_schedule = self.eval_dataloader.schedule_fn(0)
            schedule_str = ",".join(f"{task}={n}" for task, n in zip(self.eval_dataloader.task_names, curr_schedule))
            print(f"Evaluation schedule {schedule_str}")
            curr_epoch_len = sum(curr_schedule)
            for batch in tqdm(self.eval_dataloader, desc="Evaluate", total=curr_epoch_len):
                logits, labels, task_name = self.inference_step(batch)
                pred = torch.argmax(logits, dim=1)
                correct_in_batch = torch.sum(torch.eq(pred, labels)).item()

                if task_name == "base":
                    total_base_preds += labels.shape[0]
                    correct_base_preds += correct_in_batch

                total_preds += labels.shape[0]
                correct_preds += correct_in_batch

            # bad_sentences = sorted(bad_sentences, key=lambda p: p[1], reverse=True)
            return {
                "total_correct_cnt": correct_preds,
                "total_cnt": total_preds,
                "base_correct_cnt": correct_base_preds,
                "base_cnt": total_base_preds
            }


