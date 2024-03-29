# trainer.py
import math
import os
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import datasets

from transformers import get_linear_schedule_with_warmup, \
    get_constant_schedule_with_warmup


def print_stats(epoch, step, loss, acc, better=True, lr=None):
    if lr:
        print(f"Now at epoch {epoch}, step {step}, lr {lr:.4}, loss {loss:.4}, "
              f"got accuracy of {acc:.2%}{', BETTER' if better else ''}")
    else:
        print(f"Now at epoch {epoch}, step {step}, loss {loss:.4}, "
              f"got accuracy of {acc:.2%}{', BETTER' if better else ''}")



class Trainer:
    def __init__(self, data, model,
                 batch_size=64, eval_batch_size=64, batch_first=True,
                 optimizer_type="adam", weight_norm=0., lr=0.01,
                 lr_scheduler_type="", lr_warmup_ratio=0.05,
                 lr_step_epochs=10, lr_step_decay_rate=0.1,
                 max_epochs=100, run_steps=-1, evals_per_epoch=5, patient_epochs=20,
                 device="cuda",
                 model_save_path=None, res_save_dir=None,
                 **kwargs):
        self.data = data
        self.model = model
        self.device = torch.device(device)

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.batch_first = batch_first

        self.optimizer_type = optimizer_type
        self.weight_norm = weight_norm
        self.lr = lr
        self.lr_scheduler_type = lr_scheduler_type
        self.lr_warmup_ratio = lr_warmup_ratio
        self.lr_step_epochs = lr_step_epochs
        self.lr_step_decay_rate = lr_step_decay_rate

        self.max_epochs = max_epochs
        self.run_steps = run_steps
        self.evals_per_epoch = evals_per_epoch
        self.patient_epochs = patient_epochs

        self.model_save_path = model_save_path
        self.res_save_dir = res_save_dir

        # setup for early stopping
        steps_per_epoch = math.ceil(data.train.num_examples / self.batch_size)
        self.eval_steps = steps_per_epoch // self.evals_per_epoch
        self.patient_threshold = self.patient_epochs * self.evals_per_epoch \
                                 * self.eval_steps

        if self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                              weight_decay=self.weight_norm)
        elif self.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                               weight_decay=self.weight_norm)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

        self.scheduler = None
        if self.lr_scheduler_type:
            num_warmup_steps = math.floor(self.lr_warmup_ratio * steps_per_epoch)
            total_steps = steps_per_epoch * self.max_epochs
            if self.lr_scheduler_type == "linear":
                self.scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps, total_steps)
            elif self.lr_scheduler_type == "constant":
                self.scheduler = get_constant_schedule_with_warmup(
                    self.optimizer, num_warmup_steps, total_steps)
            elif self.lr_scheduler_type == "step":
                self.scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, self.lr_step_epochs * steps_per_epoch,
                    gamma=self.lr_step_decay_rate)
            else:
                raise ValueError(f"Unsupported lr scheduler type: "
                                 f"{self.lr_scheduler_type}")

        self.collate_fn = datasets.mqnli.get_collate_fxn(self.data.train, batch_first=self.batch_first)

        if self.collate_fn is not None:
            self.dataloader = DataLoader(data.train, batch_size=self.batch_size,
                                         shuffle=True, collate_fn=self.collate_fn)
        else:
            self.dataloader = DataLoader(data.train, batch_size=self.batch_size,
                                         shuffle=True)

        if "subphrase" in getattr(self.data.train, "variant", ""):
            print("Using weighted loss function")
            self.loss_fxn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fxn = nn.CrossEntropyLoss(reduction='mean')


    def config(self):
        return {
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "batch_first": self.batch_first,

            "optimizer_type": self.optimizer_type,
            "lr": self.lr,
            "lr_scheduler_type": self.lr_scheduler_type,
            "lr_warmup_ratio": self.lr_warmup_ratio,
            "lr_step_epochs": self.lr_step_epochs,
            "lr_step_decay_rate": self.lr_step_decay_rate,
            "weight_norm": self.weight_norm,

            "max_epochs": self.max_epochs,
            "run_steps": self.run_steps,
            "evals_per_epoch": self.evals_per_epoch,
            "patient_epochs": self.patient_epochs,

            "model_save_path": self.model_save_path,
            "res_save_dir": self.res_save_dir
        }

    def train(self):
        # initialize variables for early stopping
        early_stopping = False
        best_dev_acc = 0.
        steps_without_increase = 0
        best_model_checkpoint = {}
        model_save_path = None

        print("using configs---")
        print(self.config())

        if self.model_save_path or self.res_save_dir:
            train_start_time_str = datetime.now().strftime("%m%d_%H%M%S")
            model_save_path = self.model_save_path
            if self.res_save_dir:
                if not os.path.exists(self.res_save_dir):
                    os.mkdir(self.res_save_dir)

                if model_save_path.endswith(".pt"):
                    model_save_path = model_save_path[:-len(".pt")]
                model_save_path = os.path.join(self.res_save_dir,
                                         f"{model_save_path}_{train_start_time_str}.pt")


        train_start_time = time.time()

        print("========= Start training =========")

        for epoch in range(self.max_epochs):
            print("------ Beginning epoch {} ------".format(epoch))
            epoch_start_time = time.time()
            for step, input_tuple in enumerate(self.dataloader):
                self.model.train()
                self.model.zero_grad()

                input_tuple = [x.to(self.device) for x in input_tuple]
                labels = input_tuple[-1]

                logits = self.model(input_tuple)
                loss = self.loss_fxn(logits, labels)

                if "subphrase" in getattr(self.data.train, "variant", ""):
                    loss = torch.mean(loss * input_tuple[-3])

                loss.backward()

                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                if self.run_steps > 0 and step == self.run_steps - 1:
                    early_stopping = True
                    break

                if step % self.eval_steps == 0:
                    corr, total = evaluate_and_predict(self.data.dev, self.model,
                                                       eval_batch_size=self.eval_batch_size,
                                                       batch_first=self.batch_first,
                                                       device=self.device)
                    acc = corr / total

                    if acc > best_dev_acc:
                        best_dev_acc = acc
                        steps_without_increase = 0
                        best_model_duration = time.time() - train_start_time
                        best_model_checkpoint = {
                            'model_name': self.model.__class__,
                            'epoch': epoch,
                            'step': step,
                            'duration': best_model_duration,
                            'model_state_dict': self.model.state_dict(),
                            'loss': loss.item(),
                            'best_dev_acc': best_dev_acc,
                            'train_config': self.config(),
                            'model_config': self.model.config(),
                        }
                        if model_save_path:
                            torch.save(best_model_checkpoint, model_save_path)

                        print_stats(epoch, step, loss.item(), acc,
                            lr=self.scheduler.get_lr()[0] if self.scheduler else None)

                    else:
                        print_stats(epoch, step, loss.item(), acc, better=False,
                            lr=self.scheduler.get_lr()[0] if self.scheduler else None)

                        steps_without_increase += self.eval_steps
                        if steps_without_increase >= self.patient_threshold:
                            print('Ran', steps_without_increase,
                                  'steps without an increase in accuracy,',
                                  'trigger early stopping.')
                            early_stopping = True
                            break

            duration = time.time() - epoch_start_time
            print(f"---- Finished epoch {epoch} in {duration:.2f} seconds, "
                  f"best accuracy: {best_dev_acc:.2%} ----")
            if early_stopping:
                break

        if self.run_steps <= 0:
            train_duration = time.time() - train_start_time
            # checkpoint = torch.load(save_path)
            # best_model.load_state_dict(checkpoint['model_state_dict'])
            # corr, total, _ = evaluate_and_predict(mydataset.dev, best_model)
            print(f"Finished training. Total time is {train_duration:.1}s. Got final acc of {best_dev_acc:.2%}")
            print(f"Best model was obtained after {epoch + 1} epochs, in {best_model_duration:.1} seconds")
            if self.model_save_path:
                print(f"Best model saved in {model_save_path}")

        return best_model_checkpoint, model_save_path


def evaluate_and_predict(dataset, model, eval_batch_size=64,
                         batch_first=False, device=None):
    device = torch.device("cuda") if not device else device
    model.eval()
    with torch.no_grad():
        total_preds = 0
        correct_preds = 0
        batch_size = eval_batch_size

        collate_fn = datasets.mqnli.get_collate_fxn(dataset, batch_first=batch_first)
        if collate_fn is not None:
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=False, collate_fn=collate_fn)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for input_tuple in dataloader:
            input_tuple = [x.to(device) for x in input_tuple]
            labels = input_tuple[-1]

            logits = model(input_tuple)
            pred = torch.argmax(logits, dim=1)

            pred = pred.to(device)

            correct_in_batch = torch.sum(torch.eq(pred, labels)).item()
            total_preds += labels.shape[0]
            correct_preds += correct_in_batch

        # bad_sentences = sorted(bad_sentences, key=lambda p: p[1], reverse=True)
        return correct_preds, total_preds


