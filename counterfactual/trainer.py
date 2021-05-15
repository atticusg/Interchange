# trainer.py
from dataclasses import dataclass, asdict
import math
import os
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


import antra

from counterfactual.multidataloader import MultiTaskDataLoader

from transformers import get_linear_schedule_with_warmup, \
    get_constant_schedule_with_warmup


def log_stats(epoch, loss, acc, better=True, lr=None):
    if lr:
        print(f"Now at subepoch {epoch}, lr {lr:.4}, loss {loss:.4}, "
              f"got accuracy of {acc:.2%}{', BETTER' if better else ''}")
    else:
        print(f"Now at subepoch {epoch}, loss {loss:.4}, "
              f"got accuracy of {acc:.2%}{', BETTER' if better else ''}")


@dataclass
class CounterfactualTrainerConfig:
    optimizer_type="adam"
    weight_norm=0.
    lr=0.01
    lr_scheduler_type=""
    lr_warmup_subepochs=1 # changed
    max_subepochs=100 # changed
    run_steps=-1
    eval_subepochs=5 # changed
    patient_subepochs=20 # changed
    device="cuda"
    model_save_path=None
    res_save_dir=None


class CounterfactualTrainer:
    def __init__(
            self,
            train_dataloader: MultiTaskDataLoader,
            eval_dataloader: MultiTaskDataLoader,
            low_model,
            high_model,
            configs: CounterfactualTrainerConfig,
            low_base_model=None,
            **kwargs
    ):
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.high_model = high_model
        self.low_model = low_model
        self.model = getattr(low_model, "model", low_base_model)
        if self.model is None:
            raise ValueError("Must provide reference to low level nn module: "
                             "specify `low_base_model` argument or set `<low_model>.model`")

        self.configs: CounterfactualTrainerConfig = configs
        self.device = torch.device(configs.device)

        # setup for early stopping
        # steps_per_epoch = math.ceil(data.train.num_examples / self.batch_size)
        # self.eval_steps = steps_per_epoch // self.evals_per_epoch
        # self.patient_threshold = self.patient_epochs * self.evals_per_epoch \
        #                          * self.eval_steps

        if configs.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=configs.lr,
                weight_decay=configs.weight_norm)
        elif configs.optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=configs.lr,
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


    def train_step(self, step, batch):
        self.model.train()
        self.model.zero_grad()
        inputs = [x.to(self.device) for x in batch]
        labels = inputs[-1]

        logits = self.model(inputs)
        loss = self.loss_fxn(logits, labels)

        # if "subphrase" in getattr(self.base_data.train, "variant", ""):
        #     loss = torch.mean(loss * input_tuple[-3])

        return loss

    def train(self):
        # initialize variables for early stopping
        early_stopping = False
        best_dev_acc = 0.
        steps_without_increase = 0
        best_model_checkpoint = {}
        model_save_path = None
        conf = self.configs
        print("using configs---")
        print(conf)

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


        train_start_time = time.time()

        print("========= Start training =========")

        for subepoch in range(conf.max_subepochs):
            print("------ Beginning epoch {} ------".format(subepoch))
            epoch_start_time = time.time()

            total_loss = 0.
            subepoch_steps = 0
            for step, batch in enumerate(self.train_dataloader):
                loss = self.train_step(step, batch)
                loss.backward()
                total_loss += loss.item()

                self.optimizer.step()

                if self.lr_scheduler:
                    self.lr_scheduler.step()

                # run only a few steps for testing
                subepoch_steps += 1
                if conf.run_steps > 0 and step == conf.run_steps - 1:
                    early_stopping = True
                    break

            if (subepoch + 1) % conf.eval_subepochs == 0:
                # todo: edit this
                corr, total = evaluate_and_predict(
                    self.base_data.dev, self.model,
                    eval_batch_size=self.eval_batch_size,
                    batch_first=self.batch_first,
                    device=self.device
                )

                acc = corr / total
                avg_loss = total_loss / subepoch_steps
                if acc > best_dev_acc:
                    best_dev_acc = acc
                    steps_without_increase = 0
                    best_model_duration = time.time() - train_start_time

                    best_model_checkpoint = {
                        'model_name': self.model.__class__,
                        'epoch': subepoch,
                        'duration': best_model_duration,
                        'model_state_dict': self.model.state_dict(),
                        'loss': avg_loss,
                        'best_dev_acc': best_dev_acc,
                        'train_config': self.configs,
                        'model_config': self.model.config(),
                    }
                    if model_save_path:
                        torch.save(best_model_checkpoint, model_save_path)

                    log_stats(subepoch, avg_loss, acc,
                              lr=self.lr_scheduler.get_lr()[0] if self.lr_scheduler else None)

                else:
                    log_stats(subepoch, avg_loss, acc, better=False,
                              lr=self.lr_scheduler.get_lr()[0] if self.lr_scheduler else None)

                    steps_without_increase += self.eval_steps
                    if steps_without_increase >= self.patient_threshold:
                        print('Ran', steps_without_increase,
                              'steps without an increase in accuracy,',
                              'trigger early stopping.')
                        early_stopping = True
                        break

            duration = time.time() - epoch_start_time
            print(f"---- Finished epoch {subepoch} in {duration:.2f} seconds, "
                  f"best accuracy: {best_dev_acc:.2%} ----")
            if early_stopping:
                break

        if self.run_steps <= 0:
            train_duration = time.time() - train_start_time
            # checkpoint = torch.load(save_path)
            # best_model.load_state_dict(checkpoint['model_state_dict'])
            # corr, total, _ = evaluate_and_predict(mydataset.dev, best_model)
            print(f"Finished training. Total time is {train_duration:.1}s. Got final acc of {best_dev_acc:.2%}")
            print(f"Best model was obtained after {subepoch + 1} epochs, in {best_model_duration:.1} seconds")
            if self.model_save_path:
                print(f"Best model saved in {model_save_path}")

        return best_model_checkpoint, model_save_path

# TODO: modify this part for counterfactual training
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


