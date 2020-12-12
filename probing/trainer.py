import os
from datetime import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Union

def print_stats(epoch, train_loss, dev_acc, dev_loss, better=True):
    print(f"    Epoch {epoch}, train loss {train_loss:.4}, dev acc {dev_acc:.2%} "
          f"dev loss {dev_loss:.4}{', BETTER' if better else ''}")


class ProbeTrainer:
    """ Train one probe """
    def __init__(self, data, probe,
                 probe_train_batch_size: int=64,
                 probe_train_eval_batch_size: int=256,
                 probe_train_weight_norm: float=0.,
                 probe_train_max_epochs: int=40,
                 probe_train_lr: float=0.001,
                 probe_train_lr_patience_epochs: int=4,
                 probe_train_lr_anneal_factor: float=0.5,
                 res_save_dir: str= "",
                 device: Union[str, torch.device]= "cuda",
                 **kwargs):
        if isinstance(device, str):
            device = torch.device(device)

        self.device = torch.device(device)
        self.data = data
        self.probe = probe.to(self.device)

        self.probe_train_batch_size = probe_train_batch_size
        self.probe_train_eval_batch_size = probe_train_eval_batch_size
        self.probe_train_weight_norm = probe_train_weight_norm
        self.probe_train_max_epochs = probe_train_max_epochs
        self.probe_train_lr_patience_epochs = probe_train_lr_patience_epochs
        self.probe_train_lr_anneal_factor = probe_train_lr_anneal_factor
        self.res_save_dir = res_save_dir

        self.optimizer = optim.Adam(
            probe.parameters(),
            lr=probe_train_lr
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=probe_train_lr_anneal_factor,
            patience=probe_train_lr_patience_epochs
        )
        self.loss_fxn = nn.CrossEntropyLoss()

        collate_fn = data.train.get_collate_fn(probe.high_node, probe.low_node,
                                               probe.low_loc, probe.is_control)
        self.train_dataloader = DataLoader(data.train, batch_size=probe_train_batch_size,
                                           shuffle=True, collate_fn=collate_fn)
        self.dev_dataloader = DataLoader(data.dev, batch_size=probe_train_eval_batch_size,
                                         shuffle=False, collate_fn=collate_fn)

    def train(self):
        # epochs_without_increase = 0
        best_model_checkpoint = {}
        best_dev_loss = float("inf")
        best_dev_acc = 0.

        print(f"=== Start training probe {self.probe.name} ")

        for epoch in range(self.probe_train_max_epochs):
            train_loss = 0.
            num_examples = 0

            for step, batch in enumerate(self.train_dataloader):
                self.probe.train()
                self.probe.zero_grad()
                batch = [x.to(self.device) for x in batch]
                inputs, labels = batch

                logits = self.probe(inputs)
                loss = self.loss_fxn(logits, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                num_examples += labels.shape[0]

            train_loss /= num_examples
            dev_acc, dev_loss = self.eval()
            self.scheduler.step(dev_loss)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_dev_acc = dev_acc
                best_model_checkpoint = {
                    'model_name': self.probe.name,
                    'epoch': epoch,
                    'model_state_dict': self.probe.state_dict(),
                    'loss': train_loss,
                    'dev_loss': best_dev_loss,
                    'dev_acc': dev_acc,
                    'model_config': self.probe.config(),
                }
                print_stats(epoch, train_loss, dev_acc, dev_loss, better=True)
            else:
                print_stats(epoch, train_loss, dev_acc, dev_loss, better=False)
                # epochs_without_increase += 1
                # if epochs_without_increase >= self.probe_train_lr_patience_epochs:
                #     print("    Early stopping")
                #     break

        print(f"==== Finished training probe {self.probe.name}")
        print(f"    best accuracy: {best_dev_acc: .2%}")

        if self.res_save_dir:
            time_str = datetime.now().strftime('%m%d_%H%M%S')
            model_save_path = os.path.join(self.res_save_dir, f"{self.probe.name}-{time_str}.pt")
            best_model_checkpoint["model_save_path"] = model_save_path
            torch.save(best_model_checkpoint, model_save_path)
            print(f"    saved model to: {model_save_path}")

        print(f"===============")
        return best_model_checkpoint

    def eval(self):
        self.probe.eval()
        with torch.no_grad():
            total_preds = 0
            correct_preds = 0
            total_loss = 0.
            for batch in self.dev_dataloader:
                batch = [x.to(self.device) for x in batch]
                inputs, labels = batch
                logits = self.probe(inputs)

                loss = self.loss_fxn(logits, labels)
                total_loss += loss.item() * labels.shape[0]
                pred = torch.argmax(logits, dim=1)
                correct_in_batch = torch.sum(torch.eq(pred, labels)).item()
                total_preds += labels.shape[0]
                correct_preds += correct_in_batch

            avg_loss = total_loss / total_preds

        return correct_preds / total_preds, avg_loss

