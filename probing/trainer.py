import os
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

def print_stats(epoch, loss, acc, better=True, lr=None):
    print(f"    Epoch {epoch}, lr {lr:.4}, loss {loss:.4}, "
          f"got accuracy of {acc:.2%}{', BETTER' if better else ''}")


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
                 probe_save_dir: str="",
                 device: str="cuda",
                 **kwargs):
        self.device = torch.device(device)
        self.data = data
        self.probe = probe.to(self.device)

        self.probe_train_batch_size = probe_train_batch_size
        self.probe_train_eval_batch_size = probe_train_eval_batch_size
        self.probe_train_weight_norm = probe_train_weight_norm
        self.probe_train_max_epochs = probe_train_max_epochs
        self.probe_train_lr_patience_epochs = probe_train_lr_patience_epochs
        self.probe_train_lr_anneal_factor = probe_train_lr_anneal_factor
        self.probe_save_dir = probe_save_dir

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

        self.dataloader = DataLoader(data.train, batch_size=probe_train_batch_size,
                                     shuffle=True)

    def train(self):
        early_stopping = False
        best_dev_acc = 0.
        best_model_checkpoint = {}
        model_save_path = os.path.join(self.probe_save_dir,
            f"{self.probe.name}-{datetime.now().strftime('%m%d_%H%M%S')}")

        print(f"======= Start training probe {self.probe.name} ========")

        for epoch in range(self.probe_train_max_epochs):
            for step, batch in enumerate(self.dataloader):
                self.probe.train()
                self.probe.zero_grad()
                batch = [x.to(self.device) for x in batch]
                inputs, labels = batch

                logits = self.probe(inputs)
                loss = self.loss_fxn(logits, labels)
                loss.backward()
                self.optimizer.step()

            acc = self.eval()

            if acc > best_dev_acc:
                best_dev_acc = acc
                best_model_checkpoint = {
                    'model_name': self.probe.name,
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': self.probe.state_dict(),
                    'loss': loss.item(),
                    'best_dev_acc': best_dev_acc,
                    'train_config': self.config(),
                    'model_config': self.probe.config(),
                }
                if model_save_path:
                    torch.save(best_model_checkpoint, model_save_path)

                print_stats(epoch, loss.item(), acc, better=True,
                            lr=self.scheduler.get_lr()[0] if self.scheduler else None)


    def eval(self):
        return 0
