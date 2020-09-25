# train.py

import time

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import datasets


class Trainer:
    def __init__(self, data, model, batch_size=64, lr=0.01, weight_norm=0,
                 max_epochs=100, run_steps=-1, evals_per_epoch=5,
                 patient_epochs=20, batch_first=True, model_save_path=None,
                 verbose=True):
        self.data = data
        self.model = model

        self.device = model.device
        self.batch_size = batch_size
        self.lr = lr
        self.weight_norm = weight_norm
        self.max_epochs = max_epochs
        self.run_steps = run_steps
        self.evals_per_epoch = evals_per_epoch
        self.patient_epochs = patient_epochs
        self.model_save_path = model_save_path
        self.batch_first = batch_first
        self.verbose = verbose

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          weight_decay=weight_norm)
        self.collate_fn = lambda batch: datasets.my_collate(batch, batch_first=batch_first)
        self.dataloader = DataLoader(data.train, batch_size=batch_size,
                                     shuffle=True, collate_fn=self.collate_fn)
        self.loss_fxn = nn.CrossEntropyLoss(reduction='mean')

        # setup for early stopping
        self.eval_steps = (data.train.num_examples // batch_size) // evals_per_epoch
        self.patient_threshold = patient_epochs * evals_per_epoch * self.eval_steps

    def config(self):
        return {
            "batch_size": self.batch_size,
            "lr": self.lr,
            "weight_norm": self.weight_norm,
            "batch_first": self.batch_first,
            "max_epochs": self.max_epochs,
            "run_steps": self.run_steps,
            "evals_per_epoch": self.evals_per_epoch,
            "patient_epochs": self.patient_epochs,
            "model_save_path": self.model_save_path,
            "verbose": self.verbose
        }

    def train(self):
        # initialize variables for early stopping
        early_stopping = False
        best_dev_acc = 0.
        steps_without_increase = 0
        best_model_checkpoint = {}

        train_start_time = time.time()
        if self.verbose:
            print("========= Start training =========")

        for epoch in range(self.max_epochs):
            if self.verbose:
                print("------ Beginning epoch {} ------".format(epoch))
            epoch_start_time = time.time()
            for step, input_tuple in enumerate(self.dataloader):
                self.model.train()
                self.model.zero_grad()
                input_tuple = [x.to(self.device) for x in input_tuple]
                labels = input_tuple[1]

                logits = self.model(input_tuple)

                loss = self.loss_fxn(logits, labels)
                loss.backward()

                self.optimizer.step()
                if self.run_steps > 0 and step == self.run_steps - 1:
                    early_stopping = True
                    break

                if step % self.eval_steps == 0:
                    corr, total = evaluate_and_predict(self.data.dev, self.model,
                                                       batch_first=self.batch_first)
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
                        if self.model_save_path:
                            torch.save(best_model_checkpoint,
                                       self.model_save_path)
                        if self.verbose:
                            print("Now at epoch %d, step %d, loss %.4f, "
                                  "got accuracy of %.4f, BETTER"
                                  % (epoch, step, float(loss), acc))
                    else:
                        if self.verbose:
                            print("Now at epoch %d, step %d, loss %.4f, "
                                  "got accuracy of %.4f"
                                  % (epoch, step, float(loss), acc))
                        steps_without_increase += self.eval_steps
                        if steps_without_increase >= self.patient_threshold:
                            print('Ran', steps_without_increase,
                                  'steps without an increase in accuracy,',
                                  'trigger early stopping.')
                            early_stopping = True
                            break
            if self.verbose:
                duration = time.time() - epoch_start_time
                print("---- Finished epoch %d in %.2f seconds, "
                      "best accuracy: %.4f ----"
                      % (epoch, duration, best_dev_acc))
            else:
                print("---- Finished epoch {}, best accuracy: {:.4f}"
                      .format(epoch, best_dev_acc))
            if early_stopping:
                break

        if self.run_steps <= 0:
            train_duration = time.time() - train_start_time
            # checkpoint = torch.load(save_path)
            # best_model.load_state_dict(checkpoint['model_state_dict'])
            # corr, total, _ = evaluate_and_predict(mydataset.dev, best_model)
            print("Finished training. Total time is {}. Got final acc of {}"
                  .format(train_duration, best_dev_acc))
            print("Best model was obtained after {} epochs, in {} seconds"
                  .format(epoch + 1, best_model_duration))
            if self.model_save_path:
                print("Best model saved in {}".format(self.model_save_path))

        return best_model_checkpoint


def evaluate_and_predict(dataset, model, batch_first=False, get_pred=False):
    model.eval()
    with torch.no_grad():
        total_preds = 0
        correct_preds = 0
        # preds_and_labels = []
        batch_size = 100

        collate_fn = lambda batch: datasets.my_collate(batch, batch_first=batch_first)
        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=False, collate_fn=collate_fn)
        # bad_sentences = []

        for tuple in dataloader:
            tuple = [x.to(model.device) for x in tuple]
            y_batch = tuple[1]

            logits = model(tuple)
            pred = torch.argmax(logits, dim=1)

            pred = pred.to(model.device)

            """
            if get_summary:
                y = y_batch.to(model.device).type(torch.float)
                res = res.to(model.device)

            if get_pred or get_summary:
                pred_list = pred.tolist()
                if get_summary:
                    x_list = []
                    for sentence in tuple[0].tolist():
                        x_list.append([i for i in sentence if i != 0])
                    err_list = torch.abs(res - y).tolist()
                y_list = y_batch.tolist()
                if get_summary:
                    ids_and_errors = [pair for pair in \
                                      zip(range(len(y_list)), err_list,
                                          y_list)]
                    bad_sentences.extend(
                        [(x_list[pair[0]], pair[1], pair[2]) \
                         for pair in ids_and_errors if pair[1] > 0.5])
                preds_and_labels.append(list(zip(pred_list, y_list)))
            """

            correct_in_batch = torch.sum(torch.eq(pred, y_batch)).item()
            total_preds += y_batch.shape[0]
            correct_preds += correct_in_batch


        # bad_sentences = sorted(bad_sentences, key=lambda p: p[1], reverse=True)
        return correct_preds, total_preds


def load_model(model_class, save_path, device=None):
    checkpoint = torch.load(save_path)
    assert 'model_config' in checkpoint
    model_config = checkpoint['model_config']
    model = model_class(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    if device is None:
        model = model.to(model.device)
    else:
        model = model.to(device)
    return model, checkpoint
