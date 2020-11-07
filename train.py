# train.py
import os
import time
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import datasets


class Trainer:
    def __init__(self, data, model, batch_size=64, lr=0.01, weight_norm=0,
                 max_epochs=100, run_steps=-1, evals_per_epoch=5, eval_batch_size=64,
                 patient_epochs=20, use_collate=True, batch_first=True,
                 model_save_path=None, res_save_dir=None,
                 verbose=True, opts={}):
        self.data = data
        self.model = model
        self.device = model.device

        self.batch_size = opts.get("batch_size", batch_size)
        self.eval_batch_size = opts.get("eval_batch_size", eval_batch_size)
        self.lr = opts.get("lr", lr)
        self.weight_norm = opts.get("weight_norm",weight_norm)
        self.max_epochs = opts.get("max_epochs", max_epochs)
        self.run_steps = opts.get("run_steps", run_steps)
        self.evals_per_epoch = opts.get("evals_per_epoch", evals_per_epoch)
        self.patient_epochs = opts.get("patient_epochs", patient_epochs)
        self.model_save_path = opts.get("model_save_path", model_save_path)
        self.res_save_dir = opts.get("res_save_dir", res_save_dir)
        self.batch_first = opts.get("batch_first", batch_first)
        self.verbose = opts.get("verbose", verbose)
        self.use_collate = opts.get("use_collate", use_collate)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_norm)

        if self.use_collate:
            self.collate_fn = lambda batch: datasets.my_collate(batch, batch_first=self.batch_first)
            self.dataloader = DataLoader(data.train, batch_size=self.batch_size,
                                         shuffle=True, collate_fn=self.collate_fn)
        else:
            self.dataloader = DataLoader(data.train, batch_size=self.batch_size,
                                         shuffle=True)
        self.loss_fxn = nn.CrossEntropyLoss(reduction='mean')

        # setup for early stopping
        self.eval_steps = (data.train.num_examples // self.batch_size) // self.evals_per_epoch
        self.patient_threshold = self.patient_epochs * self.evals_per_epoch * self.eval_steps

    def config(self):
        return {
            "batch_size": self.batch_size,
            "eval_batch_size": self.eval_batch_size,
            "lr": self.lr,
            "weight_norm": self.weight_norm,
            "use_collate": self.use_collate,
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
        model_save_path = None

        print("using configs---")
        print(self.config())

        if self.model_save_path or self.res_save_dir:
            train_start_time_str = datetime.now().strftime("%m%d_%H%M%S")
            model_save_path = self.model_save_path
            if self.res_save_dir:
                if model_save_path.endswith(".pt"):
                    model_save_path = model_save_path[:-len(".pt")]
                model_save_path = os.path.join(self.res_save_dir,
                                         f"{model_save_path}_{train_start_time_str}.pt")


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
                labels = input_tuple[-1]

                logits = self.model(input_tuple)

                loss = self.loss_fxn(logits, labels)
                loss.backward()

                self.optimizer.step()
                if self.run_steps > 0 and step == self.run_steps - 1:
                    early_stopping = True
                    break

                if step % self.eval_steps == 0:
                    corr, total = evaluate_and_predict(self.data.dev, self.model,
                                                       eval_batch_size=self.eval_batch_size,
                                                       use_collate=self.use_collate,
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
                        if model_save_path:
                            torch.save(best_model_checkpoint, model_save_path)

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
                print("Best model saved in {}".format(model_save_path))

        return best_model_checkpoint, model_save_path


def evaluate_and_predict(dataset, model, eval_batch_size=64, use_collate=True, batch_first=False):
    model.eval()
    with torch.no_grad():
        total_preds = 0
        correct_preds = 0
        # preds_and_labels = []
        batch_size = eval_batch_size

        if use_collate:
            collate_fn = lambda batch: datasets.my_collate(batch, batch_first=batch_first)
            dataloader = DataLoader(dataset, batch_size=batch_size,
                                    shuffle=False, collate_fn=collate_fn)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        # bad_sentences = []

        for tuple in dataloader:
            tuple = [x.to(model.device) for x in tuple]
            y_batch = tuple[-1]

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
    if device:
        checkpoint = torch.load(save_path, map_location=device)
    else:
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


