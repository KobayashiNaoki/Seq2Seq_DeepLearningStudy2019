import torch
from checkpointer import Checkpointer
from tqdm import tqdm
import numpy as np
from pathlib import Path


class Trainer():
    def __init__(self,
                 config,
                 model,
                 optimizer,
                 scheduler,
                 train_iter,
                 valid_iter,
                 fields):
        self._config = config
        self._epochs = config.epochs
        self._serialization_dir = Path(config.serialization_dir)
        self._keep_all_serialized_models = config.keep_all_serialized_models
        self._log_file = config.log_file

        self._model = model
        self._optimizer = optimizer
        self._max_grad_norm = config.grad_norm
        self._scheduler = scheduler
        self._start_decay_epoch = config.start_decay_epoch
        self._checkpointer = Checkpointer(self._serialization_dir,
                                          self._keep_all_serialized_models)

        self._train_iter = train_iter
        self._valid_iter = valid_iter
        self._fields = fields

        if config.gpu < 0:
            self._device = torch.device('cpu')
        else:
            self._device = torch.device('cuda:0')
        self._model.to(self._device)
        print(self._model)

    def run(self, ):
        best_loss = np.inf
        is_best = False

        for epoch in range(1, self._epochs+1):
            if epoch >= self._start_decay_epoch:
                self._scheduler.step()
            print("epoch:", epoch)
            train_loss = self._train()
            valid_loss = self._valid()

            if valid_loss < best_loss:
                best_loss = valid_loss
                is_best = True
            else:
                is_best = False

            self._save(epoch, is_best)
            scores = {
                'train/loss': train_loss,
                'valid/loss': valid_loss,
            }
            self._report(epoch, scores)

        return

    def _train(self,):
        print('train model')
        model = self._model
        optimizer = self._optimizer
        total_loss = 0
        total_norm = 0
        model.train()
        for batch in tqdm(self._train_iter):
            optimizer.zero_grad()
            output_dict = model(batch.source_tokens, batch.target_tokens)
            loss = output_dict["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self._max_grad_norm)
            optimizer.step()

            loss = output_dict["loss"]
            total_loss += loss.item() * len(batch)
            total_norm += len(batch)

        return total_loss / total_norm

    def _valid(self,):
        print('valid model')
        model = self._model
        total_loss = 0
        total_norm = 0
        model.eval()
        for batch in tqdm(self._valid_iter):
            with torch.no_grad():
                output_dict = model(batch.source_tokens, batch.target_tokens)
            loss = output_dict["loss"]
            total_loss += loss.item() * len(batch)
            total_norm += len(batch)

        return total_loss / total_norm

    def _evaluate():
        pass

    def _save(self, epoch, is_best):
        print('save model')
        model_state = {
            "epoch": epoch,
            "model": self._model.state_dict(),
            "optim": self._optimizer.state_dict(),
            "sched": self._scheduler.state_dict(),
            "fields": self._fields,
            "config": self._config
        }
        self._checkpointer.save(epoch, model_state, is_best)
        return

    def get_best_model_path(self):
        return self._checkpointer.get_best_model_path()

    def _report(self, epoch, scores):
        # lr = self._optimizer.lr
        print("epoch: {}".format(epoch))
        # print("lr: {}".format(lr))
        print("scores:")
        for k, v in scores.items():
            print("\t{} {}".format(k, v))

        with open(self._serialization_dir / self._log_file, "a") as f:
            print("\t".join(map(str, [epoch, *scores.values()])), file=f)

        return
