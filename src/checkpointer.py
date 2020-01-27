import os
import torch


class Checkpointer():
    def __init__(self,
                 serialization_dir,
                 keep_all_serialized_models):
        self._serialization_dir = serialization_dir
        os.makedirs(serialization_dir)
        self._keep_all_serialized_models = keep_all_serialized_models

        self._best_model_path = None

    def save(self, epoch, model_state, is_best):
        model_path = os.path.join(
            self._serialization_dir,
            "model_state_epoch_{}.th".format(epoch)
        )
        torch.save(model_state, model_path)

        if not self._keep_all_serialized_models:
            previous_model_path = os.path.join(
                self._serialization_dir,
                "model_state_epoch_{}.th".format(epoch-1)
            )
            if os.path.isfile(previous_model_path):
                os.remove(previous_model_path)

        if is_best:
            model_path = os.path.join(
                self._serialization_dir,
                "best_state_epoch_{}.th".format(epoch)
            )
            torch.save(model_state, model_path)

            previous_best_model_path = self._best_model_path
            if previous_best_model_path is not None and \
               os.path.isfile(previous_best_model_path):
                os.remove(previous_best_model_path)
            self._best_model_path = model_path

    @classmethod
    def restore(cls, model_path, device):
        model_state = torch.load(model_path, device)
        return model_state

    def get_best_model_path(self):
        return self._best_model_path
