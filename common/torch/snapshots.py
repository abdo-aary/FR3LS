import os
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np
import gin
import torch as t


@gin.configurable()
class SnapshotManager:
    """
    Snapshot Manager.
    Only one, the "latest", state is supported.
    """

    def __init__(self, snapshot_dir: str, losses: list = ['training', 'testing'], other_losses: list = [], verbose: bool = True):
        self.model_snapshot_file = os.path.join(snapshot_dir, 'model')
        self.optimizer_snapshot_file = os.path.join(snapshot_dir, 'optimizer')
        self.losses_file = os.path.join(snapshot_dir, 'losses')
        self.losses_csv = os.path.join(snapshot_dir, 'losses.csv')
        self.epoch_file = os.path.join(snapshot_dir, 'epoch')
        self.time_tracking_file = os.path.join(snapshot_dir, 'time')
        self.start_time = None
        self.other_losses = other_losses
        other_losses_dict = {loss_name: {} for loss_name in other_losses}
        self.losses = {loss_name: {} for loss_name in losses}
        self.losses.update(other_losses_dict)
        self.time_track = {}
        self.verbose = verbose

        self.enable_time_tracking()

    def restore(self, model: Optional[t.nn.Module], optimizer: Optional[t.optim.Optimizer]) -> int:
        """
        Restore a model and optimizer, by mutating their state, and return the epoch number on which
        the state was persisted.

        :param model: Model architecture, weights of which should be restored.
        :param optimizer: Optimizer instance, parameters of which should be restored.
        :return: epoch number.
        """
        if model is not None and os.path.isfile(self.model_snapshot_file):
            if self.verbose:
                print("Model of path \"" + self.model_snapshot_file + "\" restored successfully")
            model.load_state_dict(t.load(self.model_snapshot_file))

        if optimizer is not None and os.path.isfile(self.optimizer_snapshot_file):
            if self.verbose:
                print("Optimizer of path \"" + self.optimizer_snapshot_file + "\" restored successfully")
            optimizer.load_state_dict(t.load(self.optimizer_snapshot_file))

        epoch = t.load(self.epoch_file)['epoch'] if os.path.isfile(self.epoch_file) else 0
        if os.path.isfile(self.losses_file):
            losses = t.load(self.losses_file)
            for key in self.losses.keys():
                if key in losses.keys():
                    loss = {k: v for k, v in losses[key].items() if k <= epoch}
                    self.losses[key] = loss

            self.snapshot(self.losses_file, self.losses)
        if os.path.isfile(self.time_tracking_file):
            self.time_track = t.load(self.time_tracking_file)
        return epoch

    def load_losses(self) -> pd.DataFrame:
        """
        Load training & testing into a dataframe.

        :return: Training losses in pandas DatFrame.
        """
        if os.path.isfile(self.losses_file):
            key = list(self.losses.keys())[0]
            losses_k = t.load(self.losses_file)[key]
            losses_k_df = pd.DataFrame({key + '_loss': list(losses_k.values())},
                                       index=list(losses_k.keys()))
            losses_df = losses_k_df

            for key in list(self.losses.keys())[1:]:
                losses_k = t.load(self.losses_file)[key]
                losses_k_df = pd.DataFrame({key + '_loss': list(losses_k.values())},
                                           index=list(losses_k.keys()))
                losses_df = pd.merge(losses_df, losses_k_df, left_index=True, right_index=True)

            return losses_df
        else:
            return pd.DataFrame([np.nan])

    def print_losses(self, top_best: int = None):
        """
        Print training & testing into a csv file.

        :return: None
        """
        losses_df = self.load_losses()
        if top_best:
            if "MAPE_loss" in losses_df.columns:
                losses_df = losses_df.sort_values(["MAPE_loss", "testing_loss"]).head(top_best)
            else:
                losses_df = losses_df.sort_values(["testing_loss"]).head(top_best)
        losses_df.to_csv(self.losses_csv, index=True)

    def enable_time_tracking(self) -> None:
        """
        Enable time tracking to estimate training time.
        """
        self.start_time = time.time()

    def register(self,
                 epoch: int,
                 training_loss: float,
                 testing_loss: float,
                 save_model: bool,
                 model: t.nn.Module,
                 other_losses_values: dict = {},
                 validation_loss: float = None,
                 optimizer: Optional[t.optim.Optimizer] = None) -> None:
        """
        Register an epoch.This method should be invoked after each epoch.
        """
        self.losses['training'][epoch] = training_loss
        self.losses['testing'][epoch] = testing_loss
        if validation_loss:
            self.losses['validation'][epoch] = validation_loss

        for key in self.other_losses:
            self.losses[key][epoch] = other_losses_values[key]

        self.snapshot(self.losses_file, self.losses)
        if save_model:  # Save only the best model
            self.snapshot(self.model_snapshot_file, model.state_dict())
            if optimizer is not None:
                self.snapshot(self.optimizer_snapshot_file, optimizer.state_dict())

        self.snapshot(self.epoch_file, {'epoch': epoch})
        if self.start_time is not None:
            self.time_track[epoch] = time.time() - self.start_time
            self.snapshot(self.time_tracking_file, self.time_track)
            self.start_time = time.time()

    @staticmethod
    def snapshot(path: str, data: Dict) -> None:
        """
        Atomic persistence for data dictionary.

        :param path: Where to persist.
        :param data: What to persist.
        """
        dir_path = os.path.dirname(path)
        if not os.path.isdir(dir_path):
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(dir=dir_path, delete=False, mode='wb')
        t.save(data, temp_file)
        temp_file.flush()
        os.fsync(temp_file.fileno())
        os.rename(temp_file.name, path)
