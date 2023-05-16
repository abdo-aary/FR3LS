import logging
import os

import gin
import numpy as np
from fire import Fire

from common.experiment import Experiment
from common.settings import DATASETS_PATH
from common.torch.snapshots import SnapshotManager
from common.utils import count_parameters
from experiments.trainers.trainer_ae import raw_t2v_trainer
import torch as t

from common.neg_sampler import TimeSeriesSampler as Neg_TimeSeriesSampler
from common.neg_sampler_traffic import TimeSeriesSampler as Neg_TimeSeriesSamplerTraffic
from models.ts2vec_models.encoder import TSEncoder


class TS2VecExperiment(Experiment):
    @gin.configurable
    def instance(self,
                 ts_dataset_name: str,

                 train_window: int,
                 train_mode: str,  # Alternative or not
                 f_input_window: int,
                 f_out_same_in_size: bool,
                 horizon: int,
                 n_test_windows: int,
                 n_val_windows: int,
                 non_overlap_batch: int,
                 shuffle: bool,

                 neg_samples_jump: int = None,

                 ts2vec_output_dims: int = 320,
                 ts2vec_hidden_dims: int = 64,
                 ts2vec_depth: int = 10,
                 ts2vec_mask_mode: str = 'binomial',

                 train_ts2vec_loss: str = 'HIER',

                 epochs: int = 750,
                 batch_size: int = 8,
                 random_state: int = 42,
                 learning_rate: float = 0.001,
                 repeat: int = 0,
                 verbose: bool = True,
                 early_stopping: bool = False,
                 patience: int = 5,
                 ) -> None:

        t.manual_seed(random_state)

        if verbose:
            print("Data loading ...")

        dataset_path = os.path.join(DATASETS_PATH, ts_dataset_name, ts_dataset_name + '.npy')
        ts_samples = np.load(dataset_path).transpose()  # ts_samples of shape (T, N)

        # Start data Pretreatment ######################

        # Work only with series not having zero as std
        train_end_time_point = ts_samples.shape[0] - (
                n_val_windows + n_test_windows) * horizon - train_window
        train_ts_std = ts_samples[:train_end_time_point + train_window].std(axis=0)

        used_mask = np.where(train_ts_std != 0)[0]
        ts_samples = ts_samples[:, used_mask]
        # End data Pretreatment ########################

        Neg_Sampler = Neg_TimeSeriesSampler if ts_dataset_name == 'electricity' else Neg_TimeSeriesSamplerTraffic

        ts_sampler = Neg_Sampler(timeseries=ts_samples,
                                 train_window=train_window,
                                 batch_size=batch_size,
                                 f_input_window=f_input_window,
                                 horizon=horizon,
                                 non_overlap_batch=non_overlap_batch,
                                 n_test_windows=n_test_windows,
                                 n_val_windows=n_val_windows,
                                 neg_samples_jump=neg_samples_jump,
                                 shuffle=shuffle)

        if verbose:
            print("\n\nModel Training ...")
        input_dim = ts_samples.shape[-1]

        model = TSEncoder(input_dims=input_dim,
                          output_dims=ts2vec_output_dims,
                          hidden_dims=ts2vec_hidden_dims,
                          depth=ts2vec_depth,
                          mask_mode=ts2vec_mask_mode)

        if verbose:
            print(model)
            print(f'parameter count: {count_parameters(model)}')

        # Train model
        snapshot_manager = SnapshotManager(
            snapshot_dir=os.path.join(self.root, 'snapshots'),
            losses=['training', 'testing'],
        )

        _ = raw_t2v_trainer(snapshot_manager=snapshot_manager,
                            model=model,
                            training_set=iter(ts_sampler),
                            sampler=ts_sampler,
                            horizon=horizon,
                            train_ts2vec_loss=train_ts2vec_loss,
                            epochs=epochs,
                            learning_rate=learning_rate,
                            verbose=verbose,
                            early_stopping=early_stopping,
                            patience=patience, )

        snapshot_manager.print_losses()

        if verbose:
            print("\n\n##############################################################")
            print("######################## || DONE :) || #######################")
            print("##############################################################")


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire(TS2VecExperiment)
