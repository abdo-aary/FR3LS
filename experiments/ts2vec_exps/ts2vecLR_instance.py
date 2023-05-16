import logging
import os

import gin
import numpy as np
import torch as t
from fire import Fire
from torch.utils.data import DataLoader, TensorDataset

from common.experiment import Experiment
from common.sampler_all_noisy import TimeSeriesSampler
from common.settings import DATASETS_PATH
from common.torch.ops import torch_dtype_dict, split_with_nan
from common.torch.snapshots import SnapshotManager
from common.utils import count_parameters
from experiments.trainers.trainer_ae import t2vLR_trainer
from models.ts2vec_models.encoder import TSEncoder


class TS2VecExperiment(Experiment):
    @gin.configurable
    def instance(self,
                 ts_dataset_name: str,

                 train_window: int,
                 train_mode: str,
                 f_input_window: int,
                 f_out_same_in_size: bool,
                 horizon: int,
                 n_test_windows: int,
                 n_val_windows: int,
                 non_overlap_batch: int,
                 shuffle: bool,

                 max_train_length: int = 3000,

                 ts2vec_output_dims: int = 320,
                 ts2vec_hidden_dims: int = 64,
                 ts2vec_depth: int = 10,
                 ts2vec_mask_mode: str = 'binomial',

                 ts2vec_loss_fn: str = 'HIER',
                 train_LR_loss: str = 'MSE',
                 test_LR_loss: str = 'MSE_MAE',

                 epochs_t2v: int = 750,
                 epochs_LR: int = 750,
                 batch_size: int = 8,
                 random_state: int = 42,
                 learning_rate_t2v: float = 0.0001,
                 learning_rate_LR: float = 0.0001,
                 repeat: int = 0,
                 num_workers: int = 0,
                 verbose: bool = True,
                 pbar_percentage: int = 20,
                 early_stopping: bool = False,
                 patience: int = 5,
                 used_dtype: str = 'float32',
                 device_id: int = None,
                 n_best_test_losses: int = None,
                 lr_warmup: int = None,
                 noise_level: float = None,
                 skip_end_n_val: bool = False,
                 ) -> None:

        t.manual_seed(random_state)
        t.set_default_dtype(torch_dtype_dict[used_dtype])

        if verbose:
            print("Data loading ...")

        dataset_path = os.path.join(DATASETS_PATH, ts_dataset_name, ts_dataset_name + '.npy')
        ts_samples = np.load(dataset_path).transpose()  # ts_samples of shape (T, N)
        ts_samples = ts_samples.astype(np.dtype(used_dtype))

        if not train_window:
            train_window = 2 * f_input_window

        # Start data Pretreatment ######################
        # Work only with series not having zero as std
        train_end_time_point = ts_samples.shape[0] - (
                n_val_windows + n_test_windows) * horizon - train_window
        train_ts_std = ts_samples[:train_end_time_point + train_window].std(axis=0)

        used_mask = np.where(train_ts_std != 0)[0]
        ts_samples = ts_samples[:, used_mask]
        # End data Pretreatment ########################

        ts_sampler = TimeSeriesSampler(timeseries=ts_samples,
                                       train_window=train_window,
                                       f_input_window=f_input_window,
                                       horizon=horizon,
                                       non_overlap_batch=non_overlap_batch,
                                       n_test_windows=n_test_windows,
                                       n_val_windows=n_val_windows,
                                       skip_end_n_val=(ts_dataset_name == 'electricity' and skip_end_n_val),
                                       noise_level=noise_level)

        train_data = ts_sampler.get_train_data()  # shape (T, N)
        train_data = train_data.reshape(train_data.shape[1], train_data.shape[0], 1)  # shape (N, T, 1), i,e, (n_instance, n_timestamps, n_features)

        if max_train_length:
            sections = train_data.shape[1] // max_train_length
            if sections >= 2:
                train_data = np.concatenate(split_with_nan(train_data, sections, axis=1), axis=0)

        print("train_data.shape =", train_data.shape)

        train_dataset = TensorDataset(t.from_numpy(train_data))

        dataloader = DataLoader(train_data, batch_size=min(batch_size, len(train_dataset)), num_workers=num_workers,
                                shuffle=shuffle, drop_last=False, pin_memory=True)

        print('ts_sampler =', ts_sampler)

        if verbose:
            print("\n\nModel Training ...")
        # input_dim = ts_samples.shape[-1]
        input_dim = 1

        model = TSEncoder(input_dims=input_dim,
                          output_dims=ts2vec_output_dims,
                          hidden_dims=ts2vec_hidden_dims,
                          depth=ts2vec_depth,
                          mask_mode=ts2vec_mask_mode)
        # avg_model = AveragedModel(model)
        # avg_model.update_parameters(model)


        linear_regression = t.nn.Linear(in_features=ts2vec_output_dims, out_features=input_dim * horizon)

        if verbose:
            print(model)
            print(f'parameter count: {count_parameters(model)}')

        # Train model
        snapshot_manager_t2v = SnapshotManager(
            snapshot_dir=os.path.join(self.root, 'snapshots_T2V'),
            losses=['training', 'testing'],
        )
        snapshot_manager_LR = SnapshotManager(
            snapshot_dir=os.path.join(self.root, 'snapshots_LR'),
            losses=['training', 'testing'],
            other_losses=['MAPE', 'WAPE', 'SMAPE']
        )

        _ = t2vLR_trainer(snapshot_manager_t2v=snapshot_manager_t2v,
                          snapshot_manager_LR=snapshot_manager_LR,
                          model=model,
                          # avg_model=avg_model,
                          avg_model=None,
                          sampler=ts_sampler,
                          linear_regression=linear_regression,
                          dataLoader=dataloader,
                          horizon=horizon,
                          ts2vec_loss_fn=ts2vec_loss_fn,
                          train_LR_loss=train_LR_loss,
                          test_LR_loss=test_LR_loss,
                          epochs_t2v=epochs_t2v,
                          epochs_LR=epochs_LR,
                          learning_rate_t2v=learning_rate_t2v,
                          learning_rate_LR=learning_rate_LR,
                          verbose=verbose,
                          pbar_percentage=pbar_percentage,
                          early_stopping=early_stopping,
                          patience=patience,
                          device_id=device_id,
                          n_best_test_losses=n_best_test_losses,
                          lr_warmup=lr_warmup,
                          max_train_length=max_train_length,)

        if verbose:
            print("\n\n##############################################################")
            print("######################## || DONE :) || #######################")
            print("##############################################################")


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire(TS2VecExperiment)
