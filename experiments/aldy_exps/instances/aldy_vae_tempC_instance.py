import logging
import os

import gin
import numpy as np
from fire import Fire

from common.experiment import Experiment
from common.settings import DATASETS_PATH
from common.torch.snapshots import SnapshotManager
from common.utils import count_parameters
from experiments.trainers.trainer_ae import vae_tempC_trainer
from models.aldy.aldy_vae_tempC import ALDy
import torch as t

from common.neg_sampler import TimeSeriesSampler as Neg_TimeSeriesSampler
from common.neg_sampler_traffic import TimeSeriesSampler as Neg_TimeSeriesSamplerTraffic


class ALDyExperiment(Experiment):
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

                 ae_hidden_dims: list[int],

                 f_model_type: str,

                 neg_samples_jump: int = None,

                 # LSTM model
                 f_hidden_size: int = None,
                 f_num_layers: int = None,

                 f_dropout: int = 0.0,

                 activation: str = 'relu',

                 train_ae_loss: str = 'MAE',
                 train_forecasting_loss: str = 'MSE',
                 val_loss_name: str = 'MAPE',
                 train_tempC_loss: str = 'TempC',
                 train_distr_reg_loss: str = 'KLD',

                 mask_mode: str = 'binomial',

                 lambda_ae: int = 1,
                 lambda_f: int = 1,
                 lambda_tempC: int = 1,
                 lambda_distr: int = 1,

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
        idx_hidden_dim = np.where(np.array([i if ae_hidden_dims[i] == ae_hidden_dims[i + 1] else 0
                                            for i in range(len(ae_hidden_dims) - 1)]) != 0)[0][0]
        latent_dim = ae_hidden_dims[idx_hidden_dim]

        if f_model_type == 'LSTM_Modified':
            f_model_params = {'model_type': f_model_type,
                              'input_size': latent_dim,
                              'output_size': latent_dim,
                              'hidden_size': f_hidden_size,
                              'num_layers': f_num_layers,
                              'dropout': f_dropout,
                              'batch_first': True}
        # elif f_model_type == 'TCN_Modified':
        #     f_model_params = {'model_type': f_model_type,
        #                       'num_inputs': f_num_inputs,
        #                       'output_size': f_output_size,
        #                       'num_channels': f_num_channels,
        #                       'kernel_size': f_kernel_size,
        #                       'dropout': f_dropout,
        #                       'leveld_init': f_leveld_init}
        else:
            raise Exception(f"Unknown f_model {f_model_type}")

        model = ALDy(input_dim=input_dim,
                     ae_hidden_dims=ae_hidden_dims,
                     f_model_params=f_model_params,
                     mask_mode=mask_mode,
                     f_input_window=f_input_window,
                     train_window=train_window,
                     activation=activation)

        if verbose:
            print(model)
            print(f'parameter count: {count_parameters(model)}')

        # Train model
        snapshot_manager = SnapshotManager(
            snapshot_dir=os.path.join(self.root, 'snapshots'),
            losses=['training', 'testing'],
            other_losses=['MAPE', 'WAPE', 'SMAPE']
        )

        _ = vae_tempC_trainer(snapshot_manager=snapshot_manager,
                              model=model,
                              training_set=iter(ts_sampler),
                              sampler=ts_sampler,
                              horizon=horizon,
                              train_ae_loss=train_ae_loss,
                              train_forecasting_loss=train_forecasting_loss,
                              train_tempC_loss=train_tempC_loss,
                              train_distr_reg_loss=train_distr_reg_loss,
                              val_loss_name=val_loss_name,
                              lambda_ae=lambda_ae,
                              lambda_f=lambda_f,
                              lambda_tempC=lambda_tempC,
                              lambda_distr=lambda_distr,
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
    Fire(ALDyExperiment)
