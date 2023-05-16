import os
import logging
import gin
import torch as t
from fire import Fire
from common.neg_sampler import TimeSeriesSampler
from common.neg_sampler_traffic import TimeSeriesSampler as TimeSeriesSamplerTraffic
from common.experiment import Experiment
from common.settings import DATASETS_PATH
from common.torch.snapshots import SnapshotManager
from common.utils import count_parameters
import numpy as np
from experiments.trainers.trainer import ts2vec_trainer
from models.aldy.tlae import TLAE

from models.ts2vec_models.encoder import TSEncoder

from common.settings import EXPERIMENTS_PATH


class TS2VecExperiment(Experiment):
    @gin.configurable
    def instance(self,
                 ts_dataset_name: str,

                 train_window: int,
                 f_input_window: int,
                 f_out_same_in_size: bool,
                 horizon: int,
                 n_test_windows: int,
                 n_val_windows: int,
                 non_overlap_batch: int,
                 shuffle: bool,

                 ae_hidden_dims: list[int],

                 f_model_path: str,
                 f_model_type: str,
                 f_num_channels: list[int] = None,
                 f_kernel_size: int = None,
                 f_dropout: int = None,
                 f_leveld_init: bool = None,

                 neg_samples_jump: int = None,

                 # LSTM model
                 f_hidden_size: int = None,
                 f_num_layers: int = None,

                 ts2vec_encoder_path: str = None,  # Path to pretrained ts2vec_model
                 ts2vec_output_dims: int = 320,
                 ts2vec_hidden_dims: int = 64,
                 ts2vec_depth: int = 10,
                 ts2vec_mask_mode: str = 'binomial',

                 activation: str = 'gelu',
                 loss_name: str = 'HIER',

                 epochs: int = 100,
                 batch_size: int = 32,
                 random_state: int = 42,
                 learning_rate: float = 0.001,
                 repeat: int = 0,
                 verbose: bool = True,
                 early_stopping: bool = True,
                 patience: int = 5,
                 ) -> None:

        t.manual_seed(random_state)

        if verbose:
            print("Data loading ...")

        dataset_path = os.path.join(DATASETS_PATH, ts_dataset_name, ts_dataset_name + '.npy')
        ts_samples = np.load(dataset_path).transpose()  # ts_samples of shape (T, N)

        #### Start data Pretreatment ####
        # Work only with series not having zero as std
        train_end_time_point = ts_samples.shape[0] - (
                n_val_windows + n_test_windows) * horizon - train_window
        train_ts_std = ts_samples[:train_end_time_point + train_window].std(axis=0)

        used_mask = np.where(train_ts_std != 0)[0]
        ts_samples = ts_samples[:, used_mask]
        #### End data Pretreatment ####

        Sampler = TimeSeriesSampler if ts_dataset_name == 'electricity' else TimeSeriesSamplerTraffic

        ts_sampler = Sampler(timeseries=ts_samples,
                             train_window=train_window,
                             batch_size=batch_size,
                             f_input_window=f_input_window,
                             horizon=horizon,
                             non_overlap_batch=non_overlap_batch,
                             n_test_windows=n_test_windows,
                             n_val_windows=n_val_windows,
                             neg_samples_jump=neg_samples_jump,
                             shuffle=shuffle)

        idx_hidden_dim = 0
        for i in range(len(ae_hidden_dims) - 1):
            if ae_hidden_dims[i] == ae_hidden_dims[i + 1]:
                idx_hidden_dim = i
                break
        latent_dim = ae_hidden_dims[idx_hidden_dim]
        f_num_inputs = latent_dim
        f_output_size = latent_dim

        if verbose:
            print("\n\nModel Training ...")
        input_dim = ts_samples.shape[-1]

        model = TSEncoder(input_dims=latent_dim,
                          output_dims=ts2vec_output_dims,
                          hidden_dims=ts2vec_hidden_dims,
                          depth=ts2vec_depth,
                          mask_mode=ts2vec_mask_mode)

        if verbose:
            print(model)
            print(f'parameter count: {count_parameters(model)}')

        ############################ Load f_model  ############################
        if f_model_type == 'TCN_Modified':
            f_model_params = {'model_type': f_model_type,
                              'num_inputs': f_num_inputs,
                              'output_size': f_output_size,
                              'num_channels': f_num_channels,
                              'kernel_size': f_kernel_size,
                              'dropout': f_dropout,
                              'leveld_init': f_leveld_init}

        elif f_model_type == 'LSTM_Modified':
            f_model_params = {'model_type': f_model_type,
                              'input_size': f_num_inputs,
                              'output_size': f_output_size,
                              'hidden_size': f_hidden_size,
                              'num_layers': f_num_layers,
                              'dropout': f_dropout,
                              'batch_first': True}
        else:
            raise Exception(f"Unknown f_model {f_model_type}")

        f_model_path_root = os.path.join(EXPERIMENTS_PATH, 'tlae_exps', ts_dataset_name, f_model_path)

        f_model_snapshot_manager = SnapshotManager(
            snapshot_dir=os.path.join(f_model_path_root, 'snapshots'),
        )
        f_model = TLAE(input_dim=input_dim,
                       ae_hidden_dims=ae_hidden_dims,
                       f_model_params=f_model_params,
                       f_input_window=f_input_window,
                       train_window=train_window,
                       f_out_same_in_size=f_out_same_in_size,
                       activation=activation)

        _ = f_model_snapshot_manager.restore(model=f_model, optimizer=None)
        ########################### END Load f_model ##########################

        # Train model
        snapshot_manager = SnapshotManager(
            snapshot_dir=os.path.join(self.root, 'snapshots'),
        )

        _ = ts2vec_trainer(snapshot_manager=snapshot_manager,
                           model=model,
                           f_model=f_model,
                           training_set=iter(ts_sampler),
                           sampler=ts_sampler,
                           loss_name=loss_name,
                           epochs=epochs,
                           learning_rate=learning_rate,
                           verbose=verbose,
                           early_stopping=early_stopping,
                           patience=patience)

        snapshot_manager.print_losses()

        if verbose:
            print("\n\n##############################################################")
            print("######################## || DONE :) || #######################")
            print("##############################################################")


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire(TS2VecExperiment)
