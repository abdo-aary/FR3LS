import os
import logging
import gin
import torch as t
from fire import Fire
from common.neg_sampler import TimeSeriesSampler
from common.neg_sampler_traffic import TimeSeriesSampler as TimeSeriesSamplerTraffic
from common.experiment import Experiment
from common.settings import DATASETS_PATH, EXPERIMENTS_PATH
from common.torch.snapshots import SnapshotManager as SnapshotManagerTLAE
from common.utils import count_parameters
import numpy as np
from experiments.trainers.trainer import aldy_trainer_pretrained_no_warmup_neg
from models.aldy.aldy_param_gen_neg import ALDy
from models.aldy.tlae import TLAE
from models.ts2vec_models.encoder import TSEncoder


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
                 f_alpha: float,
                 f_dropout: int,
                 f_num_shared_channels: list[int] = None,
                 f_nbr_param_layers: int = None,
                 f_kernel_size: int = None,
                 f_leveld_init: bool = None,
                 f_implicit_batching: bool = None,

                 neg_samples_jump: int = None,

                 # LSTM model
                 f_hidden_size: int = None,
                 f_num_layers: int = None,

                 f_model_path: str = None,
                 ts2vec_path: str = None,

                 ts2vec_encoder_path: str = None,  # Path to pretrained ts2vec_model
                 ts2vec_output_dims: int = 320,
                 ts2vec_hidden_dims: int = 64,
                 ts2vec_depth: int = 10,
                 ts2vec_mask_mode: str = 'binomial',

                 activation: str = 'gelu',
                 train_loss_name: str = 'ALDY',
                 val_loss_name: str = 'MSE',
                 lambda1: int = 1,
                 lambda2: int = 1,
                 lambda3: int = 1,
                 train_ae_loss: str = 'MAE',
                 train_forecasting_loss: str = 'MAE',

                 epochs: int = 750,
                 epochs_TLAE: int = 750,
                 epochs_TS2VEC: int = 200,
                 epochs_ParamGen: int = 200,
                 iterations: int = 150,
                 alt_epoch_nbr: int = 40,
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

        if f_model_type == 'TCN_Parameterized':
            f_model_params = {'model_type': f_model_type,
                              'num_inputs': f_num_inputs,
                              'num_shared_channels': f_num_shared_channels,
                              'nbr_param_layers': f_nbr_param_layers,
                              'ts2vec_output_dims': ts2vec_output_dims,
                              'kernel_size': f_kernel_size,
                              'dropout': f_dropout,
                              'leveld_init': f_leveld_init,
                              'implicit_batching': f_implicit_batching,
                              'alpha': f_alpha}

        elif f_model_type == 'LSTM_ParamGEN':
            f_model_params = {'model_type': f_model_type,
                              'input_size': f_num_inputs,
                              'output_size': f_output_size,
                              'hidden_size': f_hidden_size,
                              'num_layers': f_num_layers,
                              'dropout': f_dropout,
                              'alpha': f_alpha,
                              'batch_first': True}
        else:
            raise Exception(f"Unknown f_model {f_model_type}")

        if verbose:
            print("\n\nModel Training ...")
        input_dim = ts_samples.shape[-1]

        model = ALDy(input_dim=input_dim,
                     ae_hidden_dims=ae_hidden_dims,
                     f_model_params=f_model_params,
                     f_input_window=f_input_window,
                     train_window=train_window,
                     f_out_same_in_size=f_out_same_in_size,
                     ts2vec_output_dims=ts2vec_output_dims,
                     ts2vec_hidden_dims=ts2vec_hidden_dims,
                     ts2vec_depth=ts2vec_depth,
                     ts2vec_mask_mode=ts2vec_mask_mode,
                     activation=activation)

        ############################ Load f_model & TS2VEC  ############################
        pretrained = False
        if (f_model_path is not None) and (ts2vec_path is not None):
            ###### f_model loading ######
            print("!!! Loading Pretrained Models !!! ")
            if f_model_type == 'TCN_Parameterized':
                pretrained_f_model_params = {'model_type': 'TCN_Modified',
                                             'num_inputs': f_num_inputs,
                                             'output_size': f_output_size,
                                             'num_channels': f_num_shared_channels,
                                             'kernel_size': f_kernel_size,
                                             'dropout': f_dropout,
                                             'leveld_init': f_leveld_init}
            elif f_model_type == 'LSTM_ParamGEN':
                pretrained_f_model_params = {'model_type': 'LSTM_Modified',
                                             'input_size': f_num_inputs,
                                             'output_size': f_output_size,
                                             'hidden_size': f_hidden_size,
                                             'num_layers': f_num_layers,
                                             'dropout': f_dropout,
                                             'batch_first': True}
            else:
                raise Exception(f"Unknown f_model {f_model_type}")

            f_model_path_root = os.path.join(EXPERIMENTS_PATH, 'tlae_exps', ts_dataset_name, f_model_path)

            f_model_snapshot_manager = SnapshotManagerTLAE(
                snapshot_dir=os.path.join(f_model_path_root, 'snapshots'),
            )
            f_model = TLAE(input_dim=input_dim,
                           ae_hidden_dims=ae_hidden_dims,
                           f_model_params=pretrained_f_model_params,
                           f_input_window=f_input_window,
                           train_window=train_window,
                           f_out_same_in_size=f_out_same_in_size,
                           activation=activation)

            _ = f_model_snapshot_manager.restore(model=f_model, optimizer=None)

            model.encoder = f_model.encoder
            model.decoder = f_model.decoder
            if f_model_type == 'TCN_Parameterized':
                model.f_model.tcn = f_model.f_model.tcn
                model.f_model.linear_out = f_model.f_model.linear_out
            elif f_model_type == 'LSTM_ParamGEN':
                model.f_model.lstm = f_model.f_model.lstm
                model.f_model.fc = f_model.f_model.fc

            ###### Ts2Vec loading ######
            ts2vec_path_root = os.path.join(EXPERIMENTS_PATH, 'ts2vec_exps', ts_dataset_name, ts2vec_path)

            ts2vec_snapshot_manager = SnapshotManagerTLAE(
                snapshot_dir=os.path.join(ts2vec_path_root, 'snapshots'),
            )
            ts2vec = TSEncoder(input_dims=latent_dim,
                               output_dims=ts2vec_output_dims,
                               hidden_dims=ts2vec_hidden_dims,
                               depth=ts2vec_depth,
                               mask_mode=ts2vec_mask_mode)
            _ = ts2vec_snapshot_manager.restore(model=ts2vec, optimizer=None)

            model.ts2vec_encoder = ts2vec
            pretrained = True

        ########################## END Load f_model & TS2VEC  ##########################

        if verbose:
            print(model)
            print(f'parameter count: {count_parameters(model)}')

        # Train model
        snapshot_manager = SnapshotManagerTLAE(
            snapshot_dir=os.path.join(self.root, 'snapshots'),
            losses=['training', 'testing'],
            other_losses=['WAPE', 'SMAPE']
        )

        trainer = aldy_trainer_pretrained_no_warmup_neg

        _ = trainer(snapshot_manager=snapshot_manager,
                    model=model,
                    training_set=iter(ts_sampler),
                    sampler=ts_sampler,
                    horizon=horizon,
                    train_loss_name=train_loss_name,
                    val_loss_name=val_loss_name,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    lambda3=lambda3,
                    epochs=epochs,
                    epochs_TLAE=epochs_TLAE,
                    epochs_TS2VEC=epochs_TS2VEC,
                    epochs_ParamGen=epochs_ParamGen,
                    iterations=iterations,
                    alt_epoch_nbr=alt_epoch_nbr,
                    learning_rate=learning_rate,
                    verbose=verbose,
                    early_stopping=early_stopping,
                    patience=patience,
                    pretrained=pretrained)

        snapshot_manager.print_losses()

        if verbose:
            print("\n\n##############################################################")
            print("######################## || DONE :) || #######################")
            print("##############################################################")


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire(ALDyExperiment)
