import logging
import os

import gin
import numpy as np
import torch as t
from fire import Fire
from torch.utils.data import DataLoader

from torch.optim.swa_utils import AveragedModel

from common.experiment import Experiment
# from common.sampler_all import TimeSeriesSampler, worker_init_fn
from common.sampler_all_noisy import TimeSeriesSampler, worker_init_fn
from common.settings import DATASETS_PATH
from common.torch.ops import torch_dtype_dict
from common.torch.snapshots import SnapshotManager
from common.utils import count_parameters
from experiments.trainers.trainer_ae import ae_temp_trainer
from models.aldy.aldy_ae_temp import ALDy as ALDy_vanilla_ae
from models.aldy.aldy_conv_Interpol.aldy_conv_Interpol_ae_temp import ALDy as ALDy_Conv_Inter_ae
from models.aldy.aldy_conv_Interpol.aldy_conv_Interpol_ae_temp_v2 import ALDy as ALDy_Conv_Inter_ae_v2
from models.aldy.aldy_conv_Interpol.aldy_conv_Interpol_ae_temp_v3 import ALDy as ALDy_Conv_Inter_ae_v3
from models.aldy.aldy_conv_ae_temp import ALDy as ALDy_Vanilla_conv_ae


class ALDyExperiment(Experiment):
    @gin.configurable
    def instance(self,
                 ts_dataset_name: str,

                 train_mode: str,  # Alternative or not
                 f_input_window: int,
                 f_out_same_in_size: bool,
                 horizon: int,
                 n_test_windows: int,
                 n_val_windows: int,
                 non_overlap_batch: int,
                 shuffle: bool,

                 f_model_type: str,

                 ae_hidden_dims: list[int] = None,
                 encoder_dims: list[int] = None,

                 train_window: int = None,

                 neg_samples_jump: int = None,

                 dropout: float = 0.0,  # Dropout used in the AE

                 # LSTM model
                 f_hidden_size: int = None,
                 f_num_layers: int = None,

                 # TCN model
                 f_num_channels: list[int] = None,
                 f_kernel_size: int = None,
                 f_leveld_init: bool = None,

                 f_dropout: float = 0.0,  # Dropout used in the f_model

                 ts2vec_output_dims: int = None,
                 ts2vec_hidden_dims: int = None,
                 ts2vec_depth: int = None,
                 ts2vec_mask_mode: str = None,

                 activation: str = 'relu',

                 train_ae_loss: str = 'MAE',
                 train_forecasting_loss: str = 'MSE',
                 test_loss_name: str = 'MAPE',
                 train_temp_loss: str = 'TempC',

                 mask_mode: str = 'binomial',
                 augV_method: str = 'noise',

                 lambda_ae: float = 1,
                 lambda_f: float = 1,
                 lambda_temp: float = 1,
                 lambda_NC: float = 5e-3,

                 epochs: int = 750,
                 batch_size: int = 8,
                 random_state: int = 42,
                 learning_rate: float = 0.001,
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
                 type_augV_latent: str = None,
                 direct_decoding: bool = False,
                 swa: bool = False,
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

        if not ae_hidden_dims:
            if not encoder_dims:
                raise Exception('params \'ae_hidden_dims\' and \'encoder_dims\' can\'t both be None')

            ae_hidden_dims = list(encoder_dims)
            encoder_dims.reverse()
            ae_hidden_dims += encoder_dims

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

        dataloader = DataLoader(ts_sampler, batch_size=batch_size, num_workers=num_workers,
                                worker_init_fn=worker_init_fn, drop_last=False, pin_memory=True)

        print('ts_sampler =', ts_sampler)

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

        elif f_model_type == 'TCN_Modified':
            f_model_params = {'model_type': f_model_type,
                              'num_inputs': latent_dim,
                              'output_size': latent_dim,
                              'num_channels': f_num_channels,
                              'kernel_size': f_kernel_size,
                              'dropout': f_dropout,
                              'leveld_init': f_leveld_init}

        else:
            raise Exception(f"Unknown f_model {f_model_type}")

        if 'VANILLA_CONV' in train_mode:
            ALDy = ALDy_Vanilla_conv_ae
        elif 'CONV_INTER_v2' in train_mode:
            ALDy = ALDy_Conv_Inter_ae_v2
        elif 'CONV_INTER_v3' in train_mode:
            ALDy = ALDy_Conv_Inter_ae_v3
        elif 'CONV_INTER' in train_mode:
            ALDy = ALDy_Conv_Inter_ae
        else:
            ALDy = ALDy_vanilla_ae

        model = ALDy(input_dim=input_dim,
                     ae_hidden_dims=ae_hidden_dims,
                     f_model_params=f_model_params,
                     mask_mode=mask_mode,
                     f_input_window=f_input_window,
                     train_window=train_window,
                     ts2vec_output_dims=ts2vec_output_dims,
                     ts2vec_hidden_dims=ts2vec_hidden_dims,
                     ts2vec_depth=ts2vec_depth,
                     ts2vec_mask_mode=ts2vec_mask_mode,
                     dropout=dropout,
                     activation=activation,
                     type_augV_latent=type_augV_latent,
                     direct_decoding=direct_decoding,
                     augV_method=augV_method)

        avg_model = None
        if swa:
            avg_model = AveragedModel(model)
            avg_model.update_parameters(model)

        if verbose:
            print(model)
            print(f'parameter count: {count_parameters(model)}')

        # Train model
        snapshot_manager = SnapshotManager(
            snapshot_dir=os.path.join(self.root, 'snapshots'),
            losses=['training', 'testing'],
            other_losses=['MAPE', 'WAPE', 'SMAPE']
        )

        _ = ae_temp_trainer(snapshot_manager=snapshot_manager,
                            model=model,
                            avg_model=avg_model,
                            dataLoader=dataloader,
                            horizon=horizon,
                            train_ae_loss=train_ae_loss,
                            train_forecasting_loss=train_forecasting_loss,
                            train_temp_loss=train_temp_loss,
                            test_loss_name=test_loss_name,
                            lambda_ae=lambda_ae,
                            lambda_f=lambda_f,
                            lambda_NC=lambda_NC,
                            lambda_temp=lambda_temp,
                            epochs=epochs,
                            learning_rate=learning_rate,
                            verbose=verbose,
                            pbar_percentage=pbar_percentage,
                            early_stopping=early_stopping,
                            patience=patience,
                            device_id=device_id,
                            n_best_test_losses=n_best_test_losses,
                            lr_warmup=lr_warmup,
                            swa=swa,)

        if verbose:
            print("\n\n##############################################################")
            print("######################## || DONE :) || #######################")
            print("##############################################################")


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire(ALDyExperiment)
