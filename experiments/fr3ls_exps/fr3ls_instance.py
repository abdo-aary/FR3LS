import gin
import logging
import numpy as np
import os
import torch as t
from fire import Fire
from torch.utils.data import DataLoader

from common.experiment import Experiment
# from common.determinist_sampler import DeterministSampler
# from common.probabilist_sampler import worker_init_fn, ProbabilistSampler
from common.sampler import Sampler, worker_init_fn
from common.settings import DATASETS_PATH
from common.torch.ops import torch_dtype_dict
from common.torch.snapshots import SnapshotManager
from common.utils import count_parameters
from datasets.load_data_gluonts import load_data_gluonts
from experiments.trainers.trainer import trainer_determinist, trainer_probabilist
from models.fr3ls.fe3ls_determinist import FR3LS_Determinist
from models.fr3ls.fr3ls_probabilist import FR3LS_Probabilist


class FR3LS_Experiment(Experiment):
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

                 dropout: float = 0.0,  # Dropout used in the AE

                 # LSTM model
                 f_hidden_size: int = None,
                 f_num_layers: int = None,

                 f_dropout: float = 0.0,  # Dropout used in the f_model

                 activation: str = 'relu',

                 train_ae_loss: str = 'MAE',
                 train_forecasting_loss: str = 'MSE',
                 test_loss_name: str = 'MAPE',
                 train_temp_loss: str = 'TempNC',

                 mask_mode: str = 'binomial',

                 lambda_ae: float = 1,
                 lambda_f: float = 1,
                 lambda_temp: float = 1,
                 lambda_NC: float = 5e-3,

                 epochs: int = 750,
                 batch_size: int = 8,
                 random_state: int = 42,
                 model_random_seed: int = 3407,
                 learning_rate: float = 0.001,
                 repeat: int = 0,
                 num_workers: int = 0,
                 verbose: bool = True,
                 pbar_percentage: int = 20,
                 early_stopping: bool = False,
                 patience: int = 5,
                 used_dtype: str = 'float32',
                 device_id: int = None,
                 n_best_test_losses: int = 50,
                 lr_warmup: int = 10_000,
                 skip_end_n_val: bool = False,
                 pretrain_epochs: int = 0,
                 pbbilist_modeling: bool = False,
                 num_samples: int = 1000,

                 ) -> None:

        t.manual_seed(random_state)
        t.set_default_dtype(torch_dtype_dict[used_dtype])

        if verbose:
            print("Data loading ...")

        if not train_window:
            train_window = 2 * f_input_window  # w = 2 * L

        if not ae_hidden_dims:
            if not encoder_dims:
                raise Exception('params \'ae_hidden_dims\' and \'encoder_dims\' can\'t both be None')

            ae_hidden_dims = list(encoder_dims)
            encoder_dims.reverse()
            ae_hidden_dims += encoder_dims

        # Start data Pretreatment ######################
        if pbbilist_modeling and ts_dataset_name != 'traffic':
            # Only traffic data is not loaded using "load_data_gluonts" function

            train_data, test_data_in, test_data_target = load_data_gluonts(name=ts_dataset_name,
                                                                           prediction_length=horizon,
                                                                           num_test_samples=f_input_window,
                                                                           dtype=torch_dtype_dict[used_dtype])
            train_data = train_data.detach().cpu().numpy()
            test_data_in = test_data_in.detach().cpu().numpy()
            test_data_target = test_data_target.detach().cpu().numpy()

            train_ts_std = train_data.std(axis=0)
            used_mask = np.where(train_ts_std != 0)[0]

            train_data = train_data[:, used_mask]
            test_data_in = test_data_in[:, :, used_mask]
            test_data_target = test_data_target[:, :, used_mask]

            print("train_data.shape =", train_data.shape)
            print("test_data_in.shape =", test_data_in.shape)
            print("test_data_target.shape =", test_data_target.shape)

            input_dim = train_data.shape[-1]

            sampler = Sampler(train_data=train_data,
                              test_data_in=test_data_in,
                              test_data_target=test_data_target,
                              train_window=train_window,
                              f_input_window=f_input_window,
                              horizon=horizon,
                              non_overlap_batch=non_overlap_batch,
                              n_test_windows=n_test_windows,
                              n_val_windows=n_val_windows, )
        else:
            dataset_path = os.path.join(DATASETS_PATH, ts_dataset_name, ts_dataset_name + '.npy')
            ts_samples = np.load(dataset_path).transpose()  # ts_samples of shape (T, N)
            ts_samples = ts_samples.astype(np.dtype(used_dtype))

            # Work only with series not having zero as std
            train_end_time_point = ts_samples.shape[0] - (
                    n_val_windows + n_test_windows) * horizon - train_window
            train_ts_std = np.nanstd(ts_samples[:train_end_time_point + train_window], axis=0)

            used_mask = np.where(train_ts_std != 0)[0]  # We only use variables that don't have 0 as std
            ts_samples = ts_samples[:, used_mask]
            print("ts_samples.shape =", ts_samples.shape)
            # End data Pretreatment ########################

            input_dim = ts_samples.shape[-1]

            sampler = Sampler(timeseries=ts_samples,
                              train_window=train_window,
                              f_input_window=f_input_window,
                              horizon=horizon,
                              non_overlap_batch=non_overlap_batch,
                              n_test_windows=n_test_windows,
                              n_val_windows=n_val_windows,
                              skip_end_n_val=(ts_dataset_name == 'electricity' and skip_end_n_val))

        dataLoader = DataLoader(sampler, batch_size=batch_size, num_workers=num_workers,
                                worker_init_fn=worker_init_fn, drop_last=False, pin_memory=True)

        print('sampler =', sampler)

        if verbose:
            print("\n\nModel Training ...")
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
                              'batch_first': True,
                              }
        else:
            raise Exception(f"Unknown f_model {f_model_type}")

        Model = FR3LS_Probabilist if pbbilist_modeling else FR3LS_Determinist

        print('Model =', Model)

        model = Model(input_dim=input_dim,
                      ae_hidden_dims=ae_hidden_dims,
                      f_model_params=f_model_params,
                      mask_mode=mask_mode,
                      f_input_window=f_input_window,
                      train_window=train_window,
                      dropout=dropout,
                      activation=activation,
                      model_random_seed=model_random_seed,
                      )

        if verbose:
            print(model)
            print(f'parameter count: {count_parameters(model)}')

        # Train model
        snapshot_manager = SnapshotManager(
            snapshot_dir=os.path.join(self.root, 'snapshots'),
            losses=['training', 'testing'],
            other_losses=['MAPE', 'WAPE', 'SMAPE'] if not pbbilist_modeling else [],
        )

        trainer = trainer_probabilist if pbbilist_modeling else trainer_determinist

        print('trainer =', trainer)

        trainer(snapshot_manager=snapshot_manager,
                model=model,
                dataLoader=dataLoader,
                sampler=sampler,
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
                pretrain_epochs=pretrain_epochs,
                num_samples=num_samples)

        if verbose:
            print("\n\n##############################################################")
            print("######################## || DONE :) || #######################")
            print("##############################################################")


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire(FR3LS_Experiment)
