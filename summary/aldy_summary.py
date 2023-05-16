import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from common.sampler import TimeSeriesSampler
from common.settings import *
from common.torch.ops import to_tensor, default_device
from common.torch.snapshots_STEPS import SnapshotManager
from common.torch.snapshots import SnapshotManager as SnapshotManagerTLAE
from common.utils import read_config_file
from common.torch.losses import mape_loss, smape_loss, wape
from datasets.datsetsFactory import DatasetsFactory
from common.settings import DATASETS_PATH
from models.aldy.aldy_3 import ALDy
import ast

from models.aldy.tlae import TLAE


def get_model_aldy_exp(snapshot_manager, config_dict, ts_samples):
    idx_hidden_dim = 0
    ae_hidden_dims = ast.literal_eval(config_dict['ae_hidden_dims'])

    for i in range(len(ae_hidden_dims) - 1):
        if ae_hidden_dims[i] == ae_hidden_dims[i + 1]:
            idx_hidden_dim = i
            break
    latent_dim = ae_hidden_dims[idx_hidden_dim]
    f_num_inputs = latent_dim
    input_dim = ts_samples.shape[-1]

    f_num_shared_channels = ast.literal_eval(config_dict['f_num_shared_channels'])

    f_model_params = {'model_type': config_dict['f_model_type'][1:-1],
                      'num_inputs': f_num_inputs,
                      'num_shared_channels': f_num_shared_channels,
                      'nbr_param_layers': int(config_dict['f_nbr_param_layers']),
                      'ts2vec_output_dims': int(config_dict['ts2vec_output_dims']),
                      'kernel_size': int(config_dict['f_kernel_size']),
                      'dropout': float(config_dict['f_dropout']),
                      'leveld_init': eval(config_dict['f_leveld_init']),
                      'implicit_batching': eval(config_dict['f_implicit_batching']),
                      'alpha': float(config_dict['f_alpha'])}

    model = ALDy(input_dim=input_dim,
                 ae_hidden_dims=ae_hidden_dims,
                 f_model_params=f_model_params,
                 f_input_window=int(config_dict['f_input_window']),
                 train_window=int(config_dict['train_window']),
                 f_out_same_in_size=eval(config_dict['f_out_same_in_size']),
                 ts2vec_output_dims=int(config_dict['ts2vec_output_dims']),
                 ts2vec_hidden_dims=int(config_dict['ts2vec_hidden_dims']),
                 ts2vec_depth=int(config_dict['ts2vec_depth']),
                 ts2vec_mask_mode=config_dict['ts2vec_mask_mode'][1:-1],
                 activation=config_dict['activation'][1:-1])

    _ = snapshot_manager.restore(model=model, optimizer=None)
    return model


def get_model_tlae_exp(snapshot_manager, config_dict, ts_samples):
    idx_hidden_dim = 0
    ae_hidden_dims = ast.literal_eval(config_dict['ae_hidden_dims'])

    for i in range(len(ae_hidden_dims) - 1):
        if ae_hidden_dims[i] == ae_hidden_dims[i + 1]:
            idx_hidden_dim = i
            break
    latent_dim = ae_hidden_dims[idx_hidden_dim]
    f_num_inputs = latent_dim
    f_output_size = latent_dim

    f_num_channels = ast.literal_eval(config_dict['f_num_channels'])

    f_model_type = config_dict['f_model_type'][1:-1]
    if f_model_type == 'TCN_Modified':
        f_model_params = {'model_type': f_model_type,
                          'num_inputs': f_num_inputs,
                          'output_size': f_output_size,
                          'num_channels': f_num_channels,
                          'kernel_size': int(config_dict['f_kernel_size']),
                          'dropout': float(config_dict['f_dropout']),
                          'leveld_init': eval(config_dict['f_leveld_init'])}
    else:  # f_model_type == 'LSTM_Modified':
        f_model_params = {'model_type': f_model_type,
                          'input_size': f_num_inputs,
                          'output_size': f_output_size,
                          'hidden_size': int(config_dict['f_hidden_size']),
                          'num_layers': int(config_dict['f_num_layers']),
                          'dropout': float(config_dict['f_dropout']),
                          'batch_first': True}
    input_dim = ts_samples.shape[-1]
    model = TLAE(input_dim=input_dim,
                 ae_hidden_dims=ae_hidden_dims,
                 f_model_params=f_model_params,
                 f_input_window=int(config_dict['f_input_window']),
                 train_window=int(config_dict['train_window']),
                 f_out_same_in_size=eval(config_dict['f_out_same_in_size']),
                 activation=config_dict['activation'][1:-1])

    _ = snapshot_manager.restore(model=model, optimizer=None)
    return model


class ALDy_Summary:
    def __init__(self,
                 ts_dataset_name: str,
                 ):
        self.ts_dataset_name = ts_dataset_name

        # space holders initialization
        self.experiment_path = None
        self.snapshots_dir_path = None
        self.snapshot_manager = None
        self.losses = None
        self.epochs = None

        self.experiment_name = None
        self.embedding_instance_name = None
        self.experiment_type = None

        self.mape_loss = None
        self.smape_loss = None
        self.wape_loss = None

        self.model = None

        self.ts_samples = None
        self.ts_sampler = None

        self.config_dict = None

    def load_experiment(self, experiment_path: str):
        assert os.path.exists(experiment_path), "Path to experiment does not exist."
        self.experiment_path = experiment_path
        self.experiment_name = os.path.join(*[i for i in self.experiment_path.split(os.sep)[-1:]])

        self.snapshots_dir_path = os.path.join(experiment_path, 'snapshots')

        config_file_path = os.path.join(experiment_path, 'config.gin')
        config_dict = read_config_file(config_file_path)

        self.config_dict = config_dict

        self.experiment_type = config_dict['experiment_name'][1:-1]

        dataset_path = os.path.join(DATASETS_PATH, self.ts_dataset_name, self.ts_dataset_name + '.npy')
        self.ts_samples = np.load(dataset_path).transpose()  # ts_samples of shape (T, N)

        if self.experiment_type == 'aldy_exps':
            self.snapshot_manager = SnapshotManager(
                snapshot_dir=self.snapshots_dir_path
            )
            self.model = get_model_aldy_exp(self.snapshot_manager, config_dict, self.ts_samples)

        # elif self.experiment_name == 'tlae_exps':
        else:
            self.snapshot_manager = SnapshotManagerTLAE(
                snapshot_dir=self.snapshots_dir_path
            )
            self.model = get_model_tlae_exp(self.snapshot_manager, config_dict, self.ts_samples)

        self.losses = self.snapshot_manager.load_losses()
        self.ts_sampler = TimeSeriesSampler(timeseries=self.ts_samples,
                                            train_window=int(config_dict['train_window']),
                                            batch_size=int(config_dict['batch_size']),
                                            f_input_window=int(config_dict['f_input_window']),
                                            horizon=int(config_dict['horizon']),
                                            non_overlap_batch=int(config_dict['non_overlap_batch']),
                                            n_test_windows=int(config_dict['n_test_windows']),
                                            n_val_windows=int(config_dict['n_val_windows'])
                                            )

    def evaluate(self):
        assert self.losses is not None, "Run load_experiment() before evaluate()"
        self.model = self.model.to(default_device())
        self.model.eval()

        input_windows_test, labels_test, input_windows_val = map(to_tensor, self.ts_sampler.test_forecasting_windows())

        # Normalize input windows
        input_windows_test_normalized = (input_windows_test - to_tensor(self.ts_sampler.train_ts_means)) / to_tensor(
            self.ts_sampler.train_ts_std)

        Y_forecast_normalized = self.model.rolling_forecast(input_windows_test_normalized,
                                                            horizon=int(self.config_dict['horizon']))
        # Rescale forecasts back to original scale
        Y_forecast = Y_forecast_normalized * to_tensor(self.ts_sampler.train_ts_std) + to_tensor(
            self.ts_sampler.train_ts_means)

        self.mape_loss = mape_loss(prediction=Y_forecast, target=labels_test).item()
        self.smape_loss = smape_loss(prediction=Y_forecast, target=labels_test).item()
        self.wape_loss = wape(prediction=Y_forecast, target=labels_test).item()

    def reset(self, reset_dataset: bool = False):
        self.experiment_path = None
        self.snapshots_dir_path = None
        self.losses = None
        self.snapshot_manager = None
        self.epochs = None

        self.experiment_name = None
        self.embedding_instance_name = None
        self.experiment_type = None

        self.mape_loss = None
        self.smape_loss = None
        self.wape_loss = None

        if reset_dataset:
            self.ts_samples = None
            self.ts_sampler = None

        self.model = None

        self.config_dict = None

    def summarize(self):
        assert self.mape_loss and self.smape_loss and self.wape_loss, "Run evaluate() before summarize()"
        summary = {
            "EXPERIMENT_NAME": [self.experiment_name],
            "DATASET_NAME": self.ts_dataset_name,
            "MAPE_LOSS": self.mape_loss,
            "SMAPE_LOSS": self.smape_loss,
            "WAPE_LOSS": self.wape_loss
        }
        return pd.DataFrame(summary)

    def plot_loss_curves(self):
        assert self.losses is not None, "Run load_experiment() before evaluate()"
        title = self.experiment_name
        self.losses = self.losses.rename(columns={'testing_loss': 'MAPE'})
        self.losses[['training_loss', 'MAPE']].plot(title=title)
        plt.show()
