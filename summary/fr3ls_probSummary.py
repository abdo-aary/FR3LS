import numpy as np
import pandas as pd
import torch as t
from matplotlib import pyplot as plt

from common.sampler import Sampler
from common.settings import *
from common.torch import CRPS
from common.torch.losses import loss_fn
from common.torch.ops import torch_dtype_dict, to_tensor, default_device
from common.torch.snapshots import SnapshotManager
from common.utils import read_config_file
from datasets.load_data_gluonts import load_data_gluonts
from models.fr3ls.fr3ls_probabilist import FR3LS_Probabilist

import warnings


class FR3LS_ProbSummary:
    def __init__(self, ):
        # space holders initialization

        self.dataset_name = None
        self.experiment_path = None
        self.snapshots_dir_path = None
        self.snapshot_manager = None
        self.losses = None

        self.experiment_name = None
        self.embedding_instance_name = None

        self.horizon = None
        self.n_test_window = None

        self.mse_loss = None
        self.crps_sum_loss = None
        self.crps_loss = None

        self.epoch = None
        self.experiment_training_loss = None
        self.experiment_testing_loss = None

        self.config_dict = None

    def load_experiment(self, experiment_path: str):
        assert os.path.exists(experiment_path), "Path to experiment does not exist."
        self.experiment_path = experiment_path
        self.experiment_name = os.path.join(*[i for i in self.experiment_path.split(os.sep)[-1:]])
        config_file_path = os.path.join(experiment_path, 'config.gin')
        self.config_dict = read_config_file(config_file_path)
        self.dataset_name = eval(self.config_dict['ts_dataset_name'])
        self.horizon = eval(self.config_dict['horizon'])
        self.n_test_window = eval(self.config_dict['n_test_windows'])

        self.snapshots_dir_path = os.path.join(experiment_path, 'snapshots')
        self.snapshot_manager = SnapshotManager(
            snapshot_dir=self.snapshots_dir_path,
            losses=['training', 'testing'],
            verbose=False,
        )
        self.losses = self.snapshot_manager.load_losses()

    def evaluate(self, ):
        # Extract the experiment with the most inferior loss combination
        assert self.losses is not None, "Run load_experiment() before evaluate()"

        losses = self.losses.round(decimals=3).sort_values(["testing_loss"]).head(1)

        self.epoch = losses.index[0]
        self.experiment_training_loss = losses['training_loss'].iloc[0]
        self.crps_loss, self.crps_sum_loss, self.mse_loss = self.prob_evaluate()

    def summarize(self):
        assert self.experiment_training_loss is not None, "Run evaluate() before summarize()"
        summary = {
            "EXPERIMENT_NAME": [self.experiment_name],
            "DATASET_NAME": self.dataset_name,
            "EPOCH": self.epoch,
            "NUM_TEST_WINDOW": self.n_test_window,
            "HORIZON": self.horizon,
            "TRAINING_LOSS": self.experiment_training_loss,
            "MSE_LOSS": self.mse_loss,
            "CRPS-SUM": self.crps_sum_loss,
        }

        return pd.DataFrame(summary)

    def plot_loss_curves(self):
        assert self.losses is not None, "Run load_experiment() before evaluate()"
        title = self.experiment_name

        fig, axs = plt.subplots(1, 2, figsize=(4, 2))  # create a figure with a 1x2 grid of subplots

        axs[0, 0].plot(self.losses['training_loss'])
        axs[0, 0].set_title('training_loss')
        axs[0, 1].plot(self.losses['testing_loss'])
        axs[0, 1].set_title('testing_loss')

        # add a common title for the whole figure
        fig.suptitle(title)

        # adjust the layout to avoid overlapping titles and labels
        fig.tight_layout()

        plt.show()

    def reset(self):
        self.dataset_name = None

        self.experiment_path = None
        self.snapshots_dir_path = None
        self.losses = None
        self.snapshot_manager = None

        self.experiment_name = None
        self.embedding_instance_name = None

        self.horizon = None
        self.n_test_window = None

        self.mse_loss = None
        self.crps_sum_loss = None
        self.crps_loss = None

        self.epoch = None
        self.experiment_training_loss = None
        self.experiment_testing_loss = None

        self.config_dict = None

    def prob_evaluate(self, ):
        warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan")

        used_dtype = eval(self.config_dict["used_dtype"])

        t.manual_seed(eval(self.config_dict["random_state"]))
        t.set_default_dtype(torch_dtype_dict[used_dtype])

        f_input_window = eval(self.config_dict["f_input_window"])
        train_window = 2 * f_input_window

        encoder_dims = eval(self.config_dict["encoder_dims"])
        ae_hidden_dims = list(encoder_dims)
        encoder_dims.reverse()
        ae_hidden_dims += encoder_dims

        n_test_windows = eval(self.config_dict["n_test_windows"])
        n_val_windows = eval(self.config_dict["n_val_windows"])
        non_overlap_batch = eval(self.config_dict["non_overlap_batch"])

        num_samples = eval(self.config_dict["num_samples"])

        if self.dataset_name != 'traffic':
            # Only traffic data is not loaded using "load_data_gluonts" function

            train_data, test_data_in, test_data_target = load_data_gluonts(name=self.dataset_name,
                                                                           prediction_length=self.horizon,
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

            input_dim = train_data.shape[-1]

            sampler = Sampler(train_data=train_data,
                              test_data_in=test_data_in,
                              test_data_target=test_data_target,
                              train_window=train_window,
                              f_input_window=f_input_window,
                              horizon=self.horizon,
                              non_overlap_batch=non_overlap_batch,
                              n_test_windows=n_test_windows,
                              n_val_windows=n_val_windows, )

        else:
            dataset_path = os.path.join(DATASETS_PATH, self.dataset_name, self.dataset_name + '.npy')
            ts_samples = np.load(dataset_path).transpose()  # ts_samples of shape (T, N)
            ts_samples = ts_samples.astype(np.dtype(used_dtype))

            # Work only with series not having zero as std
            train_end_time_point = ts_samples.shape[0] - (
                    n_val_windows + n_test_windows) * self.horizon - train_window
            train_ts_std = np.nanstd(ts_samples[:train_end_time_point + train_window], axis=0)

            used_mask = np.where(train_ts_std != 0)[0]  # We only use variables that don't have 0 as std
            ts_samples = ts_samples[:, used_mask]
            input_dim = ts_samples.shape[-1]

            sampler = Sampler(timeseries=ts_samples,
                              train_window=train_window,
                              f_input_window=f_input_window,
                              horizon=self.horizon,
                              non_overlap_batch=non_overlap_batch,
                              n_test_windows=n_test_windows,
                              n_val_windows=n_val_windows, )

        f_model_type = eval(self.config_dict["f_model_type"])
        idx_hidden_dim = len(ae_hidden_dims) // 2 - 1
        latent_dim = ae_hidden_dims[idx_hidden_dim]

        if f_model_type == 'LSTM_Modified':
            f_model_params = {'model_type': f_model_type,
                              'input_size': latent_dim,
                              'output_size': latent_dim,
                              'hidden_size': eval(self.config_dict["f_hidden_size"]),
                              'num_layers': eval(self.config_dict["f_num_layers"]),
                              'dropout': eval(self.config_dict["f_dropout"]),
                              'batch_first': True,
                              }
        else:
            raise Exception(f"Unknown f_model {f_model_type}")

        model = FR3LS_Probabilist(input_dim=input_dim,
                                  ae_hidden_dims=ae_hidden_dims,
                                  f_model_params=f_model_params,
                                  mask_mode='binomial',
                                  f_input_window=f_input_window,
                                  train_window=train_window,
                                  dropout=0,
                                  activation=eval(self.config_dict["activation"]),
                                  model_random_seed=eval(self.config_dict["model_random_seed"]),
                                  )

        test_forecasting_loss_fn = loss_fn(eval(self.config_dict["test_loss_name"]))

        _ = self.snapshot_manager.restore(model, None)
        device = default_device(device_str_id=0)

        model = model.to(device)
        with t.no_grad():
            model.eval()
            # End of the programme, compute CRPS
            input_windows_test, labels_test = map(to_tensor,
                                                  sampler.get_test_forecasting_windows(),
                                                  (device,) * 2)

            # Normalize input windows
            input_windows_test_normalized = (input_windows_test - to_tensor(sampler.train_ts_means,
                                                                            device=device)) / to_tensor(
                sampler.train_ts_std, device=device)

            # Forecast future values
            y_forecast_samples_norm, y_forecast_mu_norm = model.rolling_forecast(y=input_windows_test_normalized,
                                                                                 horizon=self.horizon,
                                                                                 num_samples=num_samples,
                                                                                 sigma=1)

            y_forecast_samples = y_forecast_samples_norm * to_tensor(sampler.train_ts_std,
                                                                     device=device) + to_tensor(
                sampler.train_ts_means, device=device)
            y_forecast_mu = y_forecast_mu_norm * to_tensor(sampler.train_ts_std,
                                                           device=device) + to_tensor(
                sampler.train_ts_means, device=device)

            test_loss = float(test_forecasting_loss_fn(prediction=y_forecast_mu, target=labels_test))
            crps, crps_sum = CRPS.calculate_crps(y_forecast_samples.detach().cpu().numpy(),
                                                 labels_test.detach().cpu().numpy())

            return crps, crps_sum, test_loss
