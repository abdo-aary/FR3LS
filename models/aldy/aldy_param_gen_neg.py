import random

import numpy as np
import torch as t

from common.torch.ops import default_device
from models.modules_utils import load_forecasting_model, activation_layer
from models.ts2vec_models.encoder import TSEncoder


class ALDy(t.nn.Module):

    def __init__(self, input_dim: int, ae_hidden_dims: list,
                 f_model_params: dict, f_input_window: int,
                 train_window: int, f_out_same_in_size: bool,
                 ts2vec_output_dims: int = 320, ts2vec_hidden_dims: int = 64,
                 ts2vec_depth: int = 10, ts2vec_mask_mode: str = 'binomial',
                 activation: str = 'relu'):
        """
        ALDY main model

        :param input_dim:
        :param ae_hidden_dims:
        :param f_model_params:
        :param f_input_window:
        :param train_window:
        :param f_out_same_in_size
        :param ts2vec_output_dims:
        :param ts2vec_hidden_dims:
        :param ts2vec_depth:
        :param ts2vec_mask_mode:
        :param activation:
        """

        idx_hidden_dim = 0
        for i in range(len(ae_hidden_dims) - 1):
            if ae_hidden_dims[i] == ae_hidden_dims[i + 1]:
                idx_hidden_dim = i
                break

        self.input_dim = input_dim
        self.latent_dim = ae_hidden_dims[idx_hidden_dim]
        self.encoder_dims = np.array(ae_hidden_dims[:idx_hidden_dim + 1])
        self.decoder_dims = np.array(ae_hidden_dims[idx_hidden_dim + 1:])
        self.activation = activation
        self.f_input_window = f_input_window
        self.train_window = train_window
        self.f_out_same_in_size = f_out_same_in_size
        self.f_model_params = f_model_params

        # AutoEncoder architecture
        super().__init__()

        # TODO: Temporarily
        random.seed(3407)
        np.random.seed(3407)
        t.manual_seed(3407)

        self.encoder = t.nn.Sequential(*(
                [t.nn.Linear(self.input_dim, self.encoder_dims[0])] +
                list(
                    np.array([
                        [activation_layer(self.activation), t.nn.Linear(self.encoder_dims[i], self.encoder_dims[i + 1])]
                        for i in range(len(self.encoder_dims) - 1)]
                    ).flatten()
                )
        ))

        self.decoder = t.nn.Sequential(*(
                list(
                    np.array([
                        [t.nn.Linear(self.decoder_dims[i], self.decoder_dims[i + 1]), activation_layer(self.activation)]
                        for i in range(len(self.encoder_dims) - 1)]
                    ).flatten()
                )
                # + [t.nn.Linear(self.decoder_dims[-1], self.input_dim, bias=False)]
                + [t.nn.Linear(self.decoder_dims[-1], self.input_dim)]
        ))

        self.ts2vec_encoder = TSEncoder(input_dims=self.latent_dim,
                                        output_dims=ts2vec_output_dims,
                                        hidden_dims=ts2vec_hidden_dims,
                                        depth=ts2vec_depth,
                                        mask_mode=ts2vec_mask_mode)

        # Forecasting model
        self.f_model = load_forecasting_model(params=f_model_params)

        ######### Forecasting Parameters Generator model  #########
        if f_model_params['model_type'] == 'LSTM_ParamGEN':
            # LSTM Forecasting Parameters Generator model
            self.hidden_size = f_model_params['hidden_size']
            self.input_size = f_model_params['input_size']
            gen_params_len = 2 * 4 * self.hidden_size * self.hidden_size
        else:
            # LSTM Forecasting Parameters Generator model
            self.Cout = f_model_params['num_shared_channels'][-1]
            self.kernel_size = f_model_params['kernel_size']
            gen_params_len = self.Cout * self.Cout * self.kernel_size

        self.f_param_generator = t.nn.Linear(in_features=ts2vec_output_dims, out_features=gen_params_len)
        ###########################################################

        self.f_input_indices = np.array(
            [np.arange(i - f_input_window, i) for i in range(self.f_input_window, self.train_window)])

        self.f_label_indices = np.array(
            [np.arange(i - f_input_window + 1, i + 1) for i in range(self.f_input_window, self.train_window)]) \
            if self.f_out_same_in_size else np.arange(self.f_input_window, self.train_window)

    def forward(self, Y: t.Tensor):
        # Y of shape (N, Lin, Cin) Cin = nbr of channels, Lin = length of the signal sequence
        # Latent Series Generation
        X = self.encoder(Y)

        # X_f_input.shape = ((train_window - f_input_window), f_input_window, latent_dim)
        X_f_input = X[:1, self.f_input_indices, :].flatten(0, 1)

        # Context Capturing
        R = self.ts2vec_encoder(X, use_mask=False)

        # R_means.shape = (ts2vec_output_dims,)  Summarizes the hole train_window's context variable
        R_mean = R[0].mean(dim=0)

        weights = self.f_param_generator(R_mean)
        if self.f_model_params['model_type'] == 'LSTM_ParamGEN':
            weights = weights.view(2, 4 * self.hidden_size, self.hidden_size)
        else:
            weights = weights.view(self.Cout, self.Cout, self.kernel_size)

        # X_f_total of shape ((train_window-f_input_window), f_input_window, latent_dim) if self.f_out_same_in_size
        # else X_f_total of shape ((train_window-f_input_window), latent_dim), we forecast only the next value
        X_f_total = self.f_model(X_f_input, weights, self.f_out_same_in_size)

        # X_f_labels.shape = X_f_total.shape
        X_f_labels = X[:1, self.f_label_indices, :].flatten(0, 1)

        X_f = X_f_total.view(1, X_f_total.shape[0], X_f_total.shape[1])

        # X_hat = X[t+1:t+L] + X_f[t+L:t+w]
        X_hat = t.cat((X[:1, :self.f_input_window, :], X_f), dim=1)

        # Decoding Latent series into Original ones
        Y_hat = self.decoder(X_hat)

        return Y_hat, X_f_labels, X_f_total

    def rolling_forecast(self, Y: t.Tensor, horizon: int):
        """
        Performs rolling forecasting

        :param Y: of shape (N, Lin, Cin)
        :param horizon: Nbr of time points to forecast
        :return:
        """
        # Latent Series Generation
        X = self.encoder(Y)

        X_forecast = t.zeros((X.size(0), horizon, X.size(2)), device=Y.device)

        for j in range(X.shape[0]):  # Loop over the batch nbr
            for i in range(horizon):
                # Context Capturing
                R = self.ts2vec_encoder(X[j:j + 1], use_mask=False)  # R of shape (Lin, ts2vec_output_dims)
                R_mean = R[0].mean(dim=0)  # R_means of shape (ts2vec_output_dims,)
                if self.f_model_params['model_type'] == 'LSTM_ParamGEN':
                    weights = self.f_param_generator(R_mean).view(2, 4 * self.hidden_size, self.hidden_size)
                else:
                    weights = self.f_param_generator(R_mean).view(self.Cout, self.Cout, self.kernel_size)

                # Forecast
                # X_f of shape (1, Cin_X) if f_out_same_in_size = False
                X_f = self.f_model(X[j:j + 1], weights, self.f_out_same_in_size)

                # Store the forecasted values
                X_forecast[j:j + 1, i, :] = X_f

                # Add the last forecasted values to the input window for further forecasting
                X[j:j + 1] = t.cat((X[j:j + 1, 1:, :], X_f.view(X_f.size(0), 1, X_f.size(1))), dim=1)

        # Decoding Latent series into Original ones
        Y_forecast = self.decoder(X_forecast)

        return Y_forecast
