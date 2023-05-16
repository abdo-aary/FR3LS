import random
from typing import Optional, List

import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

from models.modules_utils import load_forecasting_model


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = t.full((B, T), True, dtype=t.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            k = np.random.randint(T - l + 1)
            res[i, k:k + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return t.from_numpy(np.random.binomial(1, p, size=(B, T))).to(t.bool)


class Interpolate(nn.Module):
    def __init__(self, mode: str, size: Optional[int] = None, scale_factor: Optional[List[float]] = None):
        super(Interpolate, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode)


class ALDy(nn.Module):
    def __init__(self, input_dim: int,
                 ae_hidden_dims: list,
                 f_input_window: int,
                 train_window: int,
                 f_model_params: dict = None,
                 mask_mode='binomial',
                 ts2vec_output_dims: int = 320,
                 ts2vec_hidden_dims: int = 64,
                 ts2vec_depth: int = 10,
                 ts2vec_mask_mode: str = 'binomial',
                 dropout: float = 0.0,
                 activation: str = 'relu'):
        """
        ALDY main model

        :param input_dim:
        :param ae_hidden_dims:
        :param f_input_window:
        :param train_window:
        :param activation:
        """
        super(ALDy, self).__init__()
        assert len(ae_hidden_dims) > 2, "ae_hidden_dims should be of length > 2"

        idx_hidden_dim = np.where(np.array([i if ae_hidden_dims[i] == ae_hidden_dims[i + 1] else 0
                                            for i in range(len(ae_hidden_dims) - 1)]) != 0)[0][0]

        self.input_dim = input_dim
        # self.latent_dim = ae_hidden_dims[idx_hidden_dim]
        self.latent_dim = 128
        self.encoder_dims = np.array(ae_hidden_dims[:idx_hidden_dim + 1])

        self.hidden_dim = self.encoder_dims[0]
        self.mask_mode = mask_mode

        self.decoder_dims = np.array(ae_hidden_dims[idx_hidden_dim + 1:])
        self.activation = activation
        self.f_input_window = f_input_window  # fw
        self.train_window = train_window  # tw

        # TODO: Temporarily
        random.seed(3407)
        np.random.seed(3407)
        t.manual_seed(3407)

        # self.Normal = t.distributions.Normal(0, 1)
        self.out_channels = 8
        N = self.input_dim
        dConv = N // (4 ** 5) * self.out_channels

        self.conv_encoder1st = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            Interpolate(size=N // 3, mode='nearest'),
        )
        self.conv_encoder = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            Interpolate(size=N // (4 ** 2), mode='nearest'),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            Interpolate(size=N // (4 ** 3), mode='nearest'),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            Interpolate(size=N // (4 ** 4), mode='nearest'),
            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            Interpolate(size=N // (4 ** 5), mode='nearest'),
            nn.Flatten()
        )
        self.fc_encoder = nn.Sequential(
            nn.Linear(dConv, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

        # decoder
        self.fc_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, dConv),
        )

        self.conv_decoder = nn.Sequential(
            Interpolate(size=N // (4 ** 4), mode='nearest'),
            nn.ConvTranspose1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            Interpolate(size=N // (4 ** 3), mode='nearest'),
            nn.ConvTranspose1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            Interpolate(size=N // (4 ** 2), mode='nearest'),
            nn.ConvTranspose1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            Interpolate(size=N // 3, mode='nearest'),
            nn.ConvTranspose1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            Interpolate(size=N, mode='nearest'),
            nn.ConvTranspose1d(in_channels=128, out_channels=1, kernel_size=3, padding=1),
        )

        # Forecasting model
        self.f_model = load_forecasting_model(params=f_model_params)

        self.f_input_indices = np.array(
            [np.arange(i - self.f_input_window, i) for i in range(self.f_input_window, self.train_window)])

        self.f_label_indices = np.arange(self.f_input_window, self.train_window)

    def forward(self, y, mask=None):
        # y should be of shape (batch_size, train_window, N)  # alias batch_size = bs

        # encoding
        x_v1 = self.fc_encoder(self.conv_encoder(self.generate_apply_mask(y, use_mask=True, mask=mask)))
        x_v1 = x_v1.view(y.shape[0], y.shape[1], x_v1.shape[-1])

        x_v2 = self.fc_encoder(self.conv_encoder(self.generate_apply_mask(y, use_mask=True, mask=mask))).view(
            y.shape[0], y.shape[1], x_v1.shape[-1])
        x = self.fc_encoder(self.conv_encoder(self.generate_apply_mask(y, use_mask=True, mask=mask))).view(
            y.shape[0], y.shape[1], x_v1.shape[-1])

        # X_f_input.shape = (b * (tw - fw), fw, latent_dim)
        x_f_input = x[:, self.f_input_indices, :].flatten(0, 1)

        # Forecasting in the latent space
        # X_f of shape (b, tw - fw, latent_dim)
        x_f = self.f_model(x_f_input).view(x.shape[0], self.f_input_indices.shape[0], self.latent_dim)

        # X_f_labels.shape = X_f_total.shape
        x_f_labels = x[:, self.f_label_indices, :]

        x_hat = t.cat((x[:, :self.f_input_window, :], x_f), dim=1)

        # decoding
        y_hat = self.fc_decoder(x_hat)
        y_hat = y_hat.view(np.array(y_hat.shape[:2]).prod(), self.out_channels,
                                            y_hat.shape[-1] // self.out_channels)

        y_hat = self.conv_decoder(y_hat).view(y.shape)
        return y_hat, x_v1, x_v2, x_f_labels, x_f

    def rolling_forecast(self, y: t.Tensor, horizon: int):
        """
        Performs rolling forecasting

        :param y: of shape (N, Lin, Cin)
        :param horizon: Nbr of time points to forecast
        :return:
        """

        # encoding
        x = self.fc_encoder(self.conv_encoder(self.generate_apply_mask(y, use_mask=False, mask=None)))
        x = x.view(y.shape[0], y.shape[1], x.shape[-1])

        # Latent Series Generation
        x_forecast = t.zeros((x.size(0), horizon, x.size(-1)), device=x.device)

        for i in range(horizon):
            # x_hat of shape (test_w, latent_dim)
            x_hat = self.f_model(x)  # Use the last pt to forecast the next

            x_forecast[:, i, :] = x_hat

            # Add the new forecasted elements to the last position of y on dimension 1
            x = t.cat((x[:, 1:, :], x_hat.reshape(x_hat.size(0), 1, x_hat.size(1))), dim=1)

        out_shape = x_forecast.shape
        y_forecast = self.fc_decoder(x_forecast)
        y_forecast = y_forecast.view(np.array(y_forecast.shape[:2]).prod(), self.out_channels,
                           y_forecast.shape[-1] // self.out_channels)

        y_forecast = self.conv_decoder(y_forecast).view(out_shape[0], out_shape[1], y.shape[-1])

        return y_forecast

    def generate_apply_mask(self, y: t.Tensor, use_mask: bool = True, mask=None):
        # generate & apply mask
        x = t.clone(y)
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0

        x = x.view(np.array(y.shape[:2]).prod(), 1, self.input_dim)

        x = self.conv_encoder1st(x)
        num_channels, dim = x.shape[1], x.shape[2]

        x = x.view(y.shape[0], y.shape[1], num_channels * dim)  # B x T x N

        if mask is None:
            if self.training and use_mask:
                mask = self.mask_mode
            else:
                mask = 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=t.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=t.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=t.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x[~mask] = 0

        return x.view(np.array(y.shape[:2]).prod(), num_channels, dim)
