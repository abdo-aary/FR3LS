import random

import numpy as np
import torch as t

import torch.nn as nn
import torch.nn.functional as F

from common.torch.ops import default_device, to_tensor
from models.modules_utils import load_forecasting_model, activation_layer


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


class ALDy(nn.Module):
    def __init__(self, input_dim: int,
                 ae_hidden_dims: list,
                 f_input_window: int,
                 train_window: int,
                 f_model_params: dict = None,
                 mask_mode='binomial',
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
        self.latent_dim = ae_hidden_dims[idx_hidden_dim]
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

        self.Normal = t.distributions.Normal(0, 1)

        self.input_fc = nn.Linear(self.input_dim, self.hidden_dim)

        self.encoder = t.nn.Sequential(*(
            list(
                np.array([
                    [activation_layer(self.activation), t.nn.Linear(self.encoder_dims[i], self.encoder_dims[i + 1])]
                    for i in range(len(self.encoder_dims) - 1)]
                ).flatten()
            )
        ))

        self.mu_layer = t.nn.Linear(self.encoder_dims[-1], self.latent_dim)
        self.log_sigma_layer = t.nn.Linear(self.encoder_dims[-1], self.latent_dim)

        # decoder
        self.decoder = t.nn.Sequential(*(
                list(
                    np.array([
                        [t.nn.Linear(self.decoder_dims[i], self.decoder_dims[i + 1]), activation_layer(self.activation)]
                        for i in range(len(self.encoder_dims) - 1)]
                    ).flatten()
                )
                # + [t.nn.Linear(self.decoder_dims[-1], self.input_dim), t.nn.Sigmoid()] # TODO: take out this sigmoid ?
                + [t.nn.Linear(self.decoder_dims[-1], self.input_dim)]
        ))

        # Forecasting model
        self.f_model = load_forecasting_model(params=f_model_params)

        self.f_input_indices = np.array(
            [np.arange(i - self.f_input_window, i) for i in range(self.f_input_window, self.train_window)])

        self.f_label_indices = np.arange(self.f_input_window, self.train_window)

    def forward(self, y, mask=None):
        # y should be of shape (batch_size, train_window, N)  # alias batch_size = bs

        # encoding
        x_v1 = self.encoder(self.generate_apply_mask(y, use_mask=True, mask=mask))
        x_v2 = self.encoder(self.generate_apply_mask(y, use_mask=True, mask=mask))
        x = self.encoder(self.generate_apply_mask(y, use_mask=False, mask=None))

        # get `mu` and `sigma`
        mu = self.mu_layer(x)
        sigma = t.exp(self.log_sigma_layer(x))

        # get the latent vector through reparameterization
        x = mu + sigma * to_tensor(self.Normal.sample(mu.shape), device=next(self.parameters()).device)

        # X_f_input.shape = (b * (tw - fw), fw, latent_dim)
        x_f_input = x[:, self.f_input_indices, :].flatten(0, 1)

        # Forecasting in the latent space
        # X_f of shape (b, tw - fw, latent_dim)
        x_f = self.f_model(x_f_input).reshape(x.shape[0], self.f_input_indices.shape[0], self.latent_dim)

        # X_f_labels.shape = X_f_total.shape
        x_f_labels = x[:, self.f_label_indices, :]

        x_hat = t.cat((x[:, :self.f_input_window, :], x_f), dim=1)

        # decoding
        y_hat = self.decoder(x_hat)
        return y_hat, x_v1, x_v2, x_f_labels, x_f, mu, sigma

    def rolling_forecast(self, y: t.Tensor, horizon: int):
        """
        Performs rolling forecasting

        :param y: of shape (N, Lin, Cin)
        :param horizon: Nbr of time points to forecast
        :return:
        """

        # Latent Series Generation
        y_forecast = t.zeros((y.size(0), horizon, y.size(-1)), device=y.device)

        for i in range(horizon):
            # encoding
            x = self.encoder(self.generate_apply_mask(y, use_mask=False, mask=None))

            # get `mu` and `sigma`
            mu = self.mu_layer(x)
            sigma = t.exp(self.log_sigma_layer(x))

            # get the latent vector through reparameterization
            x = mu + sigma * to_tensor(self.Normal.sample(mu.shape), device=next(self.parameters()).device)

            # x_hat of shape (test_w, latent_dim)
            x_hat = self.f_model(x)  # Use the last pt to forecast the next

            y_hat = self.decoder(x_hat)

            y_forecast[:, i, :] = y_hat

            # Add the new forecasted elements to the last position of y on dimension 1
            y = t.cat((y[:, 1:, :], y_hat.reshape(y_hat.size(0), 1, y_hat.size(1))), dim=1)

        return y_forecast

    def generate_apply_mask(self, y: t.Tensor, use_mask: bool = True, mask=None):
        # generate & apply mask
        x = t.clone(y)
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x N

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

        return x
