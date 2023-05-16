import random

import numpy as np
import torch as t
import torch.nn as nn

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
                 ts2vec_output_dims: int = 320,
                 ts2vec_hidden_dims: int = 64,
                 ts2vec_depth: int = 10,
                 ts2vec_mask_mode: str = 'binomial',
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 use_augV_latent: bool = False,
                 type_augV_latent: str = None):
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
        self.use_augV_latent = use_augV_latent
        self.type_augV_latent = type_augV_latent

        # TODO: Temporarily
        random.seed(3407)
        np.random.seed(3407)
        t.manual_seed(3407)

        # self.Normal = t.distributions.Normal(0, 1)

        self.input_fc = nn.Linear(self.input_dim, self.hidden_dim)

        self.encoder = t.nn.Sequential(*(
            list(
                np.array([
                    [nn.Dropout(p=dropout), activation_layer(self.activation),
                     t.nn.Linear(self.encoder_dims[i], self.encoder_dims[i + 1])]
                    for i in range(len(self.encoder_dims) - 1)]
                ).flatten()
            )
        ))

        # decoder
        self.decoder = t.nn.Sequential(*(
                list(
                    np.array([
                        [t.nn.Linear(self.decoder_dims[i], self.decoder_dims[i + 1]), nn.Dropout(p=dropout),
                         activation_layer(self.activation)]
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
        x_v1 = self.encoder(self.generate_apply_mask(y, use_mask=True, mask=mask))  # (bs, train_w, latent_dim)
        x_v2 = self.encoder(self.generate_apply_mask(y, use_mask=True, mask=mask))
        if self.use_augV_latent:
            if self.type_augV_latent == 'mean':
                x = t.mean(t.stack((x_v1, x_v2), dim=-1), dim=-1)
            elif self.type_augV_latent == 'max':
                x, _ = t.max(t.stack((x_v1, x_v2), dim=-1), dim=-1)
            else:
                raise Exception('Unknown type_augV_latent')
        else:
            x = self.encoder(self.generate_apply_mask(y, use_mask=False, mask=None))

        # X_f_input.shape = (b * (tw - fw), fw, latent_dim)
        x_f_input = x[:, self.f_input_indices, :].flatten(0, 1)

        # Forecasting in the latent space
        # X_f of shape (b, tw - fw, latent_dim)
        x_f = self.f_model(x_f_input).reshape(x.shape[0], self.f_input_indices.shape[0], self.latent_dim)

        # X_f_labels.shape = X_f_total.shape
        x_f_labels = x[:, self.f_label_indices, :]

        if self.direct_decoding:
            x_hat = x
        else:
            x_hat = t.cat((x[:, :self.f_input_window, :], x_f), dim=1)

        # decoding
        y_hat = self.decoder(x_hat)
        return y_hat, x_v1, x_v2, x_f_labels, x_f

    def rolling_forecast(self, y: t.Tensor, horizon: int):
        """
        Performs rolling forecasting

        :param y: of shape (N, Lin, Cin)
        :param horizon: Nbr of time points to forecast
        :return:
        """

        # encoding
        x = self.encoder(self.generate_apply_mask(y, use_mask=False, mask=None))

        # Latent Series Generation
        x_forecast = t.zeros((x.size(0), horizon, x.size(-1)), device=x.device)

        for i in range(horizon):
            # x_hat of shape (test_w, latent_dim)
            x_hat = self.f_model(x)  # Use the last pt to forecast the next

            x_forecast[:, i, :] = x_hat

            # Add the new forecasted elements to the last position of y on dimension 1
            x = t.cat((x[:, 1:, :], x_hat.reshape(x_hat.size(0), 1, x_hat.size(1))), dim=1)

        y_forecast = self.decoder(x_forecast)

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
