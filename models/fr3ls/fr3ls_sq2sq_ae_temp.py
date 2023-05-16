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


class FR3LS(nn.Module):
    def __init__(self, input_dim: int,
                 ae_hidden_dims: list,
                 f_input_window: int,
                 train_window: int,
                 f_model_params: dict = None,
                 mask_mode='binomial',
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 type_augV_latent: str = None,
                 direct_decoding: bool = False,
                 augV_method: str = 'zeros',
                 horizon: int = 24,

                 **kwargs):
        """
        ALDY main model

        :param input_dim:
        :param ae_hidden_dims:
        :param f_input_window:
        :param train_window:
        :param activation:
        """
        super(FR3LS, self).__init__()
        assert len(ae_hidden_dims) > 2, "ae_hidden_dims should be of length > 2"

        idx_hidden_dim = len(ae_hidden_dims) // 2 - 1

        self.input_dim = input_dim
        self.latent_dim = ae_hidden_dims[idx_hidden_dim]
        self.encoder_dims = np.array(ae_hidden_dims[:idx_hidden_dim + 1])

        self.hidden_dim = self.encoder_dims[0]
        self.mask_mode = mask_mode

        self.decoder_dims = np.array(ae_hidden_dims[idx_hidden_dim + 1:])
        self.activation = activation
        self.f_input_window = f_input_window  # fw
        self.train_window = train_window  # tw
        self.type_augV_latent = type_augV_latent
        self.direct_decoding = direct_decoding
        self.augV_method = augV_method

        # TODO: Temporarily
        random.seed(3407)
        np.random.seed(3407)
        t.manual_seed(3407)

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
            [np.arange(i - self.f_input_window, i) for i in range(self.f_input_window, self.train_window - horizon + 1)])

        self.f_label_indices = np.array([np.arange(i, i + horizon) for i in range(f_input_window, train_window - horizon + 1)])

    def forward(self, y, use_f_m=True, mask=None, noise_mean=0, noise_std=1):
        # y should be of shape (batch_size, train_window, N)  # alias batch_size = bs

        # encoding
        x_v1 = self.encoder(
            self.generate_apply_mask(y, use_mask=True, mask=mask, method=self.augV_method, noise_mean=noise_mean,
                                     noise_std=noise_std))  # (bs, train_w, latent_dim)
        x_v2 = self.encoder(
            self.generate_apply_mask(y, use_mask=True, mask=mask, method=self.augV_method, noise_mean=noise_mean,
                                     noise_std=noise_std))

        if self.type_augV_latent:
            if self.type_augV_latent == 'mean':
                x = t.mean(t.stack((x_v1, x_v2), dim=-1), dim=-1)
            elif self.type_augV_latent == 'max':
                x, _ = t.max(t.stack((x_v1, x_v2), dim=-1), dim=-1)
            else:
                raise Exception('Unknown type_augV_latent')
        else:
            x = self.encoder(self.generate_apply_mask(y, use_mask=False, mask=None))

        if use_f_m:
            # x_f_input.shape = (b, (tw - fw - horizon + 1), fw, latent_dim)
            x_f_input = x[:, self.f_input_indices, :]

            # x_f_labels.shape = (b, (tw - fw - horizon + 1), horizon, latent_dim)
            x_f_labels = x[:, self.f_label_indices, :]

            # Forecasting in the latent space
            # X_f of shape (b, tw - fw, latent_dim)
            # x_f = self.f_model(source=x_f_input, target=x_f_labels).reshape(x.shape[0], self.f_input_indices.shape[0], self.latent_dim)

            # x_f.shape = (b, (tw - fw - horizon + 1), horizon, latent_dim)
            x_f = self.f_model(source=x_f_input.flatten(0, 1), target=x_f_labels.flatten(0, 1)).reshape(x_f_labels.shape)

            if self.direct_decoding:
                x_hat = x
            else:
                # x_hat is structured as x[:, :fw, :] concatenated with the first element of each forecasted window of
                # the first forecasted windows before the last one. Then we concatenate the hole last forecasted window

                # Collect the first element of each forecasted window of # the first forecasted windows before the last one.
                # x_f_first_elements is of shape (b, tw - fw - horizon, latent_dim)
                x_f_first_elements = t.cat([x_f[:, i:i+1, 0, :] for i in range(x_f.shape[1] - 1)], dim=1)

                # Concatenate the hole last forecasted window
                # x_f_cat is of shape (b, tw - fw, latent_dim)
                x_f_cat = t.cat((x_f_first_elements, x_f[:, -1]), dim=1)

                # x_f_cat is of shape (b, tw, latent_dim)
                x_hat = t.cat((x[:, :self.f_input_window, :], x_f_cat), dim=1)
        else:
            x_f_labels, x_f = 0, 0
            x_hat = x

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
        x_forecast = self.f_model(source=x, target=None, horizon=horizon)

        y_forecast = self.decoder(x_forecast)

        return y_forecast

    def generate_apply_mask(self, y: t.Tensor, use_mask: bool = True, mask=None,
                            method='zeros', noise_mean=0, noise_std=1, nan_masking: str = 'specific'):
        # generate & apply mask
        x = t.clone(y)
        if nan_masking == 'hole':
            nan_mask = ~x.isnan().any(axis=-1)
            x[~nan_mask] = 0
        elif nan_masking == 'specific':
            nan_mask = ~x.isnan()
            x[~nan_mask] = 0
        else:
            raise Exception('Unknown nan_masking method ' + nan_masking)

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

        if nan_masking == 'hole':
            mask &= nan_mask

        if method == 'zeros':
            x[~mask] = 0
        elif method == 'noise':
            x[~mask] += t.randn(size=x[~mask].shape, device=x.device) * noise_std + noise_mean
        return x
