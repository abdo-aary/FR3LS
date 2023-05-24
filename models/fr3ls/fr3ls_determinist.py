import random

import numpy as np
import torch as t
import torch.nn as nn

from models.modules_utils import load_forecasting_model, activation_layer, generate_binomial_mask, \
    generate_continuous_mask


class FR3LS_Determinist(nn.Module):
    def __init__(self, input_dim: int,
                 ae_hidden_dims: list,
                 f_input_window: int,
                 train_window: int,
                 f_model_params: dict = None,
                 mask_mode: str = 'binomial',
                 dropout: float = 0.0,
                 activation: str = 'relu',
                 model_random_seed: int = 3407):
        """
        FR3LS Determinist main model

        :param input_dim: number of series variables
        :param ae_hidden_dims: autoencoder structure
        :param f_input_window: forecasting model input sequence length L
        :param train_window: train window input sequence length w
        :param f_model_params: parameters of the forecasting model
        :param mask_mode: type of mode applied for augmented views production
        :param dropout: FR3LS dropout
        :param activation: used activation type
        :param model_random_seed: random initialization of the model
        """

        super(FR3LS_Determinist, self).__init__()
        assert len(ae_hidden_dims) >= 2, "ae_hidden_dims should be of length >= 2"

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

        random.seed(model_random_seed)
        np.random.seed(model_random_seed)
        t.manual_seed(model_random_seed)

        # Input Projection Layer
        self.input_fc = nn.Linear(self.input_dim, self.hidden_dim)

        # encoder
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
                + [t.nn.Linear(self.decoder_dims[-1], self.input_dim)]
        ))

        # Forecasting model
        self.f_model = load_forecasting_model(params=f_model_params)

        self.f_input_indices = np.array(  # Positions of the input windows slicing
            [np.arange(i - self.f_input_window, i) for i in range(self.f_input_window, self.train_window)])

        self.f_label_indices = np.arange(self.f_input_window, self.train_window)  # Positions of the target points


    def forward(self, y, use_f_m=True, mask=None, noise_mean=0, noise_std=1):
        """
        forward function

        :param y: input sequence of shape (batch_size, w, N)  # alias batch_size = bs
        :param use_f_m: use forecasting model or not
        :param mask: type of mask used
        :param noise_mean: mean of the noising augmented views production
        :param noise_std: std of the noising augmented views production
        :return:
        """

        # Encoding
        # Generate the two views
        x_v1 = self.encoder(
            self.generate_apply_mask(y, use_mask=True, mask=mask, noise_mean=noise_mean,
                                     noise_std=noise_std))  # (bs, train_w, latent_dim)
        x_v2 = self.encoder(
            self.generate_apply_mask(y, use_mask=True, mask=mask, noise_mean=noise_mean,
                                     noise_std=noise_std))

        x = t.mean(t.stack((x_v1, x_v2), dim=-1), dim=-1)  # TimeStamp Mean Module to aggregate the two aug views x_v1,2

        if use_f_m:
            # X_f_input.shape = (b * (tw - fw), fw, latent_dim)
            x_f_input = x[:, self.f_input_indices, :].flatten(0, 1)

            # Forecasting in the latent space
            # X_f of shape (b, tw - fw, latent_dim)
            x_f = self.f_model(x_f_input).reshape(x.shape[0], self.f_input_indices.shape[0], self.latent_dim)

            # X_f_labels.shape = X_f_total.shape
            x_f_labels = x[:, self.f_label_indices, :]

            # concatenate forecasted and original latent series
            x_hat = t.cat((x[:, :self.f_input_window, :], x_f), dim=1)
        else:
            x_f_labels, x_f = 0, 0
            x_hat = x

        # Decoding
        y_hat = self.decoder(x_hat)
        return y_hat, x_v1, x_v2, x_f_labels, x_f


    def rolling_forecast(self, y: t.Tensor, horizon: int):
        """
        Performs rolling forecasting

        :param y: of shape (k, L, N), with k = num_test_windows
        :param horizon: Nbr of time points to forecast (i.e., tau)

        :return: y_forecast: of shape (k, horizon, N)
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

        # Decoding
        y_forecast = self.decoder(x_forecast)

        return y_forecast


    def generate_apply_mask(self, y: t.Tensor, use_mask: bool = True, mask=None,
                            noise_mean=0, noise_std=1, nan_masking: str = 'specific'):
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

        x[~mask] += t.randn(size=x[~mask].shape, device=x.device) * noise_std + noise_mean  # Timestamp Noising Module
        return x
