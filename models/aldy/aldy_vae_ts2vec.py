import random

import numpy as np
import torch as t

import torch.nn as nn
import torch.nn.functional as F

from common.torch.ops import default_device, to_tensor
from models.modules_utils import load_forecasting_model, activation_layer
from models.ts2vec_models.encoder import TSEncoder


class ALDy(nn.Module):
    def __init__(self, input_dim: int,
                 ae_hidden_dims: list,
                 f_input_window: int,
                 train_window: int,
                 f_model_params: dict = None,
                 ts2vec_output_dims: int = 320,
                 ts2vec_hidden_dims: int = 64,
                 ts2vec_depth: int = 10,
                 ts2vec_mask_mode: str = 'binomial',
                 activation: str = 'relu'):
        """
        ALDY main model

        :param input_dim:
        :param ae_hidden_dims:
        :param f_input_window:
        :param train_window:
        :param ts2vec_output_dims:
        :param ts2vec_hidden_dims:
        :param ts2vec_depth:
        :param ts2vec_mask_mode:
        :param activation:
        """
        super(ALDy, self).__init__()
        assert len(ae_hidden_dims) > 2, "ae_hidden_dims should be of length > 2"

        idx_hidden_dim = np.where(np.array([i if ae_hidden_dims[i] == ae_hidden_dims[i + 1] else 0
                                            for i in range(len(ae_hidden_dims) - 1)]) != 0)[0][0]

        self.input_dim = input_dim
        self.latent_dim = ae_hidden_dims[idx_hidden_dim]
        self.encoder_dims = np.array(ae_hidden_dims[:idx_hidden_dim + 1])
        self.input_e_dims = [self.input_dim] + list(self.encoder_dims)

        self.decoder_dims = np.array(ae_hidden_dims[idx_hidden_dim + 1:])
        self.activation = activation
        self.f_input_window = f_input_window  # fw
        self.train_window = train_window  # tw

        # TODO: Temporarily
        random.seed(3407)
        np.random.seed(3407)
        t.manual_seed(3407)

        self.Normal = t.distributions.Normal(0, 1)

        # encoder
        # self.encoder = t.nn.Sequential(*(
        #         [t.nn.Linear(self.input_dim, self.encoder_dims[0])] +
        #         (
        #             list(
        #                 np.array([
        #                     [activation_layer(self.activation),
        #                      t.nn.Linear(self.encoder_dims[i], self.encoder_dims[i + 1])]
        #                     for i in range(len(self.encoder_dims) - 2)]
        #                 ).flatten()
        #             ) if len(self.encoder_dims) - 2 > 0 else []
        #         )
        #         + [activation_layer(self.activation), t.nn.Linear(self.encoder_dims[-2], self.encoder_dims[-1])]
        # ))

        self.encoder = t.nn.Sequential(*(
            list(
                np.array(
                    [[t.nn.Linear(self.input_e_dims[i], self.input_e_dims[i + 1]), activation_layer(self.activation)]
                     for i in range(len(self.input_e_dims) - 1)]).flatten()
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

        # ts2vec_encoder
        self.ts2vec_encoder = TSEncoder(input_dims=self.latent_dim,
                                        output_dims=ts2vec_output_dims,
                                        hidden_dims=ts2vec_hidden_dims,
                                        depth=ts2vec_depth,
                                        mask_mode=ts2vec_mask_mode)

        self.f_model = t.nn.Linear(ts2vec_output_dims, self.latent_dim)  # Linear regression

        self.f_input_indices = np.array(
            [np.arange(i - self.f_input_window, i) for i in range(self.f_input_window, self.train_window)])

        self.f_label_indices = np.arange(self.f_input_window, self.train_window)

    def forward(self, y):
        # y should be of shape (batch_size, train_window, N)  # alias batch_size = bs

        # encoding
        x = self.encoder(y)

        # get `mu` and `sigma`
        mu = self.mu_layer(x)
        sigma = t.exp(self.log_sigma_layer(x))

        # get the latent vector through reparameterization
        z = mu + sigma * to_tensor(self.Normal.sample(mu.shape), device=next(self.parameters()).device)

        # Ts2Vec Encoding
        r_view1, r_view2 = self.ts2vec_encoder(z, use_mask=True), self.ts2vec_encoder(z, use_mask=True)
        r = self.ts2vec_encoder(z, use_mask=False)

        # Forecasting in the latent space
        # r_f_input = r[:, self.f_input_indices, :].mean(dim=2)  # r_f_input.shape = (bs, (tw - f_w), ts2vec_out_dim)

        # r_f_input.shape = (bs, (tw - f_w), ts2vec_out_dim)
        r_f_input = r[:, self.f_input_indices, :][:, :, -1, :]  # Use the last pt to forecast the next
        z_f_labels = z[:, self.f_label_indices, :]  # z_f_labels.shape = (bs, f_w, latent_dim)
        z_f = self.f_model(r_f_input)

        z_hat = t.cat((z[:, :self.f_input_window, :], z_f), dim=1)

        # decoding
        y_hat = self.decoder(z_hat)
        return y_hat, r_view1, r_view2, z_f_labels, z_f, mu, sigma

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
            x = self.encoder(y)

            # get `mu` and `sigma`
            mu = self.mu_layer(x)
            sigma = t.exp(self.log_sigma_layer(x))

            # get the latent vector through reparameterization
            z = mu + sigma * to_tensor(self.Normal.sample(mu.shape), device=next(self.parameters()).device)

            r = self.ts2vec_encoder(z, use_mask=False)  # r of shape (test_w, f_w, ts2vec_output_dims)

            # z_hat = self.f_model(r.mean(dim=1))  # z_hat of shape (test_w, latent_dim)

            # z_hat of shape (test_w, latent_dim)
            z_hat = self.f_model(r[:, -1, :])   # Use the last pt to forecast the next

            y_hat = self.decoder(z_hat)

            y_forecast[:, i, :] = y_hat

            # Add the new forecasted elements to the last position of y on dimension 1
            y = t.cat((y[:, 1:, :], y_hat.reshape(y_hat.size(0), 1, y_hat.size(1))), dim=1)

        return y_forecast
