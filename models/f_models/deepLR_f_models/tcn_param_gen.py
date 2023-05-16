import torch as t
import torch.nn as nn
import torch.nn.functional as F

from common.torch.ops import to_tensor
from models.f_models.deepLR_f_models.tcn import TemporalBlock


class TCN_Parameterized(nn.Module):
    def __init__(self, num_inputs: int,
                 num_shared_channels: list,
                 nbr_param_layers: int,
                 ts2vec_output_dims: int,
                 kernel_size: int = 2,
                 dropout: float = 0.2,
                 leveld_init=True):
        super(TCN_Parameterized, self).__init__()
        self.nbr_param_layers = nbr_param_layers
        self.num_shared_channels = num_shared_channels
        self.kernel_size = kernel_size
        self.Cout = num_shared_channels[-1]
        self.dropout = dropout

        layers = []
        self.num_shared_layers = len(num_shared_channels)
        for i in range(self.num_shared_layers):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 70 else num_shared_channels[i - 1]
            out_channels = num_shared_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout,
                                     leveld_init=leveld_init)]

        self.shared_network = nn.Sequential(*layers)

        # TODO: if basic weights don't work, try using temporal blocks !!!
        """
        self.last_layers = []
        for i in range(self.num_shared_layers, self.num_shared_layers + self.nbr_param_layers):
            dilation_size = 2 ** i
            in_channels = self.Cout
            out_channels = self.Cout
            self.last_layers += [
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                              padding=(kernel_size - 1) * dilation_size, dropout=dropout, leveld_init=leveld_init)]

        self.last_conv_layers = nn.ModuleList(self.last_layers)
        """

        # This Leveld initialization
        self.last_conv_layers = to_tensor(t.randn((self.nbr_param_layers * self.Cout * self.Cout * self.kernel_size),
                                                  requires_grad=True)) * 1e-3
        self.last_conv_layers += 1.0 / self.kernel_size

        # TODO: Probably will need biases too !!!

        gen_params_len = self.nbr_param_layers * self.Cout * self.Cout * self.kernel_size

        # self.f_param_generator = t.nn.Sequential(
        #     t.nn.Linear(in_features=ts2vec_output_dims, out_features=(ts2vec_output_dims + gen_params_len) // 2)
        #     , t.nn.ReLU(),
        #     t.nn.Linear(in_features=(ts2vec_output_dims + gen_params_len) // 2, out_features=gen_params_len))

        self.f_param_generator = t.nn.Linear(in_features=ts2vec_output_dims, out_features=gen_params_len)

        self.linear_out = nn.Linear(self.Cout, num_inputs)

    def forward(self, x: t.Tensor, r: t.Tensor, f_out_same_in_size: bool):
        # x should be of shape (N, Lin, Cin_x)
        # r should be of shape (N, Cin_r)

        # num_inputs = nbr of channels, Lin = length of signal sequence

        N = x.shape[0]  # nbr of batch examples
        Lin = x.shape[1]  # Length of the signal
        Cin_x = x.shape[2]  # Nbr of the signal

        weights = self.f_param_generator(r)  # weights.shape = (N, nbr_param_layers * Cout * Cout * kernel_size)

        weights += self.last_conv_layers  # conv_weights = generated_weights + last_conv_layers

        weights = weights.view(self.nbr_param_layers, N * self.Cout, self.Cout, self.kernel_size)

        src = self.shared_network(x.transpose(1, 2))
        # src would have the shape (N, Cin, Lin)
        src = src.view(1, -1, src.shape[2])  # move batch dim into channels: src would be of shape (1, N*Cout, Lin)

        for i in range(self.num_shared_layers, self.num_shared_layers + self.nbr_param_layers):
            dilation_size = 2 ** i
            padding = (self.kernel_size - 1) * dilation_size
            weights_i = weights[i - self.num_shared_layers]  # weights_i of shape (N*Cout, Cout, kernel_size)

            res = src

            # We use generated params for last conv layers of the TCN model
            src = F.conv1d(src, weights_i, stride=1, padding=padding, dilation=dilation_size,
                           groups=N)[:, :, :-padding].contiguous()  # :-padding to neglect the padding at the right
            src = F.relu(
                F.dropout(
                    F.relu(src), self.dropout
                ) + res
            )
        # Up to here, src.shape = (N, Lin, Cout)

        src = src.view(N, Lin, self.Cout)  # src.shape = (N, Lin, Cout)

        src = self.linear_out(src)  # src.shape = (N, Lin, Cin_x)

        if not f_out_same_in_size:
            src = src[:, -1, :]  # (N, 1, Cin_x) : We forecast only the next value

        return src
