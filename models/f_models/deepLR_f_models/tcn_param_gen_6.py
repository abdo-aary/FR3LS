import torch as t
import torch.nn as nn

from models.f_models.deepLR_f_models.tcn import TemporalBlock

IMPLICIT_BATCHING = True


class TCN_Parameterized(nn.Module):

    def __init__(self, num_inputs: int,
                 num_shared_channels: list,
                 nbr_param_layers: int,
                 ts2vec_output_dims: int,
                 implicit_batching: bool,
                 kernel_size: int = 2,
                 dropout: float = 0.2,
                 alpha: float = 1,
                 leveld_init=True):
        super(TCN_Parameterized, self).__init__()
        self.nbr_param_layers = nbr_param_layers
        self.num_shared_channels = num_shared_channels
        self.kernel_size = kernel_size
        self.Cout = num_shared_channels[-1]
        self.dropout = dropout
        self.implicit_batching = implicit_batching
        self.alpha = alpha

        num_channels = num_shared_channels
        output_size = num_inputs

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout,
                                     leveld_init=leveld_init)]

        self.tcn = nn.Sequential(*layers)
        self.linear_out = nn.Linear(num_channels[-1], output_size)

        # gen_params_len = self.Cout * self.Cout * self.kernel_size
        # self.f_param_generator = t.nn.Linear(in_features=ts2vec_output_dims, out_features=gen_params_len)

    # def forward(self, x: t.Tensor, r: t.Tensor, f_out_same_in_size: bool):
    def forward(self, x: t.Tensor, weights: t.Tensor, f_out_same_in_size: bool):
        # x should be of shape (1, Lin, Cin_x)
        # r should be of shape (ts2vec_output_dims,)
        # weights.shape = (Cout, Cout, kernel_size)

        # Modify last temporalBlock second layer's weights
        # weights = self.f_param_generator(r).view(self.Cout, self.Cout, self.kernel_size)

        mid_layer_index = len(self.tcn) // 2
        if type(weights) == t.Tensor:
            if weights.is_cuda and not self.tcn[mid_layer_index].conv2.weight.is_cuda:
                device = weights.device
                self.tcn[-1].conv2.weight = self.tcn[mid_layer_index].conv2.weight.to(device)

        self.tcn[-1].conv2.weight += weights * self.alpha

        # Forecast with modified weights
        forecast = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # forecast of shape (1, Lin, Cin)
        forecast = self.linear_out(forecast)

        if f_out_same_in_size:
            return forecast
        else:
            return forecast[:, -1, :]
