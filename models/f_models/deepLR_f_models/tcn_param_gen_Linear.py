import torch as t
import torch.nn as nn
import torch.nn.functional as F

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

    # def forward(self, x: t.Tensor, r: t.Tensor, f_out_same_in_size: bool):
    def forward(self, x: t.Tensor, weight: t.Tensor, bias: t.Tensor, f_out_same_in_size: bool):
        # x should be of shape (1, Lin, Cin_x)
        # r should be of shape (ts2vec_output_dims,)
        # weights.shape = (num_inputs, num_channels[-1])

        # weights = self.f_param_generator(r).view(output_size, num_channels[-1])

        # if type(weights) == t.Tensor:
        #     if weights.is_cuda and not self.linear_out.weight.is_cuda:
        #         device = weights.device
        #         self.linear_out.weight = self.linear_out.weight.to(device)

        # self.linear_out.weight = t.nn.Parameter(self.linear_out.weight.data + weights)
        # self.linear_out.weight += weights * self.alpha

        # TODO: Try latter self.linear_out.weight = self.paramGen(linear_out.weight.flatten()).reshape(output_size,
        #  num_channels[-1])
        #  With self.paramGen module takes as input linear_out.weight and apply some non linear function on them

        weights_linear_out = t.clone(self.linear_out.weight.detach())
        bias_linear_out = t.clone(self.linear_out.bias.detach())

        gen_weights = weight * weights_linear_out
        gen_bias = bias * bias_linear_out

        # gen_weights = t.ones_like(weights_linear_out) * weights_linear_out
        # gen_bias = t.ones_like(bias_linear_out) * bias_linear_out

        if gen_weights.isnan().any().item():
            print("gen_weights.isnan().any().item() =", gen_weights.isnan().any().item())

        # Forecast with modified weights
        forecast = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # forecast of shape (1, Lin, Cin)

        if forecast.isnan().any().item():
            print("forecast_before_linear.isnan().any().item() =", forecast.isnan().any().item())

        if f_out_same_in_size:
            # forecast = self.linear_out(forecast)
            forecast = F.linear(input=forecast, weight=gen_weights, bias=gen_bias)
        else:
            # forecast = self.linear_out(forecast[:, -1, :])
            forecast = F.linear(input=forecast[:, -1, :], weight=gen_weights, bias=gen_bias)
            if forecast.isnan().any().item():
                print("forecast_after_linear.isnan().any().item() =", forecast.isnan().any().item())

        return forecast
