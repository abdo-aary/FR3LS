import torch as t
import torch.nn as nn
from models.f_models.deepLR_f_models.tcn import TemporalBlock


class TCN_Modified(nn.Module):
    def __init__(self, num_inputs, output_size, num_channels, kernel_size=2, dropout=0.2, leveld_init=True):
        super(TCN_Modified, self).__init__()
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

    def forward(self, x: t.Tensor, f_out_same_in_size: bool = False):
        # x should be of shape (N, Lin, Cin) Cin = nbr of channels, Lin = length of signal sequence

        forecast = self.tcn(x.transpose(1, 2)).transpose(1, 2)  # forecast of shape (N, Lin, Cin)

        if f_out_same_in_size:
            forecast = self.linear_out(forecast)
        else:
            forecast = self.linear_out(forecast[:, -1, :])

        return forecast
