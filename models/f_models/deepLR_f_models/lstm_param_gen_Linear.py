import torch as t
import torch.nn as nn


class LSTM_Modified(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 alpha: float = 1,
                 dropout=0.2):
        super(LSTM_Modified, self).__init__()
        self.alpha = alpha

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            # proj_size=output_size,
                            dropout=dropout)

        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x: t.Tensor, weights: t.Tensor, f_out_same_in_size: bool):
        # x should be of shape (N, Lin, Cin) Cin = nbr of channels, Lin = length of signal sequence
        # weights.shape = (output_size, hidden_size)

        if type(weights) == t.Tensor:
            if weights.is_cuda and not self.fc.weight.is_cuda:
                device = weights.device
                self.fc.weight = self.fc.weight.to(device)

        self.fc.weight = self.fc.weight * weights  # weights are considered as factors
        # self.linear_out.weight += weights * self.alpha

        # TODO: Try latter self.linear_out.weight = self.paramGen(linear_out.weight.flatten()).reshape(output_size,
        #  num_channels[-1])
        #  With self.paramGen module takes as input linear_out.weight and apply some non linear function on them

        output, (_, _) = self.lstm(x)  # output of shape (N, Lin, C_h)

        if f_out_same_in_size:
            return self.fc(output)
        else:
            return self.fc(output[:, -1, :])
