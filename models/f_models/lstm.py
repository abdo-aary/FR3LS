import torch as t
import torch.nn as nn


class LSTM_Modified(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 hidden_size,
                 num_layers,
                 batch_first=True,
                 dropout=0.2):
        super(LSTM_Modified, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=batch_first,
                            # proj_size=output_size,
                            dropout=dropout)

        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x: t.Tensor):
        # x should be of shape (N, Lin, Cin) Cin = nbr of channels, Lin = length of signal sequence

        output, (_, _) = self.lstm(x)  # output of shape (N, Lin, C_h)

        return self.fc(output[:, -1, :]) if self.fc else output[:, -1, :]  # Return the last lstm output only
