import random

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from common.utils import count_parameters


class Encoder(nn.Module):
    def __init__(self, input_size,
                 e_hidden_dim,
                 e_num_layers,
                 batch_first=True,
                 dropout=0.2):
        super().__init__()

        self.rnn = nn.GRU(input_size=input_size,
                          hidden_size=e_hidden_dim,
                          num_layers=e_num_layers,
                          batch_first=True,
                          dropout=dropout)

    def forward(self, input_batch):
        # input_batch should be of shape (batch_size, Lin, Cin) Cin = nbr of channels, Lin = length of signal sequence
        outputs, e_hidden = self.rnn(input_batch)

        return outputs, e_hidden  # outputs of shape (batch_size, Lin, e_hidden_dim)


class Attention(nn.Module):
    def __init__(self, e_hidden_dim, d_hidden_dim, d_num_layers):
        super().__init__()

        self.d_hidden_dim = d_hidden_dim
        self.d_num_layers = d_num_layers

        # The input dimension will the the concatenation of
        # encoder_hidden_dim (hidden) and  decoder_hidden_dim(encoder_outputs)
        self.attn_hidden_vector = nn.Linear(e_hidden_dim + d_num_layers * d_hidden_dim, d_hidden_dim)

        # We need source len number of values for n batch as the dimension
        # of the attention weights. The attn_hidden_vector will have the
        # dimension of (batch_size, Lin, e_hidden_dim)
        # If we set the output dim of this Linear layer to 1 then the
        # effective output dimension will be [batch_size, Lin]
        self.attn_scoring_fn = nn.Linear(d_hidden_dim, 1, bias=False)

    def forward(self, d_hidden, encoder_outputs):
        # d_hidden of shape [d_num_layers, batch_size, d_hidden_dim]
        # encoder_outputs of shape (batch_size, Lin, e_hidden_dim) cause batch_first=True
        batch_size, Lin = encoder_outputs.shape[0], encoder_outputs.shape[1]

        # We need to calculate the attn_hidden for each source words.
        # Instead of repeating this using a loop, we can duplicate
        # hidden src_len number of times and perform the operations.
        d_hidden = d_hidden.view(batch_size, self.d_num_layers * self.d_hidden_dim)

        d_hidden = d_hidden.unsqueeze(1).repeat(1, Lin,
                                                1)  # hidden of shape [batch_size, Lin, d_num_layer * d_hidden_dim]

        # Calculate Attention Hidden values
        # attn_hidden of shape [batch_size, Lin, d_hidden_dim]
        attn_hidden = t.tanh(self.attn_hidden_vector(t.cat((d_hidden, encoder_outputs), dim=-1)))

        # Calculate the Scoring function. Remove 3rd dimension.
        # attn_scoring_vector of shape [batch_size, Lin]
        attn_scoring_vector = self.attn_scoring_fn(attn_hidden).squeeze(-1)

        # Softmax function for normalizing the weights to
        # probability distribution of shape (batch_size, Lin)
        return F.softmax(attn_scoring_vector, dim=-1)


class OneStepDecoder(nn.Module):
    def __init__(self, input_size,
                 e_hidden_dim,
                 d_hidden_dim,
                 d_num_layers,
                 attention,
                 dropout_prob=0.2,
                 batch_first=True):
        super().__init__()

        self.input_size = input_size  # output and
        self.e_hidden_dim = e_hidden_dim
        self.attention = attention

        # Add the encoder_hidden_dim and embedding_dim
        self.rnn = nn.GRU(input_size=input_size + e_hidden_dim,
                          hidden_size=d_hidden_dim,
                          num_layers=d_num_layers,
                          batch_first=batch_first,
                          dropout=dropout_prob)

        # Combine all the features for better prediction
        self.fc = nn.Linear(e_hidden_dim + d_hidden_dim + input_size, input_size)
        # self.fc = nn.Linear(d_hidden_dim, input_size)

    def forward(self, input, hidden, encoder_outputs):
        # input should be of shape (batch_size, input_size)
        # hidden of shape [d_num_layers, batch_size, d_hidden_dim]
        # encoder_outputs should be of shape (batch_size, Lin, e_hidden_dim) cause batch_first=True

        input = input.unsqueeze(1)  # input will be of shape (batch_size, 1, Cin)

        # Calculate the attention weights
        # probability distribution of shape (batch_size, 1, Lin)
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)

        # We need to perform the batch wise dot product.
        # Use PyTorch's bmm function to calculate the weighted e_outputs sum.
        # w will be of shape (batch_size, 1, e_hidden_dim)
        w = t.bmm(a, encoder_outputs)

        # concatenate the previous output with W
        # rnn_input will be of shape (batch_size, 1, e_hidden_dim + input_size)
        rnn_input = t.cat((input, w), dim=2)

        # output will be of shape (batch_size, 1, d_hidden_dim)
        # hidden will be of shape (d_num_layers, batch_size, d_hidden_dim)
        output, hidden = self.rnn(rnn_input, hidden)

        # Remove the sentence length dimension and pass them to the Linear layer
        # predicted_output will be of shape (batch_size, input_size)
        predicted_output = self.fc(t.cat((output.squeeze(1), w.squeeze(1), input.squeeze(1)), dim=1))

        return predicted_output, hidden, a.squeeze(1)


class Decoder(nn.Module):
    def __init__(self, input_size,
                 e_hidden_dim,
                 d_hidden_dim,
                 d_num_layers,
                 attention,
                 dropout_prob=0.2,
                 batch_first=True,):
        super().__init__()
        self.one_step_decoder = OneStepDecoder(input_size,
                                               e_hidden_dim,
                                               d_hidden_dim,
                                               d_num_layers,
                                               attention,
                                               dropout_prob,
                                               batch_first)

    def forward(self, last_src_input, encoder_outputs, hidden, target=None, horizon=None, teacher_forcing_ratio=0.5):
        # target should be of shape (batch_size, Lout, Cin) Cin = nbr of channels, Lout = length of out signal sequence
        # last_src_input = The last point before the beginning of target sequence of shape (batch_size, input_size)

        assert (horizon is not None) if (target is None) else True, "horizon must be provided in testing time"

        horizon = target.shape[1] if (target is not None) else horizon
        batch_size = last_src_input.shape[0]
        input_size = last_src_input.shape[1]

        outputs = t.zeros((batch_size, horizon, input_size), device=last_src_input.device)

        input = last_src_input

        for i in range(horizon):
            # Pass the encoder_outputs. For the first time step the
            # hidden state comes from the encoder model.
            output, hidden, a = self.one_step_decoder(input, hidden, encoder_outputs)

            outputs[:, i] = output

            teacher_force = random.random() < teacher_forcing_ratio if (target is not None) else False
            input = target[:, i] if teacher_force else output

        return outputs


class RNN_SQ2SQ(nn.Module):
    def __init__(self, input_size,
                 hidden_dim,
                 num_layers,
                 batch_first=True,
                 dropout=0.2,
                 teacher_forcing_ratio=0.5):
        super().__init__()

        e_hidden_dim = hidden_dim
        d_hidden_dim = hidden_dim

        e_num_layers = num_layers
        d_num_layers = num_layers

        self.teacher_forcing_ratio = teacher_forcing_ratio

        self.encoder = Encoder(input_size,
                 e_hidden_dim,
                 e_num_layers,
                 batch_first,
                 dropout)

        self.decoder = Decoder(input_size,
                 e_hidden_dim,
                 d_hidden_dim,
                 d_num_layers,
                 Attention(e_hidden_dim, d_hidden_dim, d_num_layers),
                 dropout,
                 batch_first)

    def forward(self, source, target=None, horizon=None):
        # source of shape (batch_size, Lin, input_dim)
        # target of shape (batch_size, Lout, input_dim)
        assert (horizon is not None) if (target is None) else True, "horizon must be provided in testing time"

        encoder_outputs, e_hidden = self.encoder(source)

        # e_hidden should be reshaped to match d_hidden structure if encoder rnn is different from decoder rnn

        last_src_input = source[:, -1]

        outputs = self.decoder(last_src_input, encoder_outputs, e_hidden, target, horizon, self.teacher_forcing_ratio)

        # outputs are of shape (batch_size, Lout, input_dim)
        return outputs
