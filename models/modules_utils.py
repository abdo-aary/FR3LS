from typing import Optional

import torch as t
import torch.nn.functional as F

from models.f_models.lstm import LSTM_Modified


def activation_case_match(src: t.Tensor, activation: Optional[str] = None) -> t.Tensor:
    if activation is None:
        return src
    elif activation == 'relu':
        return t.relu(src)
    elif activation == 'gelu':
        return F.gelu(src)
    elif activation == 'sigmoid':
        return t.sigmoid(src)
    elif activation == 'tanh':
        return t.tanh(src)
    elif activation == 'leaky_relu':
        return t.nn.functional.leaky_relu(src)
    else:
        raise ValueError(f'activation {activation} not implemented')


def activation_layer(activation: str):
    if activation == 'relu':
        return t.nn.ReLU(True)
    elif activation == 'gelu':
        return t.nn.GELU()
    elif activation == 'sigmoid':
        return t.nn.Sigmoid()
    elif activation == 'tanh':
        return t.nn.Tanh()
    elif activation == 'leaky_relu':
        return t.nn.LeakyReLU()
    else:
        raise ValueError(f'activation {activation} not implemented')


def _init_weights_xavier(m):
    if isinstance(m, t.nn.Linear):
        t.nn.init.xavier_uniform_(m.weight)


def load_forecasting_model(params: dict) -> t.nn.Module:

    if params['model_type'] == 'LSTM_Modified':
        f_model = LSTM_Modified(input_size=params['input_size'],
                                output_size=params['output_size'],
                                hidden_size=params['hidden_size'],
                                num_layers=params['num_layers'],
                                batch_first=params['batch_first'],
                                dropout=params['dropout'],
                                )
        return f_model
    else:
        raise Exception(f"Unknown model {params['model_type']}")
