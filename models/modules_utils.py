from typing import Optional

import torch as t
import torch.nn.functional as F

import numpy as np

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


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = t.full((B, T), True, dtype=t.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            k = np.random.randint(T - l + 1)
            res[i, k:k + l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return t.from_numpy(np.random.binomial(1, p, size=(B, T))).to(t.bool)
