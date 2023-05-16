import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as f

from models.modules_utils import activation_layer

def scaled_dot_product_attention(query: t.Tensor, key: t.Tensor, value: t.Tensor) -> t.Tensor:
    temp = query.matmul(key.transpose(-1, -2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.matmul(value)


# the forward function of this module takes as input 3 matrices:
# query(X), key(X), value(X) perform linear dimension reduction, then scaled dot product attention
class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_k)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_v)

    def forward(self, query: t.Tensor, key: t.Tensor, value: t.Tensor) -> t.Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))

# this module performs attention mechanism for num_heads times,
# concatenate their results and return the resultant tensor
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query: t.Tensor, key: t.Tensor, value: t.Tensor) -> t.Tensor:
        return self.linear(
            t.cat([h(query, key, value) for h in self.heads], dim=-1)
        )

def feed_forward(
    dim_input: int = 512, dim_out: int = 512, dim_feedforward: int = 2048, activation: str = "relu"
) -> nn.Module:
    activation_func = activation_layer(activation)
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        activation_func,
        nn.Linear(dim_feedforward, dim_out),
    )

class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: t.Tensor) -> t.Tensor:
        # Assume that the "value" tensor is given last, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "tanh",
    ):
        super().__init__()
        dim_k, dim_v = dim_model // num_heads, dim_model // num_heads
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_k, dim_v),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_input=dim_model, dim_out=num_heads * dim_k,
                         dim_feedforward=dim_feedforward, activation=activation),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: t.Tensor) -> t.Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim_model, num_heads, dim_feedforward, dropout, activation
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: t.Tensor) -> t.Tensor:
        for layer in self.layers:
            src = layer(src)
        return src

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)
    return P
