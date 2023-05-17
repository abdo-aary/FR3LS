"""
code taken from ...
"""
from types import NoneType

"""
PyTorch commonly used functions.
"""
from typing import Optional, Tuple, Union

import gin
import numpy as np
import torch as t
from tqdm import tqdm


@gin.configurable
def default_device(device_str_id: int = None) -> t.device:
    """
    PyTorch default device is GPU when available, CPU otherwise.

    :return: Default device.
    """
    device_id = 'cuda' if device_str_id is None else f'cuda:{device_str_id}'
    return t.device(device_id if t.cuda.is_available() else 'cpu')


def to_default_device(x: t.tensor, device_id: Optional[str] = None) -> t.tensor:
    """
    PyTorch default device is GPU when available, CPU otherwise.

    :return: Default device.
    """
    return x.to(default_device(device_id))


def to_tensor(array: Union[np.ndarray, None], device: Union[str, t.device] = None) -> Union[t.Tensor, None]:
    """
    Convert numpy array to tensor on default device.

    :param device: device to transform to, default_device() if None
    :param array: Numpy array to convert.
    :return: PyTorch tensor on default device.
    """
    if type(array) == NoneType:
        return None

    used_device = device if device else default_device()

    return t.tensor(array, device=used_device)


def divide_no_nan(a: t.Tensor, b: t.Tensor) -> t.Tensor:
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    return t.nan_to_num(a / b, 0, 0, 0)


def cov(X: t.Tensor, gd: Optional[t.Tensor] = None) -> t.Tensor:
    if gd is not None:
        X = X - gd
    D = X.shape[-1]
    C = 1 / (D - 1) * X @ X.transpose(-1, -2)
    if t.isinf(C).any() or t.isnan(C).any():
        print('BUG HERE')
    return C


def pearsonr(x: t.Tensor, y: t.Tensor) -> t.Tensor:
    mean_x = t.mean(x)
    mean_y = t.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = t.norm(xm, 2) * t.norm(ym, 2)
    r_val = r_num / r_den
    return r_val


def corr(X: t.tensor, eps: float = 1e-08) -> t.tensor:
    D = X.shape[-1]
    std = t.std(X, dim=-1).unsqueeze(-1)
    mean = t.mean(X, dim=-1).unsqueeze(-1)
    X = (X - mean) / (std + eps).detach()
    return 1 / (D - 1) * X @ X.transpose(-1, -2)


@gin.configurable
def get_corr(x: t.Tensor, y: Optional[t.Tensor] = None, y_mask: Optional[t.Tensor] = None,
             reduce: bool = True, divide_by_gt: bool = False, return_indices: bool = False, boosting_coef: int = 0,
             compute_cov=False, old_corr: bool = False) -> Tuple[Union[int, t.Tensor], t.Tensor]:
    assert len(x.shape) == 3
    if y_mask is not None:
        x = x * y_mask
    if y is not None:
        assert len(y.shape) == 3
        if not divide_by_gt:
            A = x - y
        else:
            B = (y_mask * y) + (1 - y_mask)
            A = (x - y) / B
            A[A != A] = 0
    else:
        A = x
    if compute_cov:
        # cor = -t.log(t.det(cov(A)))
        sign, cor = t.slogdet(cov(A))
        # cor = -t.log(sign * t.exp(cor))
        # cor = sign * cor
        # https://scicomp.stackexchange.com/questions/1122/how-to-add-large-exponential-terms-reliably-without-overflow-errors
        K = t.max(cor)
        cor = -t.log(sign * t.exp(cor - K))
        if t.isinf(cor).any() or t.isnan(cor).any():
            print('WARNING log determinant of slogdet is nan or inf. check if determinant too big')
            mask = t.isinf(cor) & t.isinf(cor)
            cor = cor[~mask]
        return cor.mean()
    else:
        if old_corr:
            cor = t.tril(t.abs(corr(A)), diagonal=-1)
            tril_indices = t.tril_indices(cor.shape[1], cor.shape[2], offset=-1).T.to(x.device)
            cor = cor[:, tril_indices[:, 0], tril_indices[:, 1]]
            cor = cor if not reduce else cor.mean()
            cor = cor + boosting_coef
        else:
            tril_indices = np.nan
            combs = t.combinations(t.arange(A.shape[1]), r=2)
            cor = [pearsonr(A[:, i].reshape(-1), A[:, j].reshape(-1)) for (i, j) in combs]
            cor = t.stack(cor)
            cor = cor if not reduce else cor.mean()
            cor = cor + boosting_coef
        if return_indices:
            return cor, tril_indices
        else:
            return cor


def embedd_dataset(X_forecast: np.ndarray, embedding_model: t.nn.Module, step: int = 1000,
                   device: str = None) -> np.ndarray:
    """

    :param device: device to use when computing embedding vectors
    :param X_forecast: array of shape (N, seq_length, embedd_in_dim)
    :param embedding_model: Model used for embedding
    :param step: jumping step
    :return: array of shape (N, seq_length, embedd_out_dim)
    """
    embedding_model.to(device) if device is not None else embedding_model.to(default_device())
    embedding_model.eval()

    X_out = None
    end_idx = 0
    for i in tqdm(range(step, len(X_forecast), step), desc="embedding vectors generation"):
        x = to_tensor(X_forecast[i - step: i], device)
        x_embedded = embedding_model.embedd(x).detach().cpu().numpy()
        if X_out is not None:
            X_out = np.concatenate((X_out, x_embedded))
        else:
            X_out = x_embedded

        end_idx = i

    if end_idx < len(X_forecast):
        x = to_tensor(X_forecast[end_idx:], device)
        x_embedded = embedding_model.embedd(x).detach().cpu().numpy()
        X_out = np.concatenate((X_out, x_embedded))

    return X_out


torch_dtype_dict = {
    "float16": t.float16,
    "float32": t.float32,
    "float64": t.float64,
    "int8": t.int8,
    "int16": t.int16,
    "int32": t.int32,
    "int64": t.int64,
    "uint8": t.uint8,
}


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[t.arange(all_indx.shape[0])[:, None], all_indx]


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs
