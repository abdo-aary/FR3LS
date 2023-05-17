from typing import Callable

import torch as t


def mse_loss(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """
    Mean Squared Error

    :param prediction:
    :param target:
    :return:
    """
    assert prediction.shape == target.shape
    nan_mask = ~target.isnan()
    return t.nn.MSELoss()(prediction[nan_mask], target[nan_mask])


def msse_loss(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """
    Mean Squared Error

    :param prediction:
    :param target:
    :return:
    """
    assert prediction.shape == target.shape
    loss = ((prediction - target) ** 2).sum(dim=-1)
    return loss.mean()


def mae_loss(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """
    Mean Absolute Error

    :param prediction:
    :param target:
    :return:
    """
    assert prediction.shape == target.shape
    nan_mask = ~target.isnan()

    return t.nn.L1Loss()(prediction[nan_mask], target[nan_mask])


def rmse_loss(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """
    Root Mean Squared Error

    :param prediction:
    :param target:
    :return:
    """
    assert prediction.shape == target.shape
    return t.sqrt(mse_loss(prediction, target))


def mae_rmse_loss(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    assert prediction.shape == target.shape
    return (rmse_loss(prediction, target) + mae_loss(prediction, target)) / 2

def mape_loss(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """
    Mean absolute percentage error

    :param prediction:
    :param target:
    :return:
    """
    assert prediction.shape == target.shape

    mask_indices = target > 0

    nan_mask = ~target.isnan()
    mask_indices &= nan_mask

    return ((prediction[mask_indices] - target[mask_indices]).abs() / target[mask_indices].abs()).mean()


def smape_loss(prediction: t.Tensor, target: t.Tensor) -> t.Tensor:
    """
    Mean absolute percentage error

    :param prediction:
    :param target:
    :return:
    """
    assert prediction.shape == target.shape

    mask_indices = target > 0

    nan_mask = ~target.isnan()
    mask_indices &= nan_mask

    Pz = prediction[mask_indices]
    Az = target[mask_indices]

    return (2 * (Az - Pz).abs() / (Az.abs() + Pz.abs())).mean()


def wape_loss(prediction: t.Tensor, target: t.Tensor):
    nan_mask = ~target.isnan()

    return ((target[nan_mask] - prediction[nan_mask]).abs()).mean() / target[nan_mask].abs().mean()


def kld_loss(mu: t.Tensor, sigma: t.Tensor):
    kld = (sigma ** 2 + mu ** 2 - 1 - t.log(sigma ** 2)).sum(dim=-1).mean()  # KL divergence
    return kld


def vae_loss(prediction: t.Tensor, target: t.Tensor, mu: t.Tensor, sigma: t.Tensor):
    kld = (sigma ** 2 + mu ** 2 - 1 - t.log(sigma ** 2)).sum(dim=-1).mean()  # KL divergence

    # ae_loss = ((prediction - target) ** 2).sum(dim=-1)
    ae_loss = mae_loss(prediction, target)

    # return 0.5 * (kld + ae_loss).mean()
    return kld + ae_loss

def tempNC_loss(z1: t.Tensor, z2: t.Tensor, lambda_NC: float):
    # z1 & z2 are of shape (bs, w, latent_dim)
    z1, z2 = z1.flatten(0, 1), z2.flatten(0, 1)

    batch_size = z1.shape[0]
    latent_dim = z1.shape[-1]
    bn = t.nn.BatchNorm1d(latent_dim, affine=False).to(z1.device)

    # empirical cross-correlation matrix
    c = bn(z1).T @ bn(z2)
    c.div_(batch_size)

    # loss
    on_diag = t.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambda_NC * off_diag

    return loss

def gaussianNLP(x, mu, sigma=1):
    """
    Calculate negative log likelihood of Gaussian distribution
    :param mu: mean of shape (b, tw-fw, dim)
    :param sigma: = 1
    :param x: random variable - tensor of size (b, tw-fw, dim)
    """
    n_b = x.shape[0] * x.shape[1]  # We need to di vide along number of vectors = b * (tw - fw)

    negative_log_prob = 0.5 * ( t.log(t.tensor(2 * t.pi)) + (x - mu).pow(2).sum() )
    negative_log_prob /= n_b
    return negative_log_prob


def __loss_fn(loss_name: str) -> Callable:
    def loss(**kwargs):
        if loss_name == 'MSE':
            return mse_loss(**kwargs)
        elif loss_name == 'MAE':
            return mae_loss(**kwargs)
        elif loss_name == 'RMSE':
            return rmse_loss(**kwargs)
        elif loss_name == 'MAE_RMSE':
            return mae_rmse_loss(**kwargs)
        elif loss_name == 'MAPE':
            return mape_loss(**kwargs)
        elif loss_name == 'VAE':
            return vae_loss(**kwargs)
        elif loss_name == 'MSSE':
            return msse_loss(**kwargs)
        elif loss_name == 'TempNC':
            return tempNC_loss(**kwargs)
        elif loss_name == 'KLD':
            return kld_loss(**kwargs)
        elif loss_name == 'GaussianNLP':
            return gaussianNLP(**kwargs)
        else:
            raise Exception(f'Unknown loss function: {loss_name}')

    return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
