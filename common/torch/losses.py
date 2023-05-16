from typing import Callable

import torch as t

from models.ts2vec_models.losses import hierarchical_contrastive_loss, temporal_contrastive_loss


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


def huber_loss(prediction: t.Tensor, target: t.Tensor, delta=5) -> t.Tensor:
    """
    Root Mean Squared Error

    :param delta:
    :param prediction:
    :param target:
    :return:
    """
    assert prediction.shape == target.shape

    hubLoss = t.nn.HuberLoss(reduction='mean', delta=delta)
    return hubLoss(prediction, target)


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


def aldy_loss(Y: t.Tensor, Y_hat: t.Tensor, X_f_labels: t.Tensor,
              X_f_total: t.Tensor, R_view1: t.Tensor, R_view2: t.Tensor, lambda1: int = 1,
              lambda2: int = 1, lambda3: int = 1, train_ae_loss: str = 'MAE',
              train_forecasting_loss: str = 'MSE'):
    """
    Aldy Loss

    :param Y: of shape (N, Lin, C_Y)
    :param Y_hat: of shape (N, Lin, C_Y)
    :param X_f_labels: of shape (N2, Lin2, C_X)
    :param X_f_total: of shape (N2, Lin2, C_X)
    :param R_view1: of shape (N, Lin, C_R)
    :param R_view2: of shape (N, Lin, C_R)
    :param lambda1: weight assigned to the AE loss
    :param lambda2: weight assigned to the forecasting loss
    :param lambda3: weight assigned to the hierarchical contrastive loss
    :param train_ae_loss: Type of AE loss to be used
    :param train_forecasting_loss: Type of forecasting loss to be used
    :return:
    """

    ae_loss_fn = __loss_fn(train_ae_loss)
    forecasting_loss_fn = __loss_fn(train_forecasting_loss)

    ae_loss_Y = ae_loss_fn(prediction=Y_hat, target=Y)
    forecasting_loss_X = forecasting_loss_fn(prediction=X_f_total, target=X_f_labels)
    hier_loss_R = hierarchical_contrastive_loss(R_view1, R_view2)

    loss = lambda1 * ae_loss_Y + lambda2 * forecasting_loss_X + lambda3 * hier_loss_R

    return loss, hier_loss_R


def aldy_alternative_loss(Y: t.Tensor, Y_hat: t.Tensor, X_f_labels: t.Tensor,
                          X_f_total: t.Tensor, R_view1: t.Tensor, R_view2: t.Tensor, lambda1: int = 1,
                          lambda2: int = 1, lambda3: int = 1, train_ae_loss: str = 'MAE',
                          train_forecasting_loss: str = 'MSE', use_ts2vec: bool = True):
    """
    Aldy Loss

    :param Y: of shape (N, Lin, C_Y)
    :param Y_hat: of shape (N, Lin, C_Y)
    :param X_f_labels: of shape (N2, Lin2, C_X)
    :param X_f_total: of shape (N2, Lin2, C_X)
    :param R_view1: of shape (N, Lin, C_R)
    :param R_view2: of shape (N, Lin, C_R)
    :param lambda1: weight assigned to the AE loss
    :param lambda2: weight assigned to the forecasting loss
    :param lambda3: weight assigned to the hierarchical contrastive loss
    :param train_ae_loss: Type of AE loss to be used
    :param train_forecasting_loss: Type of forecasting loss to be used
    :param use_ts2vec:
    :return:
    """

    ae_loss_fn = __loss_fn(train_ae_loss)
    forecasting_loss_fn = __loss_fn(train_forecasting_loss)

    ae_loss_Y = ae_loss_fn(prediction=Y_hat, target=Y)
    forecasting_loss_X = forecasting_loss_fn(prediction=X_f_total, target=X_f_labels)
    hier_loss_R = hierarchical_contrastive_loss(R_view1, R_view2) if use_ts2vec else t.tensor(0)

    return ae_loss_Y, forecasting_loss_X, hier_loss_R


def tlae_loss(Y: t.Tensor, Y_hat: t.Tensor, X_f_labels: t.Tensor,
              X_f_total: t.Tensor, lambda1: int = 1,
              lambda2: int = 1, train_ae_loss: str = 'MAE',
              train_forecasting_loss: str = 'MAE'):
    """
    Tlae Loss

    :param Y: of shape (N, Lin, C_Y)
    :param Y_hat: of shape (N, Lin, C_Y)
    :param X_f_labels: of shape (N2, Lin2, C_X)
    :param X_f_total: of shape (N2, Lin2, C_X)
    :param lambda1: weight assigned to the AE loss
    :param lambda2: weight assigned to the forecasting loss
    :param train_ae_loss: Type of AE loss to be used
    :param train_forecasting_loss: Type of forecasting loss to be used
    :return:
    """

    ae_loss_fn = __loss_fn(train_ae_loss)
    forecasting_loss_fn = __loss_fn(train_forecasting_loss)

    ae_loss_Y = ae_loss_fn(prediction=Y_hat, target=Y)
    forecasting_loss_X = forecasting_loss_fn(prediction=X_f_total, target=X_f_labels)

    loss = lambda1 * ae_loss_Y + lambda2 * forecasting_loss_X

    return loss


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


def tempCNC_loss(z1: t.Tensor, z2: t.Tensor, lambda_NC: float):
    # tempNC = tempNC_loss(z1=z1, z2=z2, lambda_NC=lambda_NC)
    # tempC = temporal_contrastive_loss(z1=z1, z2=z2)
    # temp_loss = (tempNC + tempC) / 2
    # return temp_loss
    return (tempNC_loss(z1=z1, z2=z2, lambda_NC=lambda_NC) + temporal_contrastive_loss(z1=z1, z2=z2)) / 2


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
        elif loss_name == 'HUBER':
            return huber_loss(**kwargs)
        elif loss_name == 'MAPE':
            return mape_loss(**kwargs)
        elif loss_name == 'ALDY':
            return aldy_loss(**kwargs)
        elif loss_name == 'ALDY_ALTERNATIVE':
            return aldy_alternative_loss(**kwargs)
        elif loss_name == 'TLAE':
            return tlae_loss(**kwargs)
        elif loss_name == 'HIER':
            return hierarchical_contrastive_loss(**kwargs)
        elif loss_name == 'VAE':
            return vae_loss(**kwargs)
        elif loss_name == 'MSSE':
            return msse_loss(**kwargs)
        elif loss_name == 'TempC':
            return temporal_contrastive_loss(**kwargs)
        elif loss_name == 'TempNC':
            return tempNC_loss(**kwargs)
        elif loss_name == 'TempCNC':
            return tempCNC_loss(**kwargs)
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
