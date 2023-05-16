from collections.abc import Iterator

import numpy as np
import torch as t
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.neg_sampler import TimeSeriesSampler
from common.torch import CRPS
from common.torch.losses import __loss_fn, wape_loss, smape_loss, mape_loss
from common.torch.ops import default_device, to_tensor, take_per_row
from common.torch.snapshots import SnapshotManager


def vae_trainer(snapshot_manager: SnapshotManager,
                model: t.nn.Module,
                training_set: Iterator,
                sampler: TimeSeriesSampler,
                horizon: int,
                val_loss_name: str,
                train_ae_loss: str,
                train_forecasting_loss: str,
                train_ts2vec_loss: str,
                lambda_ae: int = 1,
                lambda_f: int = 1,
                lambda_t2v: int = 1,
                epochs: int = 750,
                learning_rate: float = 0.001,
                verbose: bool = True,
                early_stopping: bool = False,
                patience: int = 5,
                ):
    model = model.to(default_device())
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    forecasting_loss_fn = __loss_fn(train_forecasting_loss)
    ae_loss_fn = __loss_fn(train_ae_loss)
    ts2vec_loss_fn = __loss_fn(train_ts2vec_loss)
    val_forecasting_loss_fn = __loss_fn(val_loss_name)

    epoch = snapshot_manager.restore(model, optimizer)

    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss = np.inf
        num_no_improve = 0

        for epoch_i in range(epoch + 1, epochs + 1):
            optimized = False

            # Training
            training_batch_losses = []

            # Go through all possible training batches
            pb = tqdm(range(len(sampler.train_batches_indices)),
                      desc='epoch ' + str(epoch_i) + ' - train') if verbose else range(
                len(sampler.train_batches_indices))

            model.train()

            for _ in pb:
                y = to_tensor(next(training_set))
                optimizer.zero_grad()

                y_hat, r_view1, r_view2, z_f_labels, z_f, mu, sigma = model(y)

                training_batch_loss = lambda_ae * ae_loss_fn(prediction=y_hat, target=y, mu=mu,
                                                             sigma=sigma
                                                             ) + lambda_f * forecasting_loss_fn(
                    prediction=z_f, target=z_f_labels) + lambda_t2v * ts2vec_loss_fn(z1=r_view1, z2=r_view2)

                training_batch_loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_batch_losses.append(training_batch_loss.item())

            training_loss = np.array(training_batch_losses).mean()

            if verbose:
                print("Training_loss of epoch", epoch_i, "is:", training_loss)

            # Validation
            model.eval()
            input_windows_test, labels_test, input_windows_val = map(to_tensor, sampler.test_forecasting_windows())

            # Normalize input windows
            input_windows_test_normalized = (input_windows_test - to_tensor(sampler.train_ts_means)) / to_tensor(
                sampler.train_ts_std)

            # Forecast future values
            y_forecast_normalized = model.rolling_forecast(y=input_windows_test_normalized,
                                                           horizon=horizon)

            # Rescale forecasts back to original scale
            y_forecast = y_forecast_normalized * to_tensor(sampler.train_ts_std) + to_tensor(sampler.train_ts_means)

            mape_l = float(val_forecasting_loss_fn(prediction=y_forecast, target=labels_test))
            wape_test_loss = float(wape_loss(prediction=y_forecast, target=labels_test))
            smape_test_loss = float(smape_loss(prediction=y_forecast, target=labels_test))

            if verbose:
                print("Test loss of epoch", epoch_i, "is:", mape_l)

            if mape_l < min_loss:
                min_loss = mape_l
                optimized = True

            other_losses_values = {'WAPE': wape_test_loss, 'SMAPE': smape_test_loss}
            snapshot_manager.register(epoch=epoch_i, training_loss=training_loss,
                                      testing_loss=mape_l,
                                      other_losses_values=other_losses_values,
                                      save_model=optimized,
                                      model=model, optimizer=optimizer)

            if early_stopping:
                if optimized:
                    num_no_improve = 0
                else:
                    num_no_improve += 1

                if num_no_improve >= patience:
                    break

            snapshot_manager.print_losses()


def vanilla_ae_trainer(snapshot_manager: SnapshotManager,
                       model: t.nn.Module,
                       training_set: Iterator,
                       sampler: TimeSeriesSampler,
                       horizon: int,
                       val_loss_name: str,
                       train_ae_loss: str,
                       train_forecasting_loss: str,
                       train_ts2vec_loss: str,
                       lambda_ae: int = 1,
                       lambda_f: int = 1,
                       lambda_t2v: int = 1,
                       epochs: int = 750,
                       learning_rate: float = 0.001,
                       verbose: bool = True,
                       early_stopping: bool = False,
                       patience: int = 5,
                       ):
    model = model.to(default_device())
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    forecasting_loss_fn = __loss_fn(train_forecasting_loss)
    ae_loss_fn = __loss_fn(train_ae_loss)
    ts2vec_loss_fn = __loss_fn(train_ts2vec_loss)
    val_forecasting_loss_fn = __loss_fn(val_loss_name)

    epoch = snapshot_manager.restore(model, optimizer)

    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss = np.inf
        num_no_improve = 0

        for epoch_i in range(epoch + 1, epochs + 1):
            optimized = False

            # Training
            training_batch_losses = []

            # Go through all possible training batches
            pb = tqdm(range(len(sampler.train_batches_indices)),
                      desc='epoch ' + str(epoch_i) + ' - train') if verbose else range(
                len(sampler.train_batches_indices))

            model.train()

            for _ in pb:
                y = to_tensor(next(training_set))
                optimizer.zero_grad()

                y_hat, r_view1, r_view2, z_f_labels, z_f = model(y)

                ae_loss = ae_loss_fn(prediction=y_hat, target=y)
                f_loss = forecasting_loss_fn(prediction=z_f, target=z_f_labels)
                t2v_loss = ts2vec_loss_fn(z1=r_view1, z2=r_view2)

                training_batch_loss = lambda_ae * ae_loss + lambda_f * f_loss + lambda_t2v * t2v_loss

                training_batch_loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_batch_losses.append(training_batch_loss.item())

            training_loss = np.array(training_batch_losses).mean()

            if verbose:
                print("Training_loss of epoch", epoch_i, "is:", training_loss)

            # Validation
            model.eval()
            input_windows_test, labels_test, input_windows_val = map(to_tensor, sampler.test_forecasting_windows())

            # Normalize input windows
            input_windows_test_normalized = (input_windows_test - to_tensor(sampler.train_ts_means)) / to_tensor(
                sampler.train_ts_std)

            # Forecast future values
            y_forecast_normalized = model.rolling_forecast(y=input_windows_test_normalized,
                                                           horizon=horizon)

            # Rescale forecasts back to original scale
            y_forecast = y_forecast_normalized * to_tensor(sampler.train_ts_std) + to_tensor(sampler.train_ts_means)

            test_loss = float(val_forecasting_loss_fn(prediction=y_forecast, target=labels_test))
            mape_test_loss = float(mape_loss(prediction=y_forecast, target=labels_test))
            wape_test_loss = float(wape_loss(prediction=y_forecast, target=labels_test))
            smape_test_loss = float(smape_loss(prediction=y_forecast, target=labels_test))

            if verbose:
                print("Test loss of epoch", epoch_i, "is:", test_loss)

            if test_loss < min_loss:
                min_loss = test_loss
                optimized = True

            other_losses_values = {'MAPE': mape_test_loss, 'WAPE': wape_test_loss, 'SMAPE': smape_test_loss}
            snapshot_manager.register(epoch=epoch_i, training_loss=training_loss,
                                      testing_loss=test_loss,
                                      other_losses_values=other_losses_values,
                                      save_model=optimized,
                                      model=model, optimizer=optimizer)

            if early_stopping:
                if optimized:
                    num_no_improve = 0
                else:
                    num_no_improve += 1

                if num_no_improve >= patience:
                    break

            snapshot_manager.print_losses()


def ae_t2v_trainer(snapshot_manager: SnapshotManager,
                   model: t.nn.Module,
                   training_set: Iterator,
                   sampler: TimeSeriesSampler,
                   horizon: int,
                   val_loss_name: str,
                   train_ae_loss: str,
                   train_forecasting_loss: str,
                   train_ts2vec_loss: str,
                   lambda_ae: int = 1,
                   lambda_f: int = 1,
                   lambda_t2v: int = 1,
                   epochs: int = 750,
                   learning_rate: float = 0.001,
                   verbose: bool = True,
                   early_stopping: bool = False,
                   patience: int = 5,
                   ):
    model = model.to(default_device())
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    forecasting_loss_fn = __loss_fn(train_forecasting_loss)
    ae_loss_fn = __loss_fn(train_ae_loss)
    ts2vec_loss_fn = __loss_fn(train_ts2vec_loss)
    val_forecasting_loss_fn = __loss_fn(val_loss_name)

    epoch = snapshot_manager.restore(model, optimizer)

    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss = np.inf
        num_no_improve = 0

        for epoch_i in range(epoch + 1, epochs + 1):
            optimized = False

            # Training
            training_batch_losses = []

            # Go through all possible training batches
            pb = tqdm(range(len(sampler.train_batches_indices)),
                      desc='epoch ' + str(epoch_i) + ' - train') if verbose else range(
                len(sampler.train_batches_indices))

            model.train()

            for _ in pb:
                y = to_tensor(next(training_set))
                optimizer.zero_grad()

                y_hat, z_view1, z_view2, z_f_labels, z_f = model(y)

                ae_loss = ae_loss_fn(prediction=y_hat, target=y)
                f_loss = forecasting_loss_fn(prediction=z_f, target=z_f_labels)
                t2v_loss = ts2vec_loss_fn(z1=z_view1, z2=z_view2)

                training_batch_loss = lambda_ae * ae_loss + lambda_f * f_loss + lambda_t2v * t2v_loss

                training_batch_loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_batch_losses.append(training_batch_loss.item())

            training_loss = np.array(training_batch_losses).mean()

            if verbose:
                print("Training_loss of epoch", epoch_i, "is:", training_loss)

            # Validation
            model.eval()
            input_windows_test, labels_test, input_windows_val = map(to_tensor, sampler.test_forecasting_windows())

            # Normalize input windows
            input_windows_test_normalized = (input_windows_test - to_tensor(sampler.train_ts_means)) / to_tensor(
                sampler.train_ts_std)

            # Forecast future values
            y_forecast_normalized = model.rolling_forecast(y=input_windows_test_normalized,
                                                           horizon=horizon)

            # Rescale forecasts back to original scale
            y_forecast = y_forecast_normalized * to_tensor(sampler.train_ts_std) + to_tensor(sampler.train_ts_means)

            test_loss = float(val_forecasting_loss_fn(prediction=y_forecast, target=labels_test))
            mape_test_loss = float(mape_loss(prediction=y_forecast, target=labels_test))
            wape_test_loss = float(wape_loss(prediction=y_forecast, target=labels_test))
            smape_test_loss = float(smape_loss(prediction=y_forecast, target=labels_test))

            if verbose:
                print("Test loss of epoch", epoch_i, "is:", test_loss)

            if test_loss < min_loss:
                min_loss = test_loss
                optimized = True

            other_losses_values = {'MAPE': mape_test_loss, 'WAPE': wape_test_loss, 'SMAPE': smape_test_loss}
            snapshot_manager.register(epoch=epoch_i, training_loss=training_loss,
                                      testing_loss=test_loss,
                                      other_losses_values=other_losses_values,
                                      save_model=optimized,
                                      model=model, optimizer=optimizer)

            if early_stopping:
                if optimized:
                    num_no_improve = 0
                else:
                    num_no_improve += 1

                if num_no_improve >= patience:
                    break

            snapshot_manager.print_losses()


def raw_t2v_trainer(snapshot_manager: SnapshotManager,
                    model: t.nn.Module,
                    training_set: Iterator,
                    sampler: TimeSeriesSampler,
                    horizon: int,
                    train_ts2vec_loss: str,
                    epochs: int = 750,
                    learning_rate: float = 0.001,
                    verbose: bool = True,
                    early_stopping: bool = False,
                    patience: int = 5,
                    ):
    model = model.to(default_device())
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    ts2vec_loss_fn = __loss_fn(train_ts2vec_loss)

    epoch = snapshot_manager.restore(model, optimizer)

    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss = np.inf
        num_no_improve = 0

        for epoch_i in range(epoch + 1, epochs + 1):
            optimized = False

            # Training
            training_batch_losses = []

            # Go through all possible training batches
            pb = tqdm(range(len(sampler.train_batches_indices)),
                      desc='epoch ' + str(epoch_i) + ' - train') if verbose else range(
                len(sampler.train_batches_indices))

            model.train()

            for _ in pb:
                y = to_tensor(next(training_set))
                optimizer.zero_grad()

                z_view1, z_view2 = model(y, use_mask=True), model(y, use_mask=True)

                training_batch_loss = ts2vec_loss_fn(z1=z_view1, z2=z_view2)

                training_batch_loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_batch_losses.append(training_batch_loss.item())

            training_loss = np.array(training_batch_losses).mean()

            if verbose:
                print("Training_loss of epoch", epoch_i, "is:", training_loss)

            # Validation
            model.eval()
            input_windows_test, labels_test, input_windows_val = map(to_tensor, sampler.test_forecasting_windows())

            # Normalize labels windows
            y_test = (labels_test - to_tensor(sampler.train_ts_means)) / to_tensor(
                sampler.train_ts_std)

            z_view1_test, z_view2_test = model(y_test, use_mask=True), model(y_test, use_mask=True)

            test_loss = float(ts2vec_loss_fn(z1=z_view1_test, z2=z_view2_test))

            if verbose:
                print("Test loss of epoch", epoch_i, "is:", test_loss)

            if test_loss < min_loss:
                min_loss = test_loss
                optimized = True

            snapshot_manager.register(epoch=epoch_i, training_loss=training_loss,
                                      testing_loss=test_loss,
                                      save_model=optimized,
                                      model=model, optimizer=optimizer)

            if early_stopping:
                if optimized:
                    num_no_improve = 0
                else:
                    num_no_improve += 1

                if num_no_improve >= patience:
                    break

            snapshot_manager.print_losses()


def ae_temp_trainer(snapshot_manager: SnapshotManager,
                    model: t.nn.Module,
                    dataLoader: DataLoader,
                    horizon: int,
                    test_loss_name: str,
                    train_ae_loss: str,
                    train_forecasting_loss: str,
                    train_temp_loss: str,
                    lambda_ae: float = 1,
                    lambda_f: float = 1,
                    lambda_temp: float = 1,
                    lambda_NC: float = 5e-3,
                    epochs: int = 750,
                    learning_rate: float = 0.001,
                    verbose: bool = True,
                    pbar_percentage: int = 20,
                    early_stopping: bool = False,
                    patience: int = 5,
                    device_id: int = None,
                    n_best_test_losses: int = None,
                    lr_warmup: int = None,
                    mv_training: bool = True,
                    pretrain_epochs: int = 0,
                    num_samples: int = 0,
                    ):
    device = default_device(device_str_id=device_id)
    print("device =", device)

    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    forecasting_loss_fn = __loss_fn(train_forecasting_loss)
    ae_loss_fn = __loss_fn(train_ae_loss)
    temp_loss_fn = __loss_fn(train_temp_loss)
    test_forecasting_loss_fn = __loss_fn(test_loss_name)

    epoch = snapshot_manager.restore(model, optimizer)

    model = model.to(device)
    snapshot_manager.enable_time_tracking()

    if lr_warmup:
        scheduler = t.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda j: min(j / (lr_warmup / dataLoader.dataset.train_window), 1.0))

    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss_test = np.inf
        min_loss_mape = np.inf
        num_no_improve = 0

        ae_pretraining = True
        f_model_pretraining = False

        for epoch_i in range(epoch + 1, epochs + 1):
            if epoch_i > pretrain_epochs // 2:
                if ae_pretraining:
                    ae_pretraining = False
                    print('AE Pretraining is finished')
                if not f_model_pretraining and epoch_i <= pretrain_epochs:
                    f_model_pretraining = True
                elif f_model_pretraining and epoch_i > pretrain_epochs:
                    f_model_pretraining = False
                    print('f_model Pretraining is finished')

            optimized = False

            # Training
            training_batch_losses = []

            # Go through all training batches
            if verbose:
                pbar = tqdm(total=len(dataLoader), desc='epoch ' + str(epoch_i) + ' - train')
                update_interval, progressed = len(dataLoader) * pbar_percentage / 100, 1

            model.train()

            # model.requires_grad_(True)

            for i, y in enumerate(dataLoader):
                optimizer.zero_grad()
                y = y.to(device)

                if len(y.shape) == 4:
                    y = y.view(y.shape[0] * y.shape[1], y.shape[2], y.shape[3])

                if ae_pretraining:
                    y_hat, x_v1, x_v2, _, _ = model(y, use_f_m=False)
                    ae_loss = ae_loss_fn(prediction=y_hat, target=y)
                    f_loss = 0
                    temp_loss = temp_loss_fn(z1=x_v1, z2=x_v2, lambda_NC=lambda_NC) if lambda_temp > 0 else 0
                elif f_model_pretraining:
                    _, _, _, x_f_labels, x_f = model(y, use_f_m=True)
                    ae_loss = 0
                    f_loss = forecasting_loss_fn(prediction=x_f, target=x_f_labels)
                    temp_loss = 0
                else:
                    y_hat, x_v1, x_v2, x_f_labels, x_f = model(y, use_f_m=True)
                    ae_loss = ae_loss_fn(prediction=y_hat, target=y)
                    f_loss = forecasting_loss_fn(prediction=x_f, target=x_f_labels)
                    temp_loss = temp_loss_fn(z1=x_v1, z2=x_v2, lambda_NC=lambda_NC) if lambda_temp > 0 else 0

                training_batch_loss = lambda_ae * ae_loss + lambda_f * f_loss + lambda_temp * temp_loss

                training_batch_loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if lr_warmup:
                    scheduler.step()
                training_batch_losses.append(training_batch_loss.item())
                t.cuda.empty_cache()

                if verbose:
                    if i + 1 == len(dataLoader) or (i + 1) / (progressed * update_interval) > 1:
                        pbar.update(n=round(progressed * update_interval, 2) - pbar.n)
                        progressed += 1
            if verbose:
                pbar.close()

            training_loss = np.array(training_batch_losses).mean()

            if verbose:
                print("Training_loss of epoch", epoch_i, "is:", training_loss)

            # Validation
            with t.no_grad():
                model.eval()

                input_windows_test, labels_test, input_windows_val = map(to_tensor,
                                                                         dataLoader.dataset.test_forecasting_windows(),
                                                                         (device,) * 3)

                # Normalize input windows
                input_windows_test_normalized = (input_windows_test - to_tensor(dataLoader.dataset.train_ts_means,
                                                                                device=device)) / to_tensor(
                    dataLoader.dataset.train_ts_std, device=device)

                # Forecast future values
                y_forecast_normalized = model.rolling_forecast(y=input_windows_test_normalized,
                                                               horizon=horizon)

                # Rescale forecasts back to original scale
                y_forecast = y_forecast_normalized * to_tensor(dataLoader.dataset.train_ts_std,
                                                               device=device) + to_tensor(
                    dataLoader.dataset.train_ts_means, device=device)

                test_loss = float(test_forecasting_loss_fn(prediction=y_forecast, target=labels_test))
                mape_test_loss = float(mape_loss(prediction=y_forecast, target=labels_test))
                wape_test_loss = float(wape_loss(prediction=y_forecast, target=labels_test))
                smape_test_loss = float(smape_loss(prediction=y_forecast, target=labels_test))

                if verbose:
                    print("Test loss of epoch", epoch_i, "is:", test_loss)

                # Register model if any of the metrics have changed
                if test_loss < min_loss_test:
                    min_loss_test = test_loss
                    optimized = True
                if mape_test_loss < min_loss_mape:
                    min_loss_mape = mape_test_loss
                    optimized = True

                other_losses_values = {'MAPE': mape_test_loss, 'WAPE': wape_test_loss, 'SMAPE': smape_test_loss}
                snapshot_manager.register(epoch=epoch_i, training_loss=training_loss,
                                          testing_loss=test_loss,
                                          other_losses_values=other_losses_values,
                                          save_model=optimized,
                                          model=model, optimizer=optimizer)

                t.cuda.empty_cache()

                if early_stopping:
                    if optimized:
                        num_no_improve = 0
                    else:
                        num_no_improve += 1

                    if num_no_improve >= patience:
                        break

                snapshot_manager.print_losses(top_best=n_best_test_losses)


def ae_pretraining_temp_trainer(snapshot_manager: SnapshotManager,
                                model: t.nn.Module,
                                dataLoader: DataLoader,
                                horizon: int,
                                test_loss_name: str,
                                train_ae_loss: str,
                                train_forecasting_loss: str,
                                train_temp_loss: str,
                                lambda_ae: float = 1,
                                lambda_f: float = 1,
                                lambda_temp: float = 1,
                                lambda_NC: float = 5e-3,
                                epochs: int = 750,
                                learning_rate: float = 0.001,
                                verbose: bool = True,
                                pbar_percentage: int = 20,
                                early_stopping: bool = False,
                                patience: int = 5,
                                device_id: int = None,
                                n_best_test_losses: int = None,
                                lr_warmup: int = None,
                                mv_training: bool = True,
                                pretrain_epochs: int = 0,
                                num_samples: int = None,
                                ):
    device = default_device(device_str_id=device_id)

    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    forecasting_loss_fn = __loss_fn(train_forecasting_loss)
    ae_loss_fn = __loss_fn(train_ae_loss)
    temp_loss_fn = __loss_fn(train_temp_loss)
    test_forecasting_loss_fn = __loss_fn(test_loss_name)

    epoch = snapshot_manager.restore(model, optimizer)

    model = model.to(device)
    snapshot_manager.enable_time_tracking()

    if lr_warmup:
        scheduler = t.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda j: min(j / (lr_warmup / dataLoader.dataset.train_window), 1.0))

    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss_test = np.inf
        min_pretrain_loss_test = np.inf
        min_loss_mape = np.inf
        # min_loss_mape = np.inf
        num_no_improve = 0

        ae_pretraining = True

        for epoch_i in range(epoch + 1, epochs + 1):
            if epoch_i > pretrain_epochs:
                ae_pretraining = False

            optimized = False

            # Training
            training_batch_losses = []

            # Go through all training batches
            if verbose:
                pbar = tqdm(total=len(dataLoader), desc='epoch ' + str(epoch_i) + ' - train')
                update_interval, progressed = len(dataLoader) * pbar_percentage / 100, 1

            model.train()

            # model.requires_grad_(True)

            for i, y in enumerate(dataLoader):
                optimizer.zero_grad()
                y = y.to(device)

                if len(y.shape) == 4:
                    y = y.view(y.shape[0] * y.shape[1], y.shape[2], y.shape[3])

                if mv_training:
                    if ae_pretraining:
                        y_hat_v1, y_hat_v2, x_hat_v1, x_hat_v2, _, _, _, _ = model(y, use_f_m=False)
                        ae_loss = (ae_loss_fn(prediction=y_hat_v1, target=y) + ae_loss_fn(prediction=y_hat_v2,
                                                                                          target=y)) / 2
                        f_loss = 0
                        temp_loss = temp_loss_fn(z1=x_hat_v1, z2=x_hat_v2, lambda_NC=lambda_NC)
                    else:
                        y_hat_v1, y_hat_v2, x_hat_v1, x_hat_v2, x_f_labels_v1, x_f_v1, x_f_labels_v2, x_f_v2 = model(y,
                                                                                                                     use_f_m=True)
                        ae_loss = (ae_loss_fn(prediction=y_hat_v1, target=y) + ae_loss_fn(prediction=y_hat_v2,
                                                                                          target=y)) / 2
                        f_loss = (forecasting_loss_fn(prediction=x_f_v1, target=x_f_labels_v1) + forecasting_loss_fn(
                            prediction=x_f_v2, target=x_f_labels_v2)) / 2
                        temp_loss = temp_loss_fn(z1=x_hat_v1, z2=x_hat_v2, lambda_NC=lambda_NC)
                else:
                    if ae_pretraining:
                        y_hat, x_v1, x_v2, _, _ = model(y, use_f_m=False)
                        ae_loss = ae_loss_fn(prediction=y_hat, target=y)
                        f_loss = 0
                        temp_loss = temp_loss_fn(z1=x_v1, z2=x_v2, lambda_NC=lambda_NC)
                    else:
                        y_hat, x_v1, x_v2, x_f_labels, x_f = model(y, use_f_m=True)
                        ae_loss = ae_loss_fn(prediction=y_hat, target=y)
                        f_loss = forecasting_loss_fn(prediction=x_f, target=x_f_labels)
                        temp_loss = temp_loss_fn(z1=x_v1, z2=x_v2, lambda_NC=lambda_NC)

                training_batch_loss = lambda_ae * ae_loss + lambda_f * f_loss + lambda_temp * temp_loss

                training_batch_loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if lr_warmup:
                    scheduler.step()
                training_batch_losses.append(training_batch_loss.item())
                t.cuda.empty_cache()

                if verbose:
                    if i + 1 == len(dataLoader) or (i + 1) / (progressed * update_interval) > 1:
                        pbar.update(n=round(progressed * update_interval, 2) - pbar.n)
                        progressed += 1
            if verbose:
                pbar.close()

            training_loss = np.array(training_batch_losses).mean()

            if verbose:
                print("Training_loss of epoch", epoch_i, "is:", training_loss)

            # Validation
            with t.no_grad():
                model.eval()

                input_windows_test, labels_test, input_windows_val = map(to_tensor,
                                                                         dataLoader.dataset.test_forecasting_windows(),
                                                                         (device,) * 3)

                # Normalize input windows
                input_windows_test_normalized = (input_windows_test - to_tensor(dataLoader.dataset.train_ts_means,
                                                                                device=device)) / to_tensor(
                    dataLoader.dataset.train_ts_std, device=device)
                labels_test_normalized = (labels_test - to_tensor(dataLoader.dataset.train_ts_means,
                                                                  device=device)) / to_tensor(
                    dataLoader.dataset.train_ts_std, device=device)

                if ae_pretraining and mv_training:
                    y_hat_labels_normalized = model.encode_decode(labels_test_normalized)
                    y_hat_labels = y_hat_labels_normalized * to_tensor(dataLoader.dataset.train_ts_std,
                                                                       device=device) + to_tensor(
                        dataLoader.dataset.train_ts_means, device=device)

                    test_loss = float(ae_loss_fn(prediction=y_hat_labels, target=labels_test))
                    mape_test_loss = np.inf
                    wape_test_loss = np.inf
                    smape_test_loss = np.inf
                else:
                    # Forecast future values
                    y_forecast_normalized = model.rolling_forecast(y=input_windows_test_normalized,
                                                                   horizon=horizon)

                    # Rescale forecasts back to original scale
                    y_forecast = y_forecast_normalized * to_tensor(dataLoader.dataset.train_ts_std,
                                                                   device=device) + to_tensor(
                        dataLoader.dataset.train_ts_means, device=device)

                    test_loss = float(test_forecasting_loss_fn(prediction=y_forecast, target=labels_test))
                    mape_test_loss = float(mape_loss(prediction=y_forecast, target=labels_test))
                    wape_test_loss = float(wape_loss(prediction=y_forecast, target=labels_test))
                    smape_test_loss = float(smape_loss(prediction=y_forecast, target=labels_test))

                if verbose:
                    print("Eval losses of ep", epoch_i, "are :", (test_loss, mape_test_loss, wape_test_loss,
                                                                  smape_test_loss))

                # Register model if any of the metrics have changed
                if ae_pretraining:
                    if test_loss < min_pretrain_loss_test:
                        min_pretrain_loss_test = test_loss
                        optimized = True
                else:
                    if test_loss < min_loss_test:
                        min_loss_test = test_loss
                        optimized = True
                    if mape_test_loss < min_loss_mape:
                        min_loss_mape = mape_test_loss
                        optimized = True

                other_losses_values = {'MAPE': mape_test_loss, 'WAPE': wape_test_loss, 'SMAPE': smape_test_loss}
                snapshot_manager.register(epoch=epoch_i, training_loss=training_loss,
                                          testing_loss=test_loss,
                                          other_losses_values=other_losses_values,
                                          save_model=optimized,
                                          model=model, optimizer=optimizer)

                t.cuda.empty_cache()

                if early_stopping:
                    if optimized:
                        num_no_improve = 0
                    else:
                        num_no_improve += 1

                    if num_no_improve >= patience:
                        break

                snapshot_manager.print_losses(top_best=n_best_test_losses)


def vae_tempC_trainer(snapshot_manager: SnapshotManager,
                      model: t.nn.Module,
                      training_set: Iterator,
                      sampler: TimeSeriesSampler,
                      horizon: int,
                      val_loss_name: str,
                      train_ae_loss: str,
                      train_forecasting_loss: str,
                      train_tempC_loss: str,
                      train_distr_reg_loss: str,
                      lambda_ae: int = 1,
                      lambda_f: int = 1,
                      lambda_tempC: int = 1,
                      lambda_distr: int = 1,
                      epochs: int = 750,
                      learning_rate: float = 0.001,
                      verbose: bool = True,
                      early_stopping: bool = False,
                      patience: int = 5,
                      ):
    model = model.to(default_device())
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    forecasting_loss_fn = __loss_fn(train_forecasting_loss)
    ae_loss_fn = __loss_fn(train_ae_loss)
    tempC_loss_fn = __loss_fn(train_tempC_loss)
    train_distr_reg_loss_fn = __loss_fn(train_distr_reg_loss)
    val_forecasting_loss_fn = __loss_fn(val_loss_name)

    epoch = snapshot_manager.restore(model, optimizer)

    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss = np.inf
        num_no_improve = 0

        for epoch_i in range(epoch + 1, epochs + 1):
            optimized = False

            # Training
            training_batch_losses = []

            # Go through all possible training batches
            pb = tqdm(range(len(sampler.train_batches_indices)),
                      desc='epoch ' + str(epoch_i) + ' - train') if verbose else range(
                len(sampler.train_batches_indices))

            model.train()

            for _ in pb:
                y = to_tensor(next(training_set))
                optimizer.zero_grad()

                y_hat, x_v1, x_v2, x_f_labels, x_f, mu, sigma = model(y)

                ae_loss = ae_loss_fn(prediction=y_hat, target=y)
                f_loss = forecasting_loss_fn(prediction=x_f, target=x_f_labels)
                tempC_loss = tempC_loss_fn(z1=x_v1, z2=x_v2)
                distr_loss = train_distr_reg_loss_fn(mu=mu, sigma=sigma)

                training_batch_loss = lambda_ae * ae_loss + lambda_f * f_loss + lambda_tempC * tempC_loss + lambda_distr * distr_loss

                training_batch_loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_batch_losses.append(training_batch_loss.item())

            training_loss = np.array(training_batch_losses).mean()

            if verbose:
                print("Training_loss of epoch", epoch_i, "is:", training_loss)

            # Validation
            model.eval()
            input_windows_test, labels_test, input_windows_val = map(to_tensor, sampler.test_forecasting_windows())

            # Normalize input windows
            input_windows_test_normalized = (input_windows_test - to_tensor(sampler.train_ts_means)) / to_tensor(
                sampler.train_ts_std)

            # Forecast future values
            y_forecast_normalized = model.rolling_forecast(y=input_windows_test_normalized,
                                                           horizon=horizon)

            # Rescale forecasts back to original scale
            y_forecast = y_forecast_normalized * to_tensor(sampler.train_ts_std) + to_tensor(sampler.train_ts_means)

            test_loss = float(val_forecasting_loss_fn(prediction=y_forecast, target=labels_test))
            mape_test_loss = float(mape_loss(prediction=y_forecast, target=labels_test))
            wape_test_loss = float(wape_loss(prediction=y_forecast, target=labels_test))
            smape_test_loss = float(smape_loss(prediction=y_forecast, target=labels_test))

            if verbose:
                print("Test loss of epoch", epoch_i, "is:", test_loss)

            if test_loss < min_loss:
                min_loss = test_loss
                optimized = True

            other_losses_values = {'MAPE': mape_test_loss, 'WAPE': wape_test_loss, 'SMAPE': smape_test_loss}
            snapshot_manager.register(epoch=epoch_i, training_loss=training_loss,
                                      testing_loss=test_loss,
                                      other_losses_values=other_losses_values,
                                      save_model=optimized,
                                      model=model, optimizer=optimizer)

            if early_stopping:
                if optimized:
                    num_no_improve = 0
                else:
                    num_no_improve += 1

                if num_no_improve >= patience:
                    break

            snapshot_manager.print_losses()


def t2vLR_trainer(snapshot_manager_t2v: SnapshotManager,
                  snapshot_manager_LR: SnapshotManager,
                  model: t.nn.Module,
                  avg_model: AveragedModel,
                  sampler: TimeSeriesSampler,
                  linear_regression: t.nn.Module,
                  dataLoader: DataLoader,
                  horizon: int,
                  ts2vec_loss_fn: str,
                  train_LR_loss: str,
                  test_LR_loss: str,
                  epochs_t2v: int,
                  epochs_LR: int,
                  learning_rate_t2v: float,
                  learning_rate_LR: float,
                  verbose: bool = True,
                  pbar_percentage: int = 20,
                  early_stopping: bool = False,
                  patience: int = 5,
                  device_id: int = None,
                  n_best_test_losses: int = None,
                  lr_warmup: int = None,
                  temporal_unit: int = 0,
                  max_train_length: int = 3000,
                  ):
    device = default_device(device_str_id=device_id)

    optimizer_t2v = t.optim.AdamW(model.parameters(), lr=learning_rate_t2v)
    ts2vec_loss_fn = __loss_fn(ts2vec_loss_fn)
    train_LR_loss_fn = __loss_fn(train_LR_loss)
    test_forecasting_loss_fn = __loss_fn(test_LR_loss)

    # epoch = snapshot_manager_t2v.restore(avg_model, optimizer_t2v)
    # model = avg_model.module.to(device)
    epoch = snapshot_manager_t2v.restore(model, optimizer_t2v)
    model = model.to(device)

    snapshot_manager_t2v.enable_time_tracking()

    if lr_warmup:
        scheduler = t.optim.lr_scheduler.LambdaLR(optimizer_t2v,
                                                  lambda j: min(j / (lr_warmup / sampler.train_window), 1.0))
    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss_test = np.inf
        num_no_improve = 0

        for epoch_i in range(epoch + 1, epochs_t2v + 1):
            optimized = False

            # Training
            training_batch_losses = []

            # Go through all training batches
            if verbose:
                pbar = tqdm(total=len(dataLoader), desc='epoch ' + str(epoch_i) + ' - train')
                update_interval, progressed = len(dataLoader) * pbar_percentage / 100, 1

            # avg_model.train()
            model.train()
            # model.requires_grad_(True)

            for i, x in enumerate(dataLoader):
                optimizer_t2v.zero_grad()

                if max_train_length and x.size(1) > max_train_length:
                    window_offset = np.random.randint(x.size(1) - max_train_length + 1)
                    x = x[:, window_offset: window_offset + max_train_length]

                x = x.to(device)

                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (temporal_unit + 1), high=ts_l + 1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))

                out1 = model(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft), use_mask=True)
                out1 = out1[:, -crop_l:]

                out2 = model(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left), use_mask=True)
                out2 = out2[:, :crop_l]

                training_batch_loss = ts2vec_loss_fn(z1=out1, z2=out2, temporal_unit=temporal_unit)

                training_batch_loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_t2v.step()
                # avg_model.update_parameters(model)

                if lr_warmup:
                    scheduler.step()

                training_batch_losses.append(training_batch_loss.item())
                t.cuda.empty_cache()

                if verbose:
                    if i + 1 == len(dataLoader) or (i + 1) / (progressed * update_interval) > 1:
                        pbar.update(n=round(progressed * update_interval, 2) - pbar.n)
                        progressed += 1
            if verbose:
                pbar.close()

            training_loss = np.array(training_batch_losses).mean()

            if verbose:
                print("Training_loss of epoch", epoch_i, "is:", training_loss)

            # Validation
            with t.no_grad():
                # avg_model.eval()
                model.eval()
                _, labels_test, _ = map(to_tensor,
                                        sampler.test_forecasting_windows(),
                                        (device,) * 3)

                # Normalize labels windows
                y_test = (labels_test - to_tensor(sampler.train_ts_means,  # (n_test_windows, horizon, N)
                                                  device=device)) / to_tensor(
                    sampler.train_ts_std, device=device)
                y_test = y_test.reshape(y_test.shape[0], y_test.shape[2], y_test.shape[1],
                                        1)  # y_test of shape (n_test_windows, n_instance, n_timestamps, n_features)

                test_loss = 0
                for x in y_test:
                    # out1_test, out2_test = avg_model(x, use_mask=True), avg_model(x, use_mask=True)
                    out1_test, out2_test = model(x, use_mask=True), model(x, use_mask=True)
                    test_loss += float(ts2vec_loss_fn(z1=out1_test, z2=out2_test))
                test_loss /= y_test.shape[0]

                if verbose:
                    print("Test loss of epoch", epoch_i, "is:", test_loss)

                # Register model if any of the metrics have changed
                if test_loss < min_loss_test:
                    min_loss_test = test_loss
                    optimized = True

                snapshot_manager_t2v.register(epoch=epoch_i, training_loss=training_loss,
                                              testing_loss=test_loss,
                                              save_model=optimized,
                                              # model=avg_model, optimizer=optimizer_t2v)
                                              model=model, optimizer=optimizer_t2v)

                t.cuda.empty_cache()

                if early_stopping:
                    if optimized:
                        num_no_improve = 0
                    else:
                        num_no_improve += 1

                    if num_no_improve >= patience:
                        break

                snapshot_manager_t2v.print_losses(top_best=n_best_test_losses)


def ae_temp_trainer_pbbilist(snapshot_manager: SnapshotManager,
                              model: t.nn.Module,
                              dataLoader: DataLoader,
                              horizon: int,
                              test_loss_name: str,
                              train_ae_loss: str,
                              train_forecasting_loss: str,
                              train_temp_loss: str,
                              lambda_ae: float = 1,
                              lambda_f: float = 1,
                              lambda_temp: float = 1,
                              lambda_NC: float = 5e-3,
                              epochs: int = 750,
                              learning_rate: float = 0.001,
                              verbose: bool = True,
                              pbar_percentage: int = 20,
                              early_stopping: bool = False,
                              patience: int = 5,
                              device_id: int = None,
                              n_best_test_losses: int = None,
                              lr_warmup: int = None,
                              mv_training: bool = True,
                              pretrain_epochs: int = 0,
                              num_samples: int = 1000,
                              ):
    device = default_device(device_str_id=device_id)

    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    forecasting_loss_fn = __loss_fn(train_forecasting_loss)
    ae_loss_fn = __loss_fn(train_ae_loss)
    temp_loss_fn = __loss_fn(train_temp_loss)
    test_forecasting_loss_fn = __loss_fn(test_loss_name)

    epoch = snapshot_manager.restore(model, optimizer)

    model = model.to(device)
    snapshot_manager.enable_time_tracking()

    if lr_warmup:
        scheduler = t.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda j: min(j / (lr_warmup / dataLoader.dataset.train_window), 1.0))

    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss_test = np.inf
        num_no_improve = 0

        ae_pretraining = True
        f_model_pretraining = False

        for epoch_i in range(epoch + 1, epochs + 1):
            if epoch_i > pretrain_epochs // 2:
                if ae_pretraining:
                    ae_pretraining = False
                    print('AE Pretraining is finished')
                if not f_model_pretraining and epoch_i <= pretrain_epochs:
                    f_model_pretraining = True
                elif f_model_pretraining and epoch_i > pretrain_epochs:
                    f_model_pretraining = False
                    print('f_model Pretraining is finished')

            optimized = False

            # Training
            training_batch_losses = []

            # Go through all training batches
            if verbose:
                pbar = tqdm(total=len(dataLoader), desc='epoch ' + str(epoch_i) + ' - train')
                update_interval, progressed = len(dataLoader) * pbar_percentage / 100, 1

            model.train()

            for i, y in enumerate(dataLoader):
                optimizer.zero_grad()
                y = y.to(device)

                if len(y.shape) == 4:
                    y = y.view(y.shape[0] * y.shape[1], y.shape[2], y.shape[3])

                if ae_pretraining:
                    y_hat, x_v1, x_v2, _, _ = model(y, use_f_m=False)
                    ae_loss = ae_loss_fn(prediction=y_hat, target=y)
                    f_loss = 0
                    temp_loss = temp_loss_fn(z1=x_v1, z2=x_v2, lambda_NC=lambda_NC) if lambda_temp > 0 else 0
                elif f_model_pretraining:
                    _, _, _, x_f_labels, x_f = model(y, use_f_m=True)
                    ae_loss = 0
                    f_loss = forecasting_loss_fn(prediction=x_f, target=x_f_labels)
                    temp_loss = 0
                else:
                    y_hat, x_v1, x_v2, x_f_labels, x_f = model(y, use_f_m=True)
                    ae_loss = ae_loss_fn(prediction=y_hat, target=y)
                    f_loss = forecasting_loss_fn(prediction=x_f, target=x_f_labels)
                    temp_loss = temp_loss_fn(z1=x_v1, z2=x_v2, lambda_NC=lambda_NC) if lambda_temp > 0 else 0

                training_batch_loss = lambda_ae * ae_loss + lambda_f * f_loss + lambda_temp * temp_loss

                training_batch_loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if lr_warmup:
                    scheduler.step()
                training_batch_losses.append(training_batch_loss.item())
                t.cuda.empty_cache()

                if verbose:
                    if i + 1 == len(dataLoader) or (i + 1) / (progressed * update_interval) > 1:
                        pbar.update(n=round(progressed * update_interval, 2) - pbar.n)
                        progressed += 1
            if verbose:
                pbar.close()

            training_loss = np.array(training_batch_losses).mean()

            if verbose:
                print("Training_loss of epoch", epoch_i, "is:", training_loss)

            # Validation
            with t.no_grad():
                model.eval()

                input_windows_test, labels_test, input_windows_val = map(to_tensor,
                                                                         dataLoader.dataset.test_forecasting_windows(),
                                                                         (device,) * 3)

                # Normalize input windows
                input_windows_test_normalized = (input_windows_test - to_tensor(dataLoader.dataset.train_ts_means,
                                                                                device=device)) / to_tensor(
                    dataLoader.dataset.train_ts_std, device=device)

                # Forecast future values
                _, y_forecast_mu_norm = model.rolling_forecast(y=input_windows_test_normalized,
                                                                                     horizon=horizon,
                                                                                     num_samples=0,
                                                                                     sigma=1)

                # Rescale forecasts back to original scale
                y_forecast_mu = y_forecast_mu_norm * to_tensor(dataLoader.dataset.train_ts_std,
                                                               device=device) + to_tensor(
                    dataLoader.dataset.train_ts_means, device=device)

                test_loss = float(test_forecasting_loss_fn(prediction=y_forecast_mu, target=labels_test))

                if verbose:
                    print("Test loss of epoch", epoch_i, "is:", test_loss)

                # Register model if any of the metrics have changed
                if test_loss < min_loss_test:
                    min_loss_test = test_loss
                    optimized = True

                snapshot_manager.register(epoch=epoch_i, training_loss=training_loss,
                                          testing_loss=test_loss,
                                          save_model=optimized,
                                          model=model, optimizer=optimizer)

                t.cuda.empty_cache()

                if early_stopping:
                    if optimized:
                        num_no_improve = 0
                    else:
                        num_no_improve += 1

                    if num_no_improve >= patience:
                        break

                snapshot_manager.print_losses(top_best=n_best_test_losses)

        # Reload the best model
        _ = snapshot_manager.restore(model, optimizer)
        with t.no_grad():
            model.eval()
            # End of the programme, compute CRPS
            input_windows_test, labels_test, input_windows_val = map(to_tensor,
                                                                     dataLoader.dataset.test_forecasting_windows(),
                                                                     (device,) * 3)

            # Normalize input windows
            input_windows_test_normalized = (input_windows_test - to_tensor(dataLoader.dataset.train_ts_means,
                                                                            device=device)) / to_tensor(
                dataLoader.dataset.train_ts_std, device=device)

            # Forecast future values
            y_forecast_samples_norm, y_forecast_mu_norm = model.rolling_forecast(y=input_windows_test_normalized,
                                                                                 horizon=horizon,
                                                                                 num_samples=num_samples,
                                                                                 sigma=1)

            y_forecast_samples = y_forecast_samples_norm * to_tensor(dataLoader.dataset.train_ts_std,
                                                                     device=device) + to_tensor(
                dataLoader.dataset.train_ts_means, device=device)
            y_forecast_mu = y_forecast_mu_norm * to_tensor(dataLoader.dataset.train_ts_std,
                                                           device=device) + to_tensor(
                dataLoader.dataset.train_ts_means, device=device)

            test_loss = float(test_forecasting_loss_fn(prediction=y_forecast_mu, target=labels_test))
            crps, crps_sum = CRPS.calculate_crps(y_forecast_samples.detach().cpu().numpy(), labels_test.detach().cpu().numpy())

            print('\nCRPS:', crps)
            print('CRPS-Sum:', crps_sum)
            print('MSE:', test_loss)