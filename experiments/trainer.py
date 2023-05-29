import numpy as np
import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.sampler import Sampler
from common.torch import CRPS
from common.torch.losses import loss_fn, wape_loss, smape_loss, mape_loss
from common.torch.ops import default_device, to_tensor
from common.torch.snapshots import SnapshotManager

import warnings


def trainer_determinist(snapshot_manager: SnapshotManager,
                        model: t.nn.Module,
                        dataLoader: DataLoader,
                        sampler: Sampler,
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
                        **kwargs,
                        ):
    device = default_device(device_str_id=device_id)
    print("device =", device)

    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    forecasting_loss_fn = loss_fn(train_forecasting_loss)
    ae_loss_fn = loss_fn(train_ae_loss)
    temp_loss_fn = loss_fn(train_temp_loss)
    test_forecasting_loss_fn = loss_fn(test_loss_name)

    model = model.to(device)

    epoch = snapshot_manager.restore(model, optimizer)

    snapshot_manager.enable_time_tracking()

    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer,
                                              lambda j: min(j / (lr_warmup / sampler.train_window),
                                                            1.0)) if lr_warmup else None

    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss_test = np.inf
        min_loss_mape = np.inf
        num_no_improve = 0

        for epoch_i in range(epoch + 1, epochs + 1):

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

                if len(y.shape) == 4:  # Reshape the input subseries to have a shape of 3 dimensions
                    y = y.view(y.shape[0] * y.shape[1], y.shape[2], y.shape[3])

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

                if verbose:  # Used for the tqdm loop
                    if i + 1 == len(dataLoader) or (i + 1) / (progressed * update_interval) > 1:
                        pbar.update(n=round(progressed * update_interval, 2) - pbar.n)
                        progressed += 1
            if verbose:
                pbar.close()

            training_loss = np.array(training_batch_losses).mean()

            if verbose:
                print("Training_loss of epoch", epoch_i, "is:", training_loss)

            # Testing
            with t.no_grad():
                model.eval()

                input_windows_test, labels_test = map(to_tensor,
                                                      sampler.get_test_forecasting_windows(),
                                                      (device,) * 2)

                # Normalize input windows
                input_windows_test_normalized = (input_windows_test - to_tensor(sampler.train_ts_means,
                                                                                device=device)) / to_tensor(
                    sampler.train_ts_std, device=device)

                # Forecast future values
                y_forecast_normalized = model.rolling_forecast(y=input_windows_test_normalized,
                                                               horizon=horizon)

                # Rescale forecasts back to original scale
                y_forecast = y_forecast_normalized * to_tensor(sampler.train_ts_std,
                                                               device=device) + to_tensor(
                    sampler.train_ts_means, device=device)

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


def trainer_probabilist(snapshot_manager: SnapshotManager,
                        model: t.nn.Module,
                        dataLoader: DataLoader,
                        sampler: Sampler,
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
                        num_samples: int = 1000,
                        **kwargs
                        ):
    device = default_device(device_str_id=device_id)

    print("device_id =", device_id)
    print("device =", device)

    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    forecasting_loss_fn = loss_fn(train_forecasting_loss)
    ae_loss_fn = loss_fn(train_ae_loss)
    temp_loss_fn = loss_fn(train_temp_loss)
    test_forecasting_loss_fn = loss_fn(test_loss_name)

    model = model.to(device)

    epoch = snapshot_manager.restore(model, optimizer, device=device)

    snapshot_manager.enable_time_tracking()

    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer,
                                              lambda j: min(j / (lr_warmup / sampler.train_window),
                                                            1.0)) if lr_warmup else None

    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss_test = np.inf
        num_no_improve = 0

        for epoch_i in range(epoch + 1, epochs + 1):

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

            # Testing
            with t.no_grad():
                model.eval()

                input_windows_test, labels_test = map(to_tensor,
                                                      sampler.get_test_forecasting_windows(),
                                                      (device,) * 2)

                # Normalize input windows
                input_windows_test_normalized = (input_windows_test - to_tensor(sampler.train_ts_means,
                                                                                device=device)) / to_tensor(
                    sampler.train_ts_std, device=device)

                # Forecast future values
                _, y_forecast_mu_norm = model.rolling_forecast(y=input_windows_test_normalized,
                                                               horizon=horizon,
                                                               num_samples=0,
                                                               sigma=1)

                # Rescale forecasts back to original scale
                y_forecast_mu = y_forecast_mu_norm * to_tensor(sampler.train_ts_std,
                                                               device=device) + to_tensor(
                    sampler.train_ts_means, device=device)

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
        warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan")

        _ = snapshot_manager.restore(model, optimizer, device=device)
        with t.no_grad():
            model.eval()
            # End of the programme, compute CRPS
            input_windows_test, labels_test = map(to_tensor,
                                                  sampler.get_test_forecasting_windows(),
                                                  (device,) * 2)

            # Normalize input windows
            input_windows_test_normalized = (input_windows_test - to_tensor(sampler.train_ts_means,
                                                                            device=device)) / to_tensor(
                sampler.train_ts_std, device=device)

            # Forecast future values
            y_forecast_samples_norm, y_forecast_mu_norm = model.rolling_forecast(y=input_windows_test_normalized,
                                                                                 horizon=horizon,
                                                                                 num_samples=num_samples,
                                                                                 sigma=1)

            y_forecast_samples = y_forecast_samples_norm * to_tensor(sampler.train_ts_std,
                                                                     device=device) + to_tensor(
                sampler.train_ts_means, device=device)
            y_forecast_mu = y_forecast_mu_norm * to_tensor(sampler.train_ts_std,
                                                           device=device) + to_tensor(
                sampler.train_ts_means, device=device)

            test_loss = float(test_forecasting_loss_fn(prediction=y_forecast_mu, target=labels_test))
            crps, crps_sum = CRPS.calculate_crps(y_forecast_samples.detach().cpu().numpy(),
                                                 labels_test.detach().cpu().numpy())

            print('\nCRPS:', crps)
            print('CRPS-Sum:', crps_sum)
            print('MSE:', test_loss)
