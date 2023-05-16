from typing import Iterator, Union

import numpy as np
import torch as t
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.sampler import TimeSeriesSampler
from common.sampler_traffic import TimeSeriesSampler as TimeSeriesSamplerTraffic
from common.torch.losses import __loss_fn, smape_loss, wape_loss, mape_loss
from common.torch.ops import default_device, to_tensor
from common.torch.snapshots import SnapshotManager


def tlae_trainer(snapshot_manager: SnapshotManager,
                 model: t.nn.Module,
                 dataLoader: DataLoader,
                 horizon: int,
                 train_loss_name: str = 'TLAE',
                 test_loss_name: str = 'MAE',
                 train_ae_loss: str = 'MAE',
                 train_forecasting_loss: str = 'MSE',
                 lambda1: int = 1,
                 lambda2: int = 1,
                 epochs: int = 100,
                 learning_rate: float = 0.001,
                 verbose: bool = True,
                 early_stopping: bool = True,
                 patience: int = 5,
                 pbar_percentage: int = 20,
                 device_id: int = None,
                 n_best_test_losses: int = None,
                 lr_warmup: int = None,
                 ):
    device = default_device(device_str_id=device_id)
    model = model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    training_loss_fn = __loss_fn(train_loss_name)
    test_forecasting_loss_fn = __loss_fn(test_loss_name)

    epoch = snapshot_manager.restore(model, optimizer)
    snapshot_manager.enable_time_tracking()

    if lr_warmup:
        scheduler = t.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda j: min(j / (lr_warmup / dataLoader.dataset.train_window), 1.0))
    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss_test = np.inf
        min_loss_mape = np.inf
        num_no_improve = 0

        for epoch_i in range(epoch + 1, epochs + 1):
            optimized = False

            # Training
            model.train()
            training_batch_losses = []

            # Go through all training batches
            if verbose:
                pbar = tqdm(total=len(dataLoader), desc='epoch ' + str(epoch_i) + ' - train')
                update_interval, progressed = len(dataLoader) * pbar_percentage / 100, 1

            for i, Y in enumerate(dataLoader):
                optimizer.zero_grad()
                Y = Y.to(device)

                if len(Y.shape) == 4:
                    Y = Y.view(Y.shape[0] * Y.shape[1], Y.shape[2], Y.shape[3])

                Y_hat, X_f_labels, X_f_total = model(Y)
                training_batch_loss = training_loss_fn(Y=Y, Y_hat=Y_hat,
                                                       X_f_labels=X_f_labels,
                                                       X_f_total=X_f_total,
                                                       lambda1=lambda1,
                                                       lambda2=lambda2,
                                                       train_ae_loss=train_ae_loss,
                                                       train_forecasting_loss=train_forecasting_loss)
                training_batch_loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if lr_warmup:
                    scheduler.step()
                training_batch_losses.append(training_batch_loss.item())

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
            model.eval()
            input_windows_test, labels_test, input_windows_val = map(to_tensor,
                                                                     dataLoader.dataset.test_forecasting_windows(),
                                                                     (device,) * 3)

            # Normalize input windows
            input_windows_test_normalized = (input_windows_test - to_tensor(dataLoader.dataset.train_ts_means,
                                                                            device=device)) / to_tensor(
                dataLoader.dataset.train_ts_std, device=device)

            # Forecast future values
            y_forecast_normalized = model.rolling_forecast(Y=input_windows_test_normalized,
                                                           horizon=horizon)

            # Rescale forecasts back to original scale
            y_forecast = y_forecast_normalized * to_tensor(dataLoader.dataset.train_ts_std, device=device) + to_tensor(
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

            if early_stopping:
                if optimized:
                    num_no_improve = 0
                else:
                    num_no_improve += 1

                if num_no_improve >= patience:
                    break

            snapshot_manager.print_losses(top_best=n_best_test_losses)


def ts2vec_trainer(snapshot_manager: SnapshotManager,
                   model: t.nn.Module,
                   f_model: t.nn.Module,
                   training_set: Iterator,
                   sampler: Union['TimeSeriesSampler', 'TimeSeriesSamplerTraffic'],
                   loss_name: str,
                   epochs: int = 100,
                   learning_rate: float = 0.001,
                   verbose: bool = True,
                   early_stopping: bool = True,
                   patience: int = 5,
                   ):
    model = model.to(default_device())
    f_model = f_model.to(default_device())
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = __loss_fn(loss_name)

    epoch = snapshot_manager.restore(model, optimizer)
    snapshot_manager.enable_time_tracking()

    with t.autograd.set_detect_anomaly(True):
        # Early stopping
        min_loss = np.inf
        num_no_improve = 0

        f_model.eval()
        # TS2VEC Alone training
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
                Y = to_tensor(next(training_set))
                optimizer.zero_grad()

                X = f_model.encoder(Y)
                R_view1 = model(X, use_mask=True)
                R_view2 = model(X, use_mask=True)
                train_ts2vec_loss = loss_fn(z1=R_view1, z2=R_view2)
                train_ts2vec_loss.backward()
                t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_batch_losses.append(train_ts2vec_loss.item())
            training_loss = np.array(training_batch_losses).mean()
            if verbose:
                print("TS2VEC training_loss of epoch", epoch_i, "is:", training_loss)

            # Validation
            model.eval()
            input_windows_test, labels_test, input_windows_val = map(to_tensor,
                                                                     sampler.test_forecasting_windows())

            # Normalize input windows
            Y_val = (input_windows_val - to_tensor(sampler.train_ts_means)) / to_tensor(
                sampler.train_ts_std)

            X_val = f_model.encoder(Y_val)
            R_view1_val = model(X_val, use_mask=True)
            R_view2_val = model(X_val, use_mask=True)
            test_ts2vec_loss = float(loss_fn(z1=R_view1_val, z2=R_view2_val))
            if verbose:
                print("TS2VEC test loss of epoch", epoch_i, "is:", test_ts2vec_loss)

            if test_ts2vec_loss < min_loss:
                min_loss = test_ts2vec_loss
                optimized = True

            snapshot_manager.register(epoch=epoch_i, training_loss=training_loss,
                                      testing_loss=test_ts2vec_loss,
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
