import numpy as np
import torch as t
from torch.utils.data import IterableDataset, get_worker_info


class DeterministicSampler(IterableDataset):
    def __init__(self,
                 timeseries: np.ndarray,
                 train_window: int,
                 f_input_window: int,
                 horizon: int,
                 non_overlap_batch: int,
                 n_test_windows: int,
                 n_val_windows: int,
                 skip_end_n_val=False,
                 shuffle: bool = False,
                 ):
        """
        Timeseries sampler.

        :param timeseries: numpy array of shape (T, N)
        :param train_window: time points per training window
        :param f_input_window: the forecasting input window size (Should be < train_window)
        :param horizon: number of time points to forecast
        :param non_overlap_batch:
        :param n_test_windows: number of windows to use as test windows (last windows of the dataset)
        :param n_val_windows: number of windows to use as validation windows (last windows of the dataset
               before test windows)

        """
        super().__init__()

        # The batching of the samples is controlled by the DataLoader
        # To get more batches one could lower the non_overlap_batch param

        self.timeseries = timeseries
        self.train_window = train_window
        self.f_input_window = f_input_window
        self.horizon = horizon
        self.non_overlap_batch = non_overlap_batch
        self.n_test_windows = n_test_windows
        self.n_val_windows = n_val_windows
        self.skip_end_n_val = skip_end_n_val
        self.shuffle = shuffle

        self.train_end_time_point = self.timeseries.shape[0] - (
                self.n_val_windows + self.n_test_windows) * self.horizon - self.train_window

        self.train_ts_means = np.nanmean(self.timeseries[:self.train_end_time_point + self.train_window], axis=0)
        self.train_ts_std = np.nanstd(self.timeseries[:self.train_end_time_point + self.train_window], axis=0)

        self.train_batches_indices = np.arange(self.train_end_time_point % self.non_overlap_batch,
                                               self.train_end_time_point + 1,
                                               self.non_overlap_batch)

        # Shuffle these indices
        if self.shuffle:
            np.random.shuffle(self.train_batches_indices)

    def __iter__(self):
        """
        Batches of sampled windows.

        :return: Batches of:

        """
        for sampled_index in self.train_batches_indices:
            y = self.timeseries[sampled_index:sampled_index + self.train_window]
            # Normalize the training subseries
            y = (y - self.train_ts_means) / self.train_ts_std
            y = y.reshape(1, y.shape[0], y.shape[1])
            yield y  # y of shape (1, w, N)

    def __len__(self):
        return len(self.train_batches_indices)

    def get_train_data(self):
        train_data = self.timeseries[:self.train_end_time_point + self.train_window]
        return train_data

    def test_forecasting_windows(self):
        T = self.timeseries.shape[0] - self.skip_end_n_val * (self.horizon * self.n_val_windows)
        input_windows_test = np.array(
            [self.timeseries[T - self.f_input_window - (i * self.horizon): T - (i * self.horizon)] for i in
             range(self.n_test_windows, 0, -1)])  # (n_test_windows, f_input_window, Cin)

        input_windows_val = np.array(
            [self.timeseries[T - self.train_window - (i * self.horizon): T - (i * self.horizon)] for i in
             range(self.n_test_windows, 0, -1)])  # (n_test_windows, f_train_window, Cin)

        labels_test = np.array([self.timeseries[T - i * self.horizon: T - (i - 1) * self.horizon] for i in
                                range(self.n_test_windows, 0, -1)])  # (n_test_windows, horizon, Cin)

        return input_windows_test, labels_test, input_windows_val

    def val_forecasting_windows(self):
        T = self.timeseries.shape[0] - self.horizon * self.n_test_windows
        input_windows = np.array(
            [self.timeseries[
             T - self.f_input_window - (i * self.horizon): T - (i * self.horizon)] for i in
             range(self.n_val_windows, 0, -1)])  # (n_val_windows, f_input_window, Cin)

        labels = np.array([self.timeseries[
                           T - i * self.horizon: T - (i - 1) * self.horizon]
                           for i in
                           range(self.n_val_windows, 0, -1)])  # (n_val_windows, horizon, Cin)

        return input_windows, labels


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    num_workers = worker_info.num_workers

    # set different random seed for each worker
    t.manual_seed(worker_id)

    # divide the indices array into chunks based on number of workers
    chunk_size = len(dataset.train_batches_indices) // num_workers

    chunk_start = worker_id * chunk_size
    chunk_end = len(dataset.train_batches_indices) if worker_id == (num_workers - 1) else (chunk_start + chunk_size)
    dataset.train_batches_indices = dataset.train_batches_indices[chunk_start:chunk_end]