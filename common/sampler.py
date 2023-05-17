import numpy as np
import torch as t
from torch.utils.data import IterableDataset, get_worker_info


class Sampler(IterableDataset):
    def __init__(self, train_window: int,
                 f_input_window: int,
                 horizon: int,
                 non_overlap_batch: int,
                 n_test_windows: int,
                 n_val_windows: int,
                 timeseries: np.ndarray = None,
                 train_data: t.Tensor = None,
                 test_data_in: t.Tensor = None,
                 test_data_target: t.Tensor = None,
                 skip_end_n_val: bool = False,
                 shuffle: bool = False, ):
        """
        Timeseries sampler.

        :param train_window: time points per training window (= w in the paper)
        :param f_input_window: the forecasting input window size (= L in the paper, and Should be < train_window)
        :param horizon: number of time points to forecast (= tau in the paper)
        :param non_overlap_batch:
        :param skip_end_n_val: used to match TLAE's test window preparation for electricity-large (deterministic)data
        :param n_test_windows: number of windows to use as test windows, (= k in the paper, and represents
               the last windows of the dataset)
        :param n_val_windows: number of windows to use as validation windows (last windows of the dataset
               before test windows). This is not really used, but kept to match results of TLAE paper
        :param timeseries: when provided, it's a numpy array of shape (T + (k + n_val) * tau, N), with N = input_dim,
                k = num_test_windows, and n_val = num_val_windows
        :param train_data: when provided, it's a numpy array of shape (T, N)
        :param test_data_in: when provided, it's a numpy array of shape (k * L, N)
        :param test_data_target: when provided, it's a numpy array of shape (k * tau, N)

        """
        super().__init__()

        # The batching of the samples is controlled by the DataLoader
        # To get more batches one could lower the non_overlap_batch param

        self.timeseries = timeseries
        self.train_data = train_data
        self.test_data_in = test_data_in
        self.test_data_target = test_data_target

        self.train_window = train_window
        self.f_input_window = f_input_window
        self.horizon = horizon
        self.non_overlap_batch = non_overlap_batch
        self.n_test_windows = n_test_windows
        self.n_val_windows = n_val_windows
        self.skip_end_n_val = skip_end_n_val
        self.shuffle = shuffle

        if self.train_data is None:
            self.train_end_time_point = self.timeseries.shape[0] - (
                    self.n_val_windows + self.n_test_windows) * self.horizon - self.train_window

            self.train_data = self.get_train_data()
        else:
            self.train_end_time_point = self.train_data.shape[0] - self.train_window

        self.train_ts_means = np.nanmean(self.train_data, axis=0)
        self.train_ts_std = np.nanstd(self.train_data, axis=0)

        self.train_batches_indices = np.arange(self.train_end_time_point % self.non_overlap_batch,
                                               self.train_end_time_point + 1, self.non_overlap_batch)

        # Shuffle these indices
        if self.shuffle:
            np.random.shuffle(self.train_batches_indices)

    def __iter__(self):
        """
        Training window sampling

        :return: y: of shape (1, w, N)

        """
        for sampled_index in self.train_batches_indices:
            y = self.train_data[sampled_index:sampled_index + self.train_window]

            # Normalize the training subseries
            y = (y - self.train_ts_means) / self.train_ts_std
            y = y.reshape(1, y.shape[0], y.shape[1])
            yield y

    def __len__(self):
        return len(self.train_batches_indices)

    def get_train_data(self):
        if self.train_data is not None:
            return self.train_data

        train_data = self.timeseries[
                     :self.train_end_time_point + self.train_window]
        return train_data

    def get_test_forecasting_windows(self):
        if self.test_data_in is not None:
            return self.test_data_in, self.test_data_target

        T = self.timeseries.shape[0] - self.skip_end_n_val * (self.horizon * self.n_val_windows)
        input_windows_test = np.array(
            [self.timeseries[T - self.f_input_window - (i * self.horizon): T - (i * self.horizon)] for i in
             range(self.n_test_windows, 0, -1)])  # (n_test_windows, f_input_window, N)

        labels_test = np.array([self.timeseries[T - i * self.horizon: T - (i - 1) * self.horizon] for i in
                                range(self.n_test_windows, 0, -1)])  # (n_test_windows, horizon, N)

        return input_windows_test, labels_test


def worker_init_fn(worker_id):
    """
    defines the logic of the worker_init function

    :param worker_id:
    :return:
    """
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
