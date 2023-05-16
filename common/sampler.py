import numpy as np
import tqdm
from torch.utils.data.dataset import Dataset

from common.torch.ops import to_tensor


class TimeSeriesSampler(Dataset):
    def __init__(self,
                 timeseries: np.ndarray,
                 train_window: int,
                 batch_size: int,
                 f_input_window: int,
                 horizon: int,
                 non_overlap_batch: int,
                 n_test_windows: int,
                 n_val_windows: int,
                 shuffle: bool = False
                 ):
        """
        Timeseries sampler.

        :param timeseries: of shape (T, N)
        :param train_window: time points per training window
        :param batch_size: number of subseries to sample in one batch
        :param f_input_window: the forecasting input window size (Should be < train_window)
        :param horizon: number of time points to forecast
        :param non_overlap_batch:
        :param n_test_windows: number of windows to use as test windows (last windows of the dataset)
        :param n_val_windows: number of windows to use as validation windows (last windows of the dataset
               before test windows)

        """
        super().__init__()

        self.timeseries = timeseries
        self.train_window = train_window
        self.batch_size = batch_size
        self.f_input_window = f_input_window
        self.horizon = horizon
        self.non_overlap_batch = non_overlap_batch
        self.n_test_windows = n_test_windows
        self.n_val_windows = n_val_windows
        self.shuffle = shuffle

        self.train_end_time_point = self.timeseries.shape[0] - (
                self.n_val_windows + self.n_test_windows) * self.horizon - self.train_window

        self.train_ts_means = self.timeseries[:self.train_end_time_point + self.train_window].mean(axis=0)
        self.train_ts_std = self.timeseries[:self.train_end_time_point + self.train_window].std(axis=0)

        self.train_windows_indices = np.arange(self.train_end_time_point % self.non_overlap_batch,
                                               self.train_end_time_point + 1,
                                               self.non_overlap_batch)

        # self.train_windows_indices contains all start indices of training windows (window = start_idx + train_window)
        self.train_start_window_index = len(self.train_windows_indices) - (
                len(self.train_windows_indices) // self.batch_size) * self.batch_size

        self.train_windows_indices = self.train_windows_indices[self.train_start_window_index:]

        # Shuffle these indices
        if self.shuffle:
            np.random.shuffle(self.train_windows_indices)

        # Divide these indices to batches, this way an epoch will visit all the training data !
        self.train_batches_indices = self.train_windows_indices.reshape((
            len(self.train_windows_indices) // self.batch_size, self.batch_size))

        # self.batches = self.load_batches()
        self.current_batch_index = 0

    def __iter__(self):
        """
        Batches of sampled windows.

        :return: Batches of:

        """
        while True:
            Y_B = np.zeros((self.batch_size, self.train_window, self.timeseries.shape[1]))  # Y_B of shape (b, w, N)
            # selected_train_end_time_point = self.train_end_time_point - self.train_window

            if self.current_batch_index >= len(self.train_batches_indices):
                self.current_batch_index = 0

            # sampled_time_stamps_indices = np.random.randint(selected_train_end_time_point, size=self.batch_size)
            sampled_time_stamps_indices = self.train_batches_indices[self.current_batch_index]

            # Y_B[sampled_time_stamps_indices] = self.timeseries[sampled_index:sampled_index + self.train_window]
            for i, sampled_index in enumerate(sampled_time_stamps_indices):
                subseries_i = self.timeseries[sampled_index:sampled_index + self.train_window]
                Y_B[i] = subseries_i

            # Normalize the training batch
            Y_B = (Y_B - self.train_ts_means) / self.train_ts_std

            yield Y_B  # Y_B of shape (batch_size, N, train_window)

            # Update current batch index
            self.current_batch_index += 1

    def test_forecasting_windows(self):
        T = self.timeseries.shape[0] - self.horizon * self.n_val_windows
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
        T = self.timeseries.shape[0] - self.horizon * self.n_val_windows - (
                self.horizon * (self.n_test_windows - self.n_val_windows))
        input_windows = np.array(
            [self.timeseries[
             T - self.f_input_window - (i * self.horizon): T - (i * self.horizon)] for i in
             range(self.n_val_windows, 0, -1)])  # (n_val_windows, f_input_window, Cin)

        labels = np.array([self.timeseries[
                           T - i * self.horizon: T - (i - 1) * self.horizon]
                           for i in
                           range(self.n_val_windows, 0, -1)])  # (n_val_windows, horizon, Cin)

        return input_windows, labels
