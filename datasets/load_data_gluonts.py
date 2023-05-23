import logging
import os
import tarfile
# Need to install pathlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from gluonts.dataset.common import ListDataset, DataEntry, Dataset
from gluonts.dataset.common import load_datasets

from common.settings import DATASETS_PATH

# Set this to where you extracted the datastes...
# Dataset used for Tables 2 and 3 in the paper is downloaded at: 
#  https://github.com/mbohlkeschneider/gluon-ts/tree/mv_release/datasets

default_dataset_path = Path(DATASETS_PATH)


def load_data_gluonts(name='electricity_nips',
                      prediction_length=24,
                      num_test_samples=194,
                      dtype=torch.FloatTensor,
                      verbose=False):
    if name == 'electricity_nips':
        ds = electricity(verbose=verbose)
    elif name == 'solar_nips':
        ds = solar(verbose=verbose)
    elif name == 'taxi_30min':
        ds = taxi_30min(verbose=verbose)
    elif name == 'wiki_nips':
        ds = wiki(verbose=verbose)
    elif name == 'exchange_rate_nips':
        ds = exchange_rate(verbose=verbose)
    elif name == 'traffic_nips':
        ds = traffic(verbose=verbose)

    # train_data is of size (num_timesteps, input_dim)
    train_data = torch.tensor(list(ds.train_ds)[0]['target'].T).type(dtype)
    train_data = train_data[:-1, :]

    test_data_in, test_data_target = [], []
    for i in range(len(list(ds.test_ds))):

        # temp is of size (num_timesteps, input_dim)
        temp = torch.tensor(list(ds.test_ds)[i]['target'].T).type(dtype)

        test_data_target.append(temp[(temp.shape[0] - prediction_length):, :])
        test_data_in.append(
            temp[(temp.shape[0] - prediction_length - num_test_samples): (temp.shape[0] - prediction_length), :])

        if verbose: print(temp.shape[0] - prediction_length)

    if verbose: print(train_data.shape)
    if verbose: print(len(test_data_target), test_data_target[0].shape)
    if verbose: print(len(test_data_in), test_data_in[0].shape)

    test_data_target = torch.stack(test_data_target)
    test_data_in = torch.stack(test_data_in)

    return train_data, test_data_in, test_data_target


def extract_dataset(dataset_name: str, verbose: bool = False):
    dataset_folder = default_dataset_path / dataset_name

    if verbose:
        print(dataset_folder)
    if os.path.exists(dataset_folder):
        logging.info(f"found local file in {dataset_folder}, skip extracting")
        return

    unextracted_ds_path = os.path.join(DATASETS_PATH, "unextracted_ds")
    compressed_data_path = Path(unextracted_ds_path)

    tf = tarfile.open(compressed_data_path / (dataset_name + ".tar.gz"))
    tf.extractall(default_dataset_path)


def pivot_dataset(dataset):
    ds_list = list(dataset)
    return [
        {
            'item': '0',
            'start': ds_list[0]['start'],
            'target': np.vstack([d['target'] for d in ds_list]),
        }
    ]


class MultivariateDatasetInfo(NamedTuple):
    name: str
    train_ds: Dataset
    test_ds: Dataset
    prediction_length: int
    freq: str
    target_dim: int


def make_dataset(
        values: np.ndarray,
        prediction_length: int,
        start: str = "1700-01-01",
        freq: str = "1H",
):
    target_dim = values.shape[0]

    print(
        f"making dataset with {target_dim} dimension and {values.shape[1]} observations."
    )

    start = pd.Timestamp(start, freq)

    train_ds = [
        {'item': '0', 'start': start, 'target': values[:, :-prediction_length]}
    ]
    test_ds = [{'item': '0', 'start': start, 'target': values}]

    return MultivariateDatasetInfo(
        name="custom",
        train_ds=train_ds,
        test_ds=test_ds,
        target_dim=target_dim,
        freq=freq,
        prediction_length=prediction_length,
    )


class Grouper:

    # todo the contract of this grouper is missing from the documentation, what it does when, how it pads values etc
    def __init__(
            self,
            fill_value: float = 0.0,
            max_target_dim: int = None,
            align_data: bool = True,
            num_test_dates: int = None,
    ) -> None:
        self.fill_value = fill_value

        self.first_timestamp = pd.Timestamp(2200, 1, 1, 12)
        self.last_timestamp = pd.Timestamp(1800, 1, 1, 12)
        self.frequency = None
        self.align_data = align_data
        self.max_target_length = 0
        self.num_test_dates = num_test_dates
        self.max_target_dimension = max_target_dim

    def __call__(self, dataset: Dataset) -> Dataset:
        self._preprocess(dataset)
        return self._group_all(dataset)

    def _group_all(self, dataset: Dataset) -> Dataset:

        if self.align_data:
            funcs = {'target': self._align_data_entry}

        if self.num_test_dates is None:
            grouped_dataset = self._prepare_train_data(dataset, funcs)
        else:
            grouped_dataset = self._prepare_test_data(dataset)
        return grouped_dataset

    def to_ts(self, data: DataEntry):
        return pd.Series(
            data['target'],
            index=pd.date_range(
                start=data['start'].to_timestamp(),
                periods=len(data['target']),
                freq=data['start'].freq,
            ),
        )

    def _align_data_entry(self, data: DataEntry) -> DataEntry:
        d = data.copy()
        # fill target invidually if we want to fill all of them, we should use a dataframe
        ts = self.to_ts(data)
        d['target'] = ts.reindex(
            pd.date_range(
                start=self.first_timestamp.to_timestamp(),
                end=self.last_timestamp.to_timestamp(),
                freq=d['start'].freq,
            ),
            fill_value=ts.mean(),
        )
        d['start'] = self.first_timestamp
        return d

    def _preprocess(self, dataset: Dataset) -> None:
        """
        The preprocess function iterates over the dataset to gather data that
        is necessary for grouping.
        This includes:
            1) Storing first/last timestamp in the dataset
            2) Aligning time series
            3) Calculating groups
        """
        try:
            for data in dataset:
                timestamp = data['start']
                self.first_timestamp = min(pd.Period(self.first_timestamp, freq=timestamp.freq), timestamp)
                self.last_timestamp = max(
                    pd.Period(self.last_timestamp, freq=timestamp.freq), timestamp + len(data['target'])
                )
                self.frequency = (
                    timestamp.freq if self.frequency is None else self.frequency
                )
                # todo
                self.max_target_length = max(
                    self.max_target_length, len(data['target'])
                )
        except:
            pass
        logging.info(
            f"first/last timestamp found: {self.first_timestamp}/{self.last_timestamp}"
        )

    def _prepare_train_data(self, dataset, funcs):
        logging.info("group training time-series to datasets")
        grouped_data = {}
        for key in funcs.keys():
            grouped_entry = [funcs[key](data)[key] for data in dataset]

            # we check that each time-series has the same length
            assert (
                    len(set([len(x) for x in grouped_entry])) == 1
            ), f"alignement did not work as expected more than on length found: {set([len(x) for x in grouped_entry])}"
            grouped_data[key] = np.array(grouped_entry)
        if self.max_target_dimension is not None:
            # targets are often sorted by incr amplitude, use the last one when restricted number is asked
            grouped_data['target'] = grouped_data['target'][
                                     -self.max_target_dimension:, :
                                     ]
        grouped_data['item_id'] = "all_items"
        grouped_data['start'] = self.first_timestamp
        grouped_data['feat_static_cat'] = [0]
        return ListDataset(
            [grouped_data], freq=self.frequency, one_dim_target=False
        )

    def _prepare_test_data(self, dataset):
        logging.info("group test time-series to datasets")

        def left_pad_data(data: DataEntry):
            ts = self.to_ts(data)
            filled_ts = ts.reindex(
                pd.date_range(
                    start=self.first_timestamp.to_timestamp(),
                    end=ts.index[-1],
                    freq=data['start'].freq,
                ),
                fill_value=0.0,
            )
            return filled_ts.values

        grouped_entry = [left_pad_data(data) for data in dataset]

        grouped_entry = np.array(grouped_entry, dtype=object)

        split_dataset = np.split(grouped_entry, self.num_test_dates)

        all_entries = list()
        for dataset_at_test_date in split_dataset:
            grouped_data = dict()
            assert (
                    len(set([len(x) for x in dataset_at_test_date])) == 1
            ), "all test time-series should have the same length"
            grouped_data['target'] = np.array(
                list(dataset_at_test_date), dtype=np.float32
            )
            if self.max_target_dimension is not None:
                grouped_data['target'] = grouped_data['target'][
                                         -self.max_target_dimension:, :
                                         ]
            grouped_data['item_id'] = "all_items"
            grouped_data['start'] = self.first_timestamp
            grouped_data['feat_static_cat'] = [0]
            all_entries.append(grouped_data)

        return ListDataset(
            all_entries, freq=self.frequency, one_dim_target=False
        )


# def extract_dataset(dataset_name: str):
#     dataset_folder = default_dataset_path / dataset_name
#
#     print(dataset_folder)
#     if os.path.exists(dataset_folder):
#         logging.info(f"found local file in {dataset_folder}, skip extracting")
#         return
#
#     compressed_data_path = Path("datasets")
#     tf = tarfile.open(compressed_data_path / (dataset_name + ".tar.gz"))
#     tf.extractall(default_dataset_path)


def make_multivariate_dataset(
        dataset_name: str,
        num_test_dates: int,
        prediction_length: int,
        max_target_dim: int = None,
        dataset_benchmark_name: str = None,
        verbose: bool = False,
):
    """
    :param verbose:
    :param dataset_name:
    :param num_test_dates:
    :param prediction_length:
    :param max_target_dim:
    :param dataset_benchmark_name: in case the name is different in the repo and in the benchmark, for instance
    'wiki-rolling' and 'wikipedia'
    :return:
    """

    extract_dataset(dataset_name=dataset_name, verbose=verbose)

    metadata, train_ds, test_ds = load_datasets(
        metadata=default_dataset_path / dataset_name / 'metadata',
        train=default_dataset_path / dataset_name / 'train',
        test=default_dataset_path / dataset_name / 'test',
    )
    if verbose:
        print(metadata)

    dim = len(train_ds) if max_target_dim is None else max_target_dim

    grouper_train = Grouper(max_target_dim=dim)
    grouper_test = Grouper(
        align_data=False, num_test_dates=num_test_dates, max_target_dim=dim
    )
    return MultivariateDatasetInfo(
        dataset_name
        if dataset_benchmark_name is None
        else dataset_benchmark_name,
        grouper_train(train_ds),
        grouper_test(test_ds),
        prediction_length,
        metadata.freq,
        dim,
    )


def electricity(max_target_dim: int = None, verbose: bool = False):
    return make_multivariate_dataset(
        dataset_name="electricity_nips",
        num_test_dates=7,
        prediction_length=24,
        max_target_dim=max_target_dim,
        verbose=verbose,
    )


def solar(max_target_dim: int = None, verbose: bool = False):
    return make_multivariate_dataset(
        dataset_name="solar_nips",
        num_test_dates=7,
        prediction_length=24,
        max_target_dim=max_target_dim,
        verbose=verbose,
    )


def traffic(max_target_dim: int = None, verbose: bool = False):
    return make_multivariate_dataset(
        dataset_name="traffic_nips",
        num_test_dates=7,
        prediction_length=24,
        max_target_dim=max_target_dim,
        verbose=verbose,
    )


def wiki(max_target_dim: int = None, verbose: bool = False):
    return make_multivariate_dataset(
        dataset_name="wiki-rolling_nips",
        dataset_benchmark_name="wikipedia",
        num_test_dates=5,
        prediction_length=30,
        # we dont use 9K timeseries due to OOM issues
        max_target_dim=2000 if max_target_dim is None else max_target_dim,
        verbose=verbose,
    )


def exchange_rate(max_target_dim: int = None, verbose: bool = False):
    return make_multivariate_dataset(
        dataset_name="exchange_rate_nips",
        dataset_benchmark_name="exchange_rate_nips",
        num_test_dates=5,
        prediction_length=30,
        max_target_dim=max_target_dim,
        verbose=verbose,
    )


def taxi_30min(max_target_dim: int = None, verbose: bool = False):
    """
    Taxi dataset limited to the most active area, with lower and upper bound:
        lb = [ 40.71, -74.01]
        ub = [ 40.8 , -73.95]
    :param verbose:
    :param max_target_dim:
    :return:
    """
    return make_multivariate_dataset(
        dataset_name="taxi_30min",
        dataset_benchmark_name="taxi_30min",
        # The dataset corresponds to the taxi dataset used in this reference:
        # https://arxiv.org/abs/1910.03002 but only contains 56 evaluation
        # windows. The last evaluation window was removed because there was an
        # overlap of five time steps in the last and the penultimate
        # evaluation window.
        num_test_dates=56,
        prediction_length=24,
        max_target_dim=max_target_dim,
        verbose=verbose,
    )
