"""
Electricity TimeSeries Dataset
"""
import logging
import os
from dataclasses import dataclass

import gin
import numpy as np
import pandas as pd
import tqdm

from common.settings import DATASETS_PATH
from datasets.representedDataset import RepresentedDataset

import urllib.request
from zipfile import ZipFile

URL_ELECTRICITY_DATASET_FILE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00321/LD2011_2014.txt.zip'


@dataclass()
class ElectricityDataset(RepresentedDataset):
    ids: np.ndarray  # shape: (N, ) mapping from index to TS ids (stock_symbol for finance data)
    ts_samples: np.ndarray  # shape: (T, N)  TS' values
    timestamps: np.ndarray  # shape: (T, )  TS' timestamps

    def __init__(self, ids: np.ndarray,
                 ts_samples: np.ndarray,
                 timestamps: np.ndarray,
                 ):
        self.ids = ids
        self.ts_samples = ts_samples
        self.timestamps = timestamps

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for idx, _ in enumerate(self.ids):
            yield self.sample(idx)

    @staticmethod
    @gin.configurable()
    def load(ts_dataset_name: str) -> 'RepresentedDataset':
        """
        Load cached dataset.

        :param ts_dataset_name: The name of the dataset to load
        :return: A RepresentedDataset object
        """
        dataset_path = os.path.join(DATASETS_PATH, ts_dataset_name)
        ts_dataset_cache_path = os.path.join(dataset_path, ts_dataset_name + '.gzip')

        data = pd.read_parquet(ts_dataset_cache_path)

        timestamps = data.index.values
        ids = data.columns.values
        ts_samples = data.to_numpy()  # (T, N)

        return ElectricityDataset(ids=ids,
                                  timestamps=timestamps,
                                  ts_samples=ts_samples,
                                  )

    @staticmethod
    @gin.configurable()
    def download(ts_dataset_name: str, delete_cache: bool = False) -> None:
        """
        Download dataset if doesn't exist.
        """

        dataset_path = os.path.join(DATASETS_PATH, ts_dataset_name)
        os.makedirs(dataset_path, exist_ok=True)

        ts_dataset_cache_path = os.path.join(dataset_path, ts_dataset_name + '.gzip')
        if os.path.isdir(ts_dataset_cache_path):
            logging.info(f'{ts_dataset_cache_path} file already exists.')
            if delete_cache:
                logging.warning(f'skip: deleting previous cache; rebuilding new one')
                os.remove(ts_dataset_cache_path)
            else:
                return

        data_zip_path = os.path.join(dataset_path, ts_dataset_name + '.zip')
        os.makedirs(data_zip_path, exist_ok=True)
        if not os.path.exists(data_zip_path):
            urllib.request.urlretrieve(URL_ELECTRICITY_DATASET_FILE, data_zip_path)
        data_unzipped_path = os.path.join(dataset_path, ts_dataset_name + '_unzipped')
        if not os.path.isdir(data_unzipped_path):
            ZipFile(data_zip_path).extractall(data_unzipped_path)
        data_txt_path = os.path.join(data_unzipped_path, get_element_by_extension(path=data_unzipped_path, ext='.txt'))

        # Data preprocessing
        data = pd.read_csv(data_txt_path, delimiter=';')
        data = data.rename(columns={data.columns[0]: 'timestamp'})
        data['timestamp'] = data.to_datetime(data['timestamp'])
        data = data.set_index('timestamp')
        for i, column in tqdm.tqdm(enumerate(data.columns), desc='preprocessing the data'):
            data[column] = data[column].apply(lambda x: x.replace(',', '.') if type(x) == str else x)
            data[column] = pd.to_numeric(data[column])

        df_processed = data.resample('H').sum()
        # df_processed.shape = (T, N)  1 for the date columns
        df_processed.to_parquet(ts_dataset_cache_path)

    def sample(self, idx):
        return self.ids[idx]


def get_element_by_extension(path: str, ext: str = '.txt'):
    for element in os.listdir(path):
        if os.path.splitext(element)[-1] == ext:
            file_name = element
            return file_name
    return None
