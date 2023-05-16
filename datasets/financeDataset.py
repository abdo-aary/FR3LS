"""
Finance TimeSeries Dataset
"""
import logging
import os
from dataclasses import dataclass
import gin
import numpy as np
import pandas as pd
import tqdm
from datasets.representedDataset import RepresentedDataset
from common.settings import DATASETS_PATH
import yfinance as yf

URL_SP500_CONSTITUENTS = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'


@dataclass()
class FinanceDataset(RepresentedDataset):
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

        ids = data.id.values
        dates = data.date.values
        ts_samples = data.price.values

        common_dates = np.intersect1d(dates[0], dates[1], assume_unique=True)

        for j in tqdm.tqdm(range(2, len(ids))):
            common_dates = np.intersect1d(common_dates, dates[j], assume_unique=True)

        common_dates = np.sort(common_dates)

        ts_samples_final = np.zeros((len(common_dates), len(ids)))
        for j in tqdm.tqdm(range(len(ids))):
            _, _, common_indices_j = np.intersect1d(common_dates, dates[j], assume_unique=True, return_indices=True)
            ts_samples_final[:, j] = ts_samples[j][common_indices_j]  # TS values follow the order of common_dates

        return FinanceDataset(ids=ids,
                              timestamps=common_dates,
                              ts_samples=ts_samples_final,
                              )

    @staticmethod
    @gin.configurable()
    def download(ts_dataset_name: str, id_col: str, start_year: int = 2010, delete_cache: bool = True) -> None:
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

        def build_cache():
            info_all_universe = pd.read_csv(assets_universe_path)  # universe of assets to include
            num_symbol = info_all_universe[id_col].values
            entries = []

            for idx, s in tqdm.tqdm(enumerate(num_symbol)):
                hist = yf.download(s, period='max')
                if len(hist.index.values) != 0 and start_year in hist.index.year:
                    entries.append([s, hist.index.values, hist.Close.values])

            df_processed = pd.DataFrame(entries, columns=['id', 'date', 'price'])
            df_processed.to_parquet(ts_dataset_cache_path)

        assets_universe_path = os.path.join(DATASETS_PATH, ts_dataset_name, 'assets_universe.csv')
        if not os.path.isdir(assets_universe_path):
            prepare_assets_universe(ts_dataset_name)

        build_cache()

    def sample(self, idx):
        return self.ids[idx]


def prepare_assets_universe(ts_dataset_name: str):
    if ts_dataset_name == 'sp500':
        dataset_path = os.path.join(DATASETS_PATH, ts_dataset_name)
        assets_universe_path = os.path.join(dataset_path, 'assets_universe.csv')
        sp500_tickers = pd.read_html(URL_SP500_CONSTITUENTS)
        sp500_symbols = sp500_tickers[0].Symbol
        sp500_symbols.to_csv(assets_universe_path, index=False)
