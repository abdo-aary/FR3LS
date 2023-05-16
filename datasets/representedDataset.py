from abc import ABC
import numpy as np


class RepresentedDataset(ABC):
    ids: np.ndarray  # shape: (N, ) mapping from index to TS ids (stock_symbol for finance data)
    ts_samples: np.ndarray  # shape: (T, N)  TS' values
    timestamps: np.ndarray  # shape: (T, )  TS' dates

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    @staticmethod
    def load(ts_dataset_name: str) -> 'RepresentedDataset':
        raise NotImplementedError

    @staticmethod
    def download(ts_dataset_name: str, delete_cache: bool = True) -> None:
        raise NotImplementedError

    def sample(self, idx):
        raise NotImplementedError
