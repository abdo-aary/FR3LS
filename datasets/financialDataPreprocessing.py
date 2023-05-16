import os

import numpy as np
import pandas as pd

from common.settings import DATASETS_PATH


def prepare_csv_data(ts_dataset_name: str, file_name: str = None):
    dataset_path = os.path.join(DATASETS_PATH, ts_dataset_name, file_name) if file_name else os.path.join(DATASETS_PATH,
                                                                                                  ts_dataset_name,
                                                                                                  ts_dataset_name + '.csv')
    out_path = os.path.join(DATASETS_PATH, ts_dataset_name, ts_dataset_name + '.npy')
    df = pd.read_csv(filepath_or_buffer=dataset_path, low_memory=False)
    data = df.to_numpy()
    data = data[1:,1:].transpose().astype(float)
    np.save(file=out_path, arr=data, allow_pickle=True)