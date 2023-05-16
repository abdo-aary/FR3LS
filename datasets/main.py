"""
Datasets module
"""
import logging
import gin

from fire import Fire

from datasets.financeDataset import FinanceDataset


def build_dataset(config_path: str) -> None:
    gin.parse_config_file(config_path)
    logging.info('Downloading Dataset ...')
    FinanceDataset.download()


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire(build_dataset)
