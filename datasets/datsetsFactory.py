from datasets.financeDataset import FinanceDataset
from datasets.electrecityDataset import ElectricityDataset
from datasets.representedDataset import RepresentedDataset


class DatasetsFactory:
    """
    Factory that allow loading any model implemented
    """

    def __init__(self):
        pass

    @staticmethod
    def get_dataset(dataset_type: str, **kwargs) -> 'RepresentedDataset':
        if dataset_type == 'FinanceDataset':
            representedDataset = FinanceDataset.load(**kwargs)
        elif dataset_type == 'ElectricityDataset':
            representedDataset = ElectricityDataset.load(**kwargs)
        else:
            raise Exception(f'Unknown dataset type {dataset_type}')
        return representedDataset
