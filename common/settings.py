"""
Common settings
"""
import os

STORAGE = os.getenv('STORAGE')
STORAGE = 'storage' if STORAGE is None else STORAGE
DATASETS_PATH = os.path.join(STORAGE, 'datasets')
EXPERIMENTS_PATH = os.path.join(STORAGE, 'experiments')
TESTS_STORAGE_PATH = os.path.join(STORAGE, 'test')
RESULTS_PATH = os.path.join(STORAGE, 'results')