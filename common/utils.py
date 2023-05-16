import os.path
import re
import torch as t

import numpy as np
from typing import List, Optional


def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]


def atoi(text):
    return int(text) if text.isdigit() else text


def split_array_by_chunk(x: np.ndarray, chunk_size: int, step: Optional[int] = None, expanding: bool = False) -> List[
    np.ndarray]:
    step = 1 if step is None else step
    if expanding:
        idxs = [(0, i) for i in range(chunk_size, len(x), step)]
    else:
        idxs = [(i - chunk_size, i) for i in range(chunk_size, len(x), step)]
    return [x[idx_start:idx_end] for (idx_start, idx_end) in idxs]


def count_parameters(model: t.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def deco_print(line, end='\n'):
    print('>==================> ' + line, end=end)


def read_config_file(config_file_path: str) -> dict:
    assert os.path.exists(config_file_path), "Path to config file des not exist"
    with open(config_file_path) as f:
        lines = f.read().splitlines()

    config_dict = {}
    for line in lines:
        if len(line) > 0 and '#' == line[0]:
            continue
        partitions = line.partition('.')
        key = partitions[2].partition('=')[0].replace(" ", "")
        value = line.partition('=')[2].split('#')[0]
        value = value.replace(" ", "")
        if key == '':
            continue
        config_dict[key] = value

    return config_dict

def get_nbr_layers_coverage(sequence_length, kernel_size):
    num_layers_coverage = int(np.log2(((sequence_length - 1) * (2 - 1)) / (kernel_size - 1)) + 1)
    return num_layers_coverage

def get_receptive_field(kernel_size, nbr_layers):
    receptive_field = 1 + (kernel_size - 1) * (2 ** nbr_layers)/(2 - 1)
    return receptive_field

