"""
Experiment class
"""
import logging
import os
import sys
import subprocess
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from itertools import product
from pathlib import Path
from shutil import copy
from typing import List, Optional, Union, Tuple, Any
import gin
from tqdm import tqdm

from common.settings import EXPERIMENTS_PATH

# Return the git revision as a string
from common.utils import split_array_by_chunk


def git_version() -> str:
    """
    multi-platform routine to return the git revision
    code taken from https://stackoverflow.com/a/40170206
    """

    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = {}
        for k in ['SYSTEMROOT', 'PATH']:
            v = os.environ.get(k)
            if v is not None:
                env[k] = v
        # LANGUAGE is used on win32
        env['LANGUAGE'] = 'C'
        env['LANG'] = 'C'
        env['LC_ALL'] = 'C'
        out = subprocess.Popen(cmd, stdout=subprocess.PIPE, env=env).communicate()[0]
        return out

    try:
        out = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        GIT_REVISION = out.strip().decode('ascii')
    except OSError:
        GIT_REVISION = "Unknown"
    return GIT_REVISION


class Experiment(ABC):
    """
    Experiment base class.
    """

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.root = Path(config_path).parent
        self.freeze_when_done = False

        gin.parse_config_file(self.config_path)

    @abstractmethod
    def instance(self) -> None:
        """"
        Instance logic method must be implemented with @gin.configurable()
        """

    def build_ensemble(self) -> None:
        """
        Build ensemble from the given configuration.
        :return:
        """
        if EXPERIMENTS_PATH in str(self.root):
            raise Exception('Cannot build ensemble from ensemble member configuration.')
        self.build()

    def parse_inst_name(self, name: Any, value: Any) -> str:
        if isinstance(value, (tuple, list)):
            s = '%s=%s' % (name, value)
            s = s.replace(" ", "")
            return s
        else:
            return self._parse_inst_name(name, value)

    @staticmethod
    def _parse_inst_name(name: str, value: object) -> str:
        if isinstance(value, float):
            return '%s=%.4g' % (name, value)
        elif isinstance(value, dict):
            s = '%s=%s' % (name, value)
            for w in ["{", "}", "\\", ":", "'"]:
                s = s.replace(w, "")
            s = s.replace(" ", '_')
            return s
        elif isinstance(value, np.datetime64):
            return '%s=%s' % (name, pd.to_datetime(str(value)).strftime('%Y-%m-%d'))
        elif isinstance(value, str):
            return '%s=%s' % (name, value)
        else:
            return '%s=%s' % (name, value)

    @gin.configurable()
    def build(self,
              experiment_name: str,
              ts_dataset_name: str,
              repeats: int,
              losses: Optional[List[str]] = None,
              continue_if_exist=False,
              do_not_create_path=False,
              parameters: Optional[dict] = None):
        # create experiment instance(s)
        logging.info('Creating experiment instances ...')
        parameters = dict() if parameters is None else parameters
        experiment_path = os.path.join(EXPERIMENTS_PATH, experiment_name)
        experiment_path = os.path.join(experiment_path, ts_dataset_name)
        default_variable_names = ['repeat', 'loss'] if losses is not None else ['repeat']
        variable_names = default_variable_names + list(parameters.keys())
        dataset_info_variable_names = ['ts_dataset_name']
        dataset_info_variables = [ts_dataset_name]
        if losses is not None:
            ensemble_variables = [list(range(repeats)), losses] + [parameters[k] for k in
                                                                   variable_names[len(default_variable_names):]]
        else:
            ensemble_variables = [list(range(repeats))] + [parameters[k] for k in
                                                           variable_names[len(default_variable_names):]]
        instance_to_run = []
        for instance_values in tqdm(product(*ensemble_variables)):
            instance_dataset_info = dict(zip(dataset_info_variable_names, dataset_info_variables))
            instance_variables = dict(zip(variable_names, instance_values))
            instance_name = ','.join([self.parse_inst_name(name, value) for name, value in instance_variables.items()])
            instance_path = os.path.join(experiment_path, instance_name)
            for k in list(instance_variables.keys()):
                if '.' not in k:
                    instance_variables[f'instance.{k}'] = instance_variables.pop(k)
            for k in list(instance_dataset_info.keys()):
                if '.' not in k:
                    instance_variables[f'instance.{k}'] = instance_dataset_info.pop(k)
            if Path(instance_path).exists() and continue_if_exist:
                pass
            else:
                if do_not_create_path:
                    continue
                Path(instance_path).mkdir(parents=True, exist_ok=False)

            # write parameters
            instance_config_path = os.path.join(instance_path, 'config.gin')
            copy(self.config_path, instance_config_path)
            with open(instance_config_path, 'a') as cfg:
                for name, value in instance_variables.items():
                    value = f"'{value}'" if isinstance(value, str) else str(value)
                    cfg.write(f'{name} = {value}\n')

            # write command file
            command_file = os.path.join(instance_path, 'command')
            with open(command_file, 'w') as cmd:
                cmd.write(f'python {sys.modules["__main__"].__file__} '
                          f'--config_path={instance_config_path} '
                          f'run >> {instance_path}/instance.log 2>&1')

            # git_info = os.path.join(instance_path, '_GIT')
            # with open(git_info, 'w') as git:
            #     git.write(f"{git_version()}")

            instance_to_run.append(instance_path)

        return instance_to_run

    def run(self):
        """
        Run instance logic.
        """
        success_flag = os.path.join(self.root, '_SUCCESS')
        if os.path.isfile(success_flag):
            return

        self.instance()

        # mark experiment as finished.
        Path(success_flag).touch()

        if self.freeze_when_done:
            # make experiment directory and its content read-only.
            for root, dirs, files in os.walk(self.root):
                os.chmod(root, 0o555)
                for directory in dirs:
                    os.chmod(os.path.join(root, directory), 0o555)
                for file in files:
                    os.chmod(os.path.join(root, file), 0o444)
