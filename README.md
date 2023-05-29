# FR3LS

This repository contains the official implementation for the paper _**FR3LS: a Forecasting model with Robust and
Reduced Redundancy Latent Series**_.

## Requirements

The recommended requirements for FR3LS are specified as follows:

- _**python**_ 3.10.4
- gin-config==0.5.0
- numpy==1.23.5
- torch==2.0.1
- typing==3.7.4.3
- tqdm==4.65.0
- pandas==1.5.3
- gluonts==0.12.8
- matplotlib==3.5.3
- fire==0.4.0
- orjson==3.8.12

The dependencies can be installed by:
   ```shell script
   pip install -r requirements.txt
   ```
We recommend working with a virtual environment with the specified python version to avoid dependencies problems. <br/>

## Steps to run a new experiment
Steps to run the main project file `experiments/fr3ls_instance.py` are summarised as follows:
1. Set up requirements 
    ```shell script
    pip install -r requirements.txt
    ```
2. Download datasets
    ```shell script
    bash datasets/download-data-exps.sh
    ```
3. Build experiment
    ```shell script
    python experiments/fr3ls_instance.py build_ensemble --config_path=experiments/configs/type_model/dataset_name.gin
    ```
4. Run experiment
    ```shell script
    bash -c "`cat storage/experiments/fr3ls_type_exps/dataset_name/experiment_name/command`"
    ```

Further details about the project and experiments' set-up are provided bellow.

## Project Structure
### Structure Overview
```
FR3LS
├── common/
├── datasets/
│   ├── download-data-exp.sh
│   └── load_data_gluonts.py
├── experiments/
│   └── configs/
│       ├── determinist/
│       │   └── dataset_name.gin
│       ├── probabilist/ # same as det
│       ├── fr3ls_instance.py
│       └── trainer.py
├── models/
├── notebooks/
├── summaries/
└── storage/
    ├── datasets/
    │   ├── unextracted_ds/
    │   │   └── dataset_name.zip
    │   └── dataset_name/
    └── experiments/
        ├── fr3ls_det_exps/
        │   └── dataset_name/
        │       └── exp_name/
        │           ├── snapshots/
        │           │   ├── epoch
        │           │   ├── losses
        │           │   ├── losses.csv
        │           │   ├── model  # contains trained model weights
        │           │   ├── optimizer
        │           │   └── time
        │           ├── _SUCCESS
        │           ├── command
        │           ├── config.gin
        │           └── instance.log
        └── fr3ls_prob_exps # Same structure as det_exps
```

### Packages description
#### `common`
- Package containing utile python files

#### `datasets`
- Package containing a bash script for downloading necessary datasets + already conducted experiments.  

#### `experiments`
- Package containing main file to be run `fr3ls_instance.py` along with experiments config files stored under `experiments/configs`.<br/>
- We are using the gin lightweight configuration framework (gin-config) allowing default parameter values to be supplied from a config file. For more information please visit the package's official GitHub repository https://github.com/google/gin-config.

#### `models`
- Package containing lstm forecasting model in `models/f_model/lstm.py` as well as main FR3LS determinist and probabilist implementations under `models/fr3ls`.

#### `notebooks`
- Summary of already conducted can be found in `notebooks/fr3ls_exps_summary.ipynb` by running the cells 2 (for determinist experiments, i.e., results reported in table III) and 3 (for probabilist experiments, i.e., results reported in table IV).
- Make sure to run this notebook by a kernel that uses packages from the virtual environment.

#### `summaries`
- Package containing summary classes for both probabilist & deterministic models.

#### `storage` (not tracked by Git)
- Directory where datasets as well as experiments to run are stored.

## Setting up Data & Conducted Experiments
- All the used datasets as well as experiments can be downloaded by running the script `bash datasets/download-data-exps.sh`. <br/>
- Deterministic datasets will be stored under `storage/datasets`, whereas probabilistic unextracted datasets will be stored at `storage/datasets/unextracted_ds` .<br/>
- Already conducted (finished) experiments will be downloaded into `storage/experiments` directory.

## Launching a new experiment

To run the project, follow the following steps:


1. Build an experiment
   ```shell script
   python experiments/fr3ls_instance.py build_ensemble --config_path=experiments/configs/type_model/dataset_name.gin
   ```
   
2. Run a specific experiment
   ```shell script
   bash -c "`cat storage/experiments/fr3ls_type_exps/dataset_name/experiment_name/command`"
   ```
   
3. Parallel automate running of all built sub-experiments of a specific model_type can be done as:
   ```shell script
   for instance in `/bin/ls -d storage/experiments/fr3ls_type_exps/*/*`; do 
       echo $instance
       bash -c "`cat ${instance}/command` &"  # character '&' is added here to run the experiment in the background
   done
   ```

4. Similarly Parallel automate running of all built experiments at once can be done as:   
   ```shell script
   for instance in `/bin/ls -d storage/experiments/*/*/*`; do 
    echo $instance
    bash -c "`cat ${instance}/command` &"  # character '&' is added here to run the experiment in the background  
   done
   ```

## License
This repository follows the MIT License