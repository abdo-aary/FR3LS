# FR3LS

[//]: # (Datasets used can be found here: https://drive.google.com/drive)

To run the project, follow the following steps:

1. Configure python path in the project root
   ```shell script
    export PYTHONPATH=$PYTHONPATH:`pwd`
    ```
2. Build (Download/cache) dataset
   ```shell script
    python datasets/main.py --config_path=datasets/configs/dataset_config.gin
    ```
3. Build an embedding experiment
    ```shell script
    python experiments/experiment_name/forecasting_main_instance.py build_ensemble --config_path=experiments/experiment_name/configs/embedding_simple_autoencoder.gin
    ```
   
4. Run specific experiment
    ```shell script
    bash -c "`cat storage/experiments/path/to/experiment/command`"
    ```
   
5. You can automate running all sub-experiments of a specific model sequentially with the following example:
    ```shell script
   # For a certain model // model_name = simple_autoencoder
   for instance in `/bin/ls -d storage/experiments/simple_autoencoder/*/*`; do 
        echo $instance
        bash -c "`cat ${instance}/command`"
   done

<!--
for instance in `/bin/ls -d storage/experiments/embedding_simple_autoencoder/daily*/*series2vec*`; do 
        echo $instance
        bash -c "`cat ${instance}/command`"
   done
!-->

6. You can automate running all experiments of all models sequentially with the following example:   
   ```shell script
   # For all models
   for instance in `/bin/ls -d storage/experiments/*/*/*`; do 
    echo $instance
    bash -c "`cat ${instance}/command`"
   done
    ```

## License
This repository follows the MIT License