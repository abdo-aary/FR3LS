import os
import logging
import gin
import torch as t
from fire import Fire
from common.neg_sampler import TimeSeriesSampler
from common.experiment import Experiment
from common.settings import DATASETS_PATH
from common.torch.snapshots import SnapshotManager
from common.utils import count_parameters
import numpy as np
from experiments.trainers.trainer import aldy_trainer
from models.aldy.aldy_2 import ALDy


class ALDyExperiment(Experiment):
    @gin.configurable
    def instance(self,
                 ts_dataset_name: str,

                 train_window: int,
                 train_mode: str,  # Alternative or not
                 f_input_window: int,
                 f_out_same_in_size: bool,
                 horizon: int,
                 n_test_windows: int,
                 n_val_windows: int,
                 non_overlap_batch: int,
                 shuffle: bool,

                 f_model_type: str,
                 f_num_shared_channels: list[int],
                 f_nbr_param_layers: int,
                 f_kernel_size: int,
                 f_dropout: int,
                 f_leveld_init: bool,
                 f_implicit_batching: bool,
                 f_alpha: float,

                 ae_hidden_dims: list[int],

                 f_model_path: str = None,
                 ts2vec_path: str = None,

                 ts2vec_encoder_path: str = None,  # Path to pretrained ts2vec_model
                 ts2vec_output_dims: int = 320,
                 ts2vec_hidden_dims: int = 64,
                 ts2vec_depth: int = 10,
                 ts2vec_mask_mode: str = 'binomial',

                 activation: str = 'gelu',
                 train_loss_name: str = 'ALDY',
                 val_loss_name: str = 'MSE',
                 lambda1: int = 1,
                 lambda2: int = 1,
                 lambda3: int = 1,
                 train_ae_loss: str = 'MAE',
                 train_forecasting_loss: str = 'MAE',

                 epochs: int = 750,
                 epochs_TLAE: int = 750,
                 epochs_TS2VEC: int = 200,
                 epochs_ParamGen: int = 200,
                 iterations: int = 150,
                 alt_epoch_nbr: int = 40,
                 batch_size: int = 32,
                 random_state: int = 42,
                 learning_rate: float = 0.001,
                 repeat: int = 0,
                 verbose: bool = True,
                 early_stopping: bool = True,
                 patience: int = 5,
                 ) -> None:

        t.manual_seed(random_state)

        if verbose:
            print("Data loading ...")

        dataset_path = os.path.join(DATASETS_PATH, ts_dataset_name, ts_dataset_name + '.npy')
        ts_samples = np.load(dataset_path).transpose()  # ts_samples of shape (T, N)

        ts_sampler = TimeSeriesSampler(timeseries=ts_samples,
                                       train_window=train_window,
                                       batch_size=batch_size,
                                       f_input_window=f_input_window,
                                       horizon=horizon,
                                       non_overlap_batch=non_overlap_batch,
                                       n_test_windows=n_test_windows,
                                       n_val_windows=n_val_windows,
                                       shuffle=shuffle)

        idx_hidden_dim = 0
        for i in range(len(ae_hidden_dims) - 1):
            if ae_hidden_dims[i] == ae_hidden_dims[i + 1]:
                idx_hidden_dim = i
                break
        latent_dim = ae_hidden_dims[idx_hidden_dim]
        f_num_inputs = latent_dim
        f_output_size = latent_dim

        f_model_params = {'model_type': f_model_type,
                          'num_inputs': f_num_inputs,
                          'num_shared_channels': f_num_shared_channels,
                          'nbr_param_layers': f_nbr_param_layers,
                          'ts2vec_output_dims': ts2vec_output_dims,
                          'kernel_size': f_kernel_size,
                          'dropout': f_dropout,
                          'leveld_init': f_leveld_init,
                          'implicit_batching': f_implicit_batching,
                          'alpha': f_alpha}

        if verbose:
            print("\n\nModel Training ...")
        input_dim = ts_samples.shape[-1]

        model = ALDy(input_dim=input_dim,
                     ae_hidden_dims=ae_hidden_dims,
                     f_model_params=f_model_params,
                     f_input_window=f_input_window,
                     train_window=train_window,
                     f_out_same_in_size=f_out_same_in_size,
                     ts2vec_output_dims=ts2vec_output_dims,
                     ts2vec_hidden_dims=ts2vec_hidden_dims,
                     ts2vec_depth=ts2vec_depth,
                     ts2vec_mask_mode=ts2vec_mask_mode,
                     activation=activation)

        if verbose:
            print(model)
            print(f'parameter count: {count_parameters(model)}')

        # Train model
        snapshot_manager = SnapshotManager(
            snapshot_dir=os.path.join(self.root, 'snapshots'),
            other_losses=['training_hier_R', 'validation_hier_R', 'SMAPE', 'WAPE']
        )

        trainer = aldy_trainer

        _ = trainer(snapshot_manager=snapshot_manager,
                    model=model,
                    training_set=iter(ts_sampler),
                    sampler=ts_sampler,
                    horizon=horizon,
                    train_loss_name=train_loss_name,
                    val_loss_name=val_loss_name,
                    lambda1=lambda1,
                    lambda2=lambda2,
                    lambda3=lambda3,
                    epochs=epochs,
                    iterations=iterations,
                    learning_rate=learning_rate,
                    verbose=verbose,
                    early_stopping=early_stopping,
                    patience=patience)

        snapshot_manager.print_losses()

        if verbose:
            print("\n\n##############################################################")
            print("######################## || DONE :) || #######################")
            print("##############################################################")


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire(ALDyExperiment)
