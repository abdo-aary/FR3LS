from typing import Optional

import torch as t
import torch.nn.functional as F

from models.f_models.deepLR_f_models.lstm import LSTM_Modified
from models.f_models.deepLR_f_models.lstm_param_gen import LSTM_Modified as LSTM_ParamGEN
from models.f_models.deepLR_f_models.rnn_encoder_decoder import RNN_SQ2SQ
from models.f_models.deepLR_f_models.tcn_modified import TCN_Modified
# from models.f_models.tcn_param_gen import TCN_Parameterized
from models.f_models.deepLR_f_models.tcn_param_gen_6 import TCN_Parameterized
from models.f_models.deepLR_f_models.tcn_param_gen_Linear import TCN_Parameterized as TCN_Linear_Parameterized


def activation_case_match(src: t.Tensor, activation: Optional[str] = None) -> t.Tensor:
    if activation is None:
        return src
    elif activation == 'relu':
        return t.relu(src)
    elif activation == 'gelu':
        return F.gelu(src)
    elif activation == 'sigmoid':
        return t.sigmoid(src)
    elif activation == 'tanh':
        return t.tanh(src)
    elif activation == 'leaky_relu':
        return t.nn.functional.leaky_relu(src)
    else:
        raise ValueError(f'activation {activation} not implemented')


def activation_layer(activation: str):
    if activation == 'relu':
        return t.nn.ReLU(True)
    elif activation == 'gelu':
        return t.nn.GELU()
    elif activation == 'sigmoid':
        return t.nn.Sigmoid()
    elif activation == 'tanh':
        return t.nn.Tanh()
    elif activation == 'leaky_relu':
        return t.nn.LeakyReLU()
    else:
        raise ValueError(f'activation {activation} not implemented')


def _init_weights_xavier(m):
    if isinstance(m, t.nn.Linear):
        t.nn.init.xavier_uniform_(m.weight)


def load_forecasting_model(params: dict) -> t.nn.Module:
    if 'TCN_Param' in params['model_type']:
        if params['model_type'] == 'TCN_Parameterized':
            TCN_Model = TCN_Parameterized
        elif params['model_type'] == 'TCN_Param_Linear':
            TCN_Model = TCN_Linear_Parameterized
        else:
            raise Exception(f"Unknown TCN model {params['model_type']}")
        f_model = TCN_Model(num_inputs=params['num_inputs'],
                            num_shared_channels=params['num_shared_channels'],
                            nbr_param_layers=params['nbr_param_layers'],
                            ts2vec_output_dims=params['ts2vec_output_dims'],
                            kernel_size=params['kernel_size'],
                            dropout=params['dropout'],
                            leveld_init=params['leveld_init'],
                            implicit_batching=params['implicit_batching'],
                            alpha=params['alpha'])

        return f_model

    elif params['model_type'] == 'TCN_Modified':
        f_model = TCN_Modified(num_inputs=params['num_inputs'],
                               output_size=params['output_size'],
                               num_channels=params['num_channels'],
                               kernel_size=params['kernel_size'],
                               dropout=params['dropout'],
                               leveld_init=params['leveld_init']
                               )

        return f_model

    elif params['model_type'] == 'LSTM_Modified':
        f_model = LSTM_Modified(input_size=params['input_size'],
                                output_size=params['output_size'],
                                hidden_size=params['hidden_size'],
                                num_layers=params['num_layers'],
                                batch_first=params['batch_first'],
                                dropout=params['dropout'],
                                )
        return f_model
    elif params['model_type'] == 'RNN_SEQ2SEQ':
        f_model = RNN_SQ2SQ(input_size=params['input_size'],
                            hidden_dim=params['hidden_size'],
                            num_layers=params['num_layers'],
                            batch_first=params['batch_first'],
                            dropout=params['dropout'],
                            teacher_forcing_ratio=params['teacher_forcing_ratio'],
                            )
        return f_model

    elif params['model_type'] == 'LSTM_ParamGEN':
        f_model = LSTM_ParamGEN(input_size=params['input_size'],
                                output_size=params['output_size'],
                                hidden_size=params['hidden_size'],
                                num_layers=params['num_layers'],
                                batch_first=params['batch_first'],
                                dropout=params['dropout'],
                                alpha=params['alpha'],
                                )
        return f_model
    else:
        raise Exception(f"Unknown model {params['model_type']}")
