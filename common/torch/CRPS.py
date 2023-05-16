import numpy as np
import pandas as pd

from gluonts.evaluation import MultivariateEvaluator
from  gluonts.model.forecast import SampleForecast

import datetime


def calculate_crps(Y_samples, test_data_target):
    """
    :param Y_samples: is 4D of size (num_rollings, rolling_window, num_samples, num_series)
    :param test_data_target: is 3D of size (num_rollings, rolling_window, num_series)

    :return: crps and crps_sum
    """
    # targets_in_batch is a list of series where each of size (rolling_window, num_rollings)
    targets_in_batch = [np.array(test_data_target[:, :, i]).T for i in range(test_data_target.shape[2])]

    window_size = targets_in_batch[0].shape[0]

    # Start date is arbitrary - any date can be used here
    start_date = pd.Timestamp(datetime.date(2019, 1, 1))
    start_period = pd.Period(start_date, freq='1d')

    def get_target_df(targets_in_batch):
        index = pd.date_range(
            start=start_date,
            freq='1d',
            periods=window_size,
        ).to_period(freq='D')
        return pd.DataFrame(index=index, data=targets_in_batch)

    target_dfs = [get_target_df(target) for target in targets_in_batch]

    # Get forecast samples, a list of number of series, each of size (num_samples, rolling_window, num_rollings)

    # Y_samples is 4D of size (num_rollings, rolling_window, num_samples, num_series)
    num_rollings, rolling_window, num_samples, num_series = Y_samples.shape
    all_sampled_forecasts = Y_samples.transpose(3, 2, 1, 0)

    all_sampled_forecasts = [SampleForecast(samples=forecast_sample, start_date=start_period) for forecast_sample in
                             all_sampled_forecasts]

    mve = MultivariateEvaluator(quantiles=(np.arange(20) / 20.0)[1:], target_agg_funcs={'sum': np.sum})

    agg_metrics, item_metrics = mve(target_dfs, all_sampled_forecasts, num_series)

    crps = agg_metrics["mean_wQuantileLoss"]
    crps_sum = agg_metrics["m_sum_mean_wQuantileLoss"]

    return crps, crps_sum