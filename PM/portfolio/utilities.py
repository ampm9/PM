"""Utility functions for portfolio analytics
"""

import itertools
import logging
import datetime
import dateutil

import numpy as np
import pandas as pd

from . import constants as pc


logger = logging.getLogger(__name__)


# @requires([pd.DataFrame, pd.Series])
def normalise(data, initial_value=100.):
    """Normalise the input time series of pandas Series or DataFrame;
    time series are normalised to their respective first available values.

    Args:
        data (pandas.Series or pandas.DataFrame): Input Series or DataFrame
        initial_value (float): A number specifying the initial value(s) of the time series.

    Returns:
        pandas.Series or pandas.DataFrame

    Raises:
        TypeError: If data is not pandas.Series nor pandas.DataFrame
    """
    if isinstance(data, pd.Series):
        first_index = data.first_valid_index()
        if first_index is None:
            return data
        return (initial_value / data.loc[first_index]) * data
    else:  # DataFrame
        out = data.copy()
        for c in data.columns:
            out[c] = normalise(data[c], initial_value)
        return out


def prev_bdate(in_date):
    """Get previous business date
    Input date is assumed to be business date. No holiday is handled.
    """
    return in_date - pd.tseries.offsets.BDay(1)


def next_bdate(in_date):
    """Get next business date
    Input date is assumed to be business date. No holiday is handled.
    """
    return in_date + pd.tseries.offsets.BDay(1)


def get_year_frac(start_date, end_date):
    """Get number of year in fraction between two dates.

    This implementation is an estimate, revisit here Jason.
    """

    delta = dateutil.relativedelta.relativedelta(end_date, start_date)
    return delta.years + delta.months / pc.MONTHS_PER_YEAR + delta.days / pc.DAYS_PER_YEAR_365


# @requires([pd.DataFrame, pd.Series])
def return2tri(data, initial_date=None, initial_value=100.):
    """Convert a return time-series to total return index

    Args:
        data(pandas.Series or pandas.DataFrame): Input Series or DataFrame
        initial_date: the initial index value to preappend (usually type of datetime)
        initial_value(float): initial value of total return index, defaults to 100.

    Returns:
        pandas.Series or pandas.DataFrame

    Raises:
        TypeError: If data is not pandas.Series nor pandas.DataFrame
    """

    # Check, length of input must be great than 2, empty in, empty out

    initial_date = pd.to_datetime(initial_date)

    if isinstance(data, pd.Series):

        if initial_date is not None:
            if initial_date > data.index.min():
                raise ValueError('Initial_date should be before all index dates')
            elif initial_date == data.index.min():
                raise ValueError('Input initial_date should not be equal to date index')

            prepended_row = pd.Series({initial_date: 0}, name=data.name)
            ret = pd.concat([prepended_row, data])
        elif pd.isnull(data.iloc[0]) and data.iloc[1:].notnull().all():
            ret = data
            ret.iloc[0] = 0
        elif data.notnull().all():
            initial_date = prev_bdate(data.index[0])  # previous business date
            prepended_row = pd.Series({initial_date: 0}, name=data.name)
            ret = pd.concat([prepended_row, data])
        # elif: # some NaN values exists, raise warning, but carry on calculation on non-NaN values
        else:
            raise ValueError('With no input initial_date, first data value must be NaN, the rest must be non-NaN')

        ratio = 1 + ret
        ret = initial_value * ratio.cumprod()  # remove NaN values
        return ret

    else:  # DataFrame
        ret_dict = {x: None for x in data.columns}
        for c in data.columns:
            ret_dict[c] = return2tri(data[c], initial_date=initial_date, initial_value=initial_value)
        ret = pd.DataFrame(ret_dict)
        ret = ret[data.columns]  # reorder columns
        return ret


def get_period_returns(data, freq):
    """Period returns of a time-series, un-annualized returns

    Args:
        data(pandas.Series or pandas.DataFrame): Input Series or DataFrame
        freq: periodic freq ['W', 'M', 'Q', 'H', 'A']

    Returns:
        pandas.Series or pandas.DataFrame

    Raises:
        TypeError: If data is not pandas.Series nor pandas.DataFrame
    """

    # freq can be 'M', 'Q', 'A'
    resampled = data.resample(freq)
    periodic_val = resampled.last()
    shifted_val = periodic_val.shift(1)
    periodic_ret = np.divide(periodic_val, shifted_val) - 1

    periodic_num = resampled.count()
    if periodic_num.iloc[0] > 1:
        periodic_ret.iloc[0] = np.divide(data.iloc[0], data.iloc[0]) - 1
    elif pd.isnull(periodic_ret.iloc[0]):
        periodic_ret = periodic_ret.iloc[1:]  # ignore the first value NaN
    else:
        raise ValueError('This should not occur')

    # hand the last period if it's not fully, append YTD, QTD, MTD
    return periodic_ret


def rolling_returns(data, period=pc.DAYS_PER_YEAR, option='simple'):
    if option == 'log':
        # avoid taking the log of non-positive numbers
        ratio = data / data.shift(period)
        return_data = np.log(ratio.where(ratio > 0))
    elif option == 'simple':
        shifted = data.shift(period)
        return_data = (data - shifted) / shifted.abs()
    else:
        raise ValueError('{} is not a valid option!'.format(option))
    return return_data


def rolling_cagr(data, period=pc.DAYS_PER_YEAR, option='simple'):
    """Rolling Compound Annual Growth Rate

    Args:
        data (pandas.Series or pandas.DataFrame): Input Series and DataFrame.
        period (Optional[int]): int representing the lookback period.
        option (Optional[str]): Simple or log return. Available options: {'log', 'simple'}. 'log' by default.

    Returns:
        float or pandas.Series
    """
    return_data = rolling_returns(data, period, option)

    years = np.float64(period) / np.float64(pc.DAYS_PER_YEAR)
    if option == 'log':
        return_data = return_data / years
    else:
        return_data = (1 + return_data) ** (np.float64(1) / np.float64(years)) - 1

    return return_data


def rolling_realised_vol(data, period=pc.DAYS_PER_YEAR):
    """
    Args:
        data (pandas.Series or pandas.DataFrame): Input Series and DataFrame.
        period (Optional[int]): int representing the lookback period. Defaulted to be DAYS_PER_YEAR.

    Returns:
        float or pandas.Series

    Raises:
        TypeError: If data is not pandas.Series nor pandas.DataFrame
    """
    daily_returns = data.pct_change()
    temp = daily_returns.rolling(window=period, center=False).std()
    return temp * np.sqrt(pc.DAYS_PER_YEAR)


def rolling_sharpe(data, period=pc.DAYS_PER_YEAR):
    """Rolling Sharpe ratio

    Args:
        data (pandas.Series or pandas.DataFrame): Input Series and DataFrame.
        period (Optional[int]): int representing the lookback period.

    Returns:
        float or pandas.Series

    Raises:
        TypeError: If data is not pandas.Series nor pandas.DataFrame
    """
    rol_ret = rolling_returns(data, period)
    rol_vol = rolling_realised_vol(data, period)
    rol_sharpe = rol_ret / rol_vol
    return rol_sharpe




