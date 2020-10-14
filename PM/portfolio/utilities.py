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
def return2tri(data, initial_value=100., initial_date=None):
    """Convert a return time-seres to total return index.

    If initial date is not specified, initial date is data's first date index, first value does not matter;
    If initial date is specified and it is earlier than data index, prepending initial date.

    Args:
        data(pandas.Series or pandas.DataFrame): Input Series or DataFrame
        initial_date(date): the initial index value to prepend (usually type of datetime)
        initial_value(float): initial value of total return index, defaults to 100.

    Returns:
        pandas.Series or pandas.DataFrame

    Raises:
        TypeError: If data is not pandas.Series nor pandas.DataFrame
    """
    initial_date = pd.to_datetime(initial_date)

    if isinstance(data, pd.Series):
        if data.isempty():  # edge case, empty in, empty out
            return data

        if initial_date is None:
            ret = data
            ret.iloc[0] = 0.
        elif initial_date == data.index[0]:
            ret = data
            ret.iloc[0] = 0.
        elif initial_date < data.index.min():
            prepended_row = pd.Series({initial_date: 0.}, name=data.name)
            ret = pd.concat([prepended_row, data])
        elif initial_date in pd.to_datetime(data.index):
            ret = data[data.index >= initial_date]
            ret.iloc[0] = 0.
            raise Warning('Input initial date {} is not data.index[0] = {}, truncate input data series'.format(
                initial_date.strftime(pc.FORMAT_DATE),
                data.index[0].strftime(pc.FORMAT_DATE)
            ))
        else:
            raise ValueError('Initial_date {} should be either in data.index or before all date index'.format(
                initial_date.strftime(pc.FORMAT_DATE)
            ))

        # process NaN-value, raise Warnings
        nan_idx = ret.isnull()
        if nan_idx.sum() > 0:
            raise Warning('{}/{} NaN-values in input return data default to 0'.format(nan_idx.sum(), len(nan_idx)))
        ret[nan_idx] = 0.

        # returns to TRI
        ratio = 1 + ret
        ret = initial_value * ratio.cumprod()
        return ret

    elif isinstance(data, pd.DataFrame):
        ret_dict = {x: None for x in data.columns}
        for c in data.columns:
            ret_dict[c] = return2tri(data[c], initial_date=initial_date, initial_value=initial_value)
        ret = pd.DataFrame(ret_dict)
        ret = ret[data.columns]  # reorder columns
        return ret

    elif data is None:
        return None

    else:
        raise TypeError('Input data must be either pandas.Series or pandas.DataFrame')


def process_tri_or_return(tri=None, ret=None, initial_value=100., initial_date=None):
    """Compute total return index or return if either one of them are input;
    If neither TRI and return are not None, check if they represent the same index.

    Args:
        tri(pandas.Series or pandas.DataFrame): Input total return index Time-Series or DataFrame
        ret(pandas.Series or pandas.DataFrame): Input return Time-Series or DataFrame
        initial_date(date): the initial index value to prepend (usually type of datetime)
        initial_value(float): initial value of total return index, defaults to 100.

    Returns:
        (pandas.Series or pandas.DataFrame, pandas.Series or pandas.DataFrame, boolean)
    """
    if tri is None and ret is None:
        return None, None, True

    if tri is not None and ret is None:
        if initial_date is not None:
            raise Warning('Initial_date and initial_value are not required if return data is not input')
        ret = tri.pct_change()
        return tri, ret, True

    if tri is None and ret is not None:
        tri = return2tri(ret, initial_date=initial_value, initial_date=initial_date)
        return tri, ret, True

    if tri is not None and ret is not None:
        if all(isinstance(x, pd.Series) for x in [tri, ret]):
            if tri.empty or ret.empty:
                return tri, ret, tri.empty and ret.empty

            ret2 = tri.pct_change()
            if len(ret.index) != len(ret2.index):
                raise Warning('Input TRI and return data index length mismatch ')
                return tri, ret, False
            is_equal = np.isclose(ret.iloc[1:].to_numpy(), ret2.iloc[1:].to_numpy())
            return tri, ret, all(is_equal)

        elif all(isinstance(x, pd.DataFrame) for x in [tri, ret]):
            if len(tri.columns.symmetric_difference(ret.columns)) > 1:
                raise Warning('Input TRI and return DataFrame columns mismatch')
                return tri, ret, False

            is_equal_dict = {x: False for x in tri.columns}
            for c in tri.columns:
                _, _, is_equal = process_tri_or_return(tri=tri[c], ret=ret[c])
                is_equal_dict[c] = is_equal
            is_equal = all(is_equal_dict.values())
            return tri, ret, False

        else:
            raise Warning('Input TRI and return type mismatch ')
            return tri, ret, False


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


def get_simple_return(data):
    """Return of input pandas Series or DataFrames"""
    if data.iloc[0] == 0:
        return np.inf
    else:
        return data.iloc[-1] / data.iloc[0] - 1
