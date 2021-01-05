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


def prev_bdate(in_date, d=1):
    """Get previous d business date(s), d defaults to be 1.
    Input date is assumed to be business date. No holiday is handled.
    """
    return in_date - pd.tseries.offsets.BDay(d)


def next_bdate(in_date, d=1):
    """Get next d business date(s), d defaults to be 1.
    Input date is assumed to be business date. No holiday is handled.
    """
    return in_date + pd.tseries.offsets.BDay(d)


def get_year_frac(start_date, end_date):
    """Get number of year in fraction between two dates.

    This implementation is an estimate, revisit here Jason.
    """

    delta = dateutil.relativedelta.relativedelta(end_date, start_date)
    return delta.years + delta.months / pc.MONTHS_PER_YEAR + delta.days / pc.DAYS_PER_YEAR_365


# @requires([pd.DataFrame, pd.Series])
def return2tri(data, initial_value=pc.DEFAULT_INITIAL_VALUE, initial_date=None):
    """Convert a return time-seres to its total return index.

    If initial date is not specified, initial date is data's first date index, first value does not matter;
    If initial date is specified and it is earlier than all data dates, prepending initial date.

    NaN values are preserved as NaN values, except the initial value.

    Args:
        data(pandas.Series or pandas.DataFrame): Input Series or DataFrame
        initial_value(float): initial value of total return index, defaults to 100.
        initial_date(date): the initial index value to prepend (usually type of datetime)

    Returns:
        pandas.Series or pandas.DataFrame

    Raises:
        TypeError: If data is not pandas.Series nor pandas.DataFrame
    """
    initial_date = pd.to_datetime(initial_date)

    if isinstance(data, pd.Series):
        if data.empty:  # edge case, empty in, empty out
            return data

        if initial_date is None:
            ret = data.copy()
            ret.iloc[0] = 0.
        elif initial_date == data.index[0]:
            ret = data.copy()
            ret.iloc[0] = 0.
        elif initial_date < data.index.min():
            prepended_row = pd.Series({initial_date: 0.}, name=data.name)
            ret = pd.concat([prepended_row, data])
        elif initial_date in pd.to_datetime(data.index):
            data2 = data[data.index >= initial_date].copy()
            ret2 = return2tri(data2, initial_value=initial_value, initial_date=initial_date)
            ret, _ = ret2.align(data, join='right')
            return ret
            ret.iloc[0] = 0.
        else:
            raise ValueError('Initial_date {} should be either in data.index or before all date index'.format(
                initial_date.strftime(pc.FORMAT_DATE)
            ))

        # process NaN-value in between with Warnings
        nan_idx = ret.isnull()
        if nan_idx.sum() > 0:
            msg = '{}/{} NaN-values in input return data {}'.format(nan_idx.sum(), len(nan_idx), data.name)
            logging.info(msg)
        ret[nan_idx] = 0.

        # returns to TRI
        ratio = 1 + ret
        tri = initial_value * ratio.cumprod()

        tri[nan_idx] = None  # re-set non-initial NaN-value
        return tri

    elif isinstance(data, pd.DataFrame):
        tri_dict = {x: None for x in data.columns}
        for c in data.columns:
            tri_dict[c] = return2tri(data[c], initial_date=initial_date, initial_value=initial_value)
        tri = pd.DataFrame(tri_dict)
        tri = tri[data.columns]  # reorder columns
        return tri

    elif data is None:
        return None

    else:
        raise TypeError('Input data must be either pandas.Series or pandas.DataFrame')


def tri2return(data):
    """Convert a time-series (total return) index to its returns (pct_change)

    This function differs from pandas pct_change() on how NAs are handled.
    This function preserves the NAs instead of filling NAs up (pad, ffill)

    Args:
        data(pandas.Series or pandas.DataFrame): Input Series or DataFrame

    Returns:
        pandas.Series or pandas.DataFrame

    Raises:
        TypeError: If data is not pandas.Series nor pandas.DataFrame
    """
    if isinstance(data, pd.Series):
        s = data[data.notnull()]
        if s.empty:
            return data.copy()  # empty in, empty out
        ret = s.pct_change()
        ret, _ = ret.align(data, join='right', axis=0)
        return ret

    elif isinstance(data, pd.DataFrame):
        ret_dict = {c: tri2return(data[c]) for c in data.columns}
        ret = pd.DataFrame(ret_dict)
        return ret[data.columns]  # reorder columns

    elif data is None:
        return None

    else:
        raise TypeError('Input data must be either pandas.Series or pandas.DataFrame')

def get_period_returns(data, freq, include_first_period=False, include_last_period=False):
    """Period returns of a time-series, un-annualized returns

    Args:
        data(pandas.Series or pandas.DataFrame): Input Series or DataFrame
        freq: periodic freq ['W', 'M', 'Q', 'H', 'A']
        include_first_period: whether include first period
        include_last_period: whether to include last period

    Returns:
        pandas.Series or pandas.DataFrame

    Raises:
        TypeError: If data is not pandas.Series nor pandas.DataFrame

    If include first period, first period must have more than 1 data point;
    If the last period is not a full period, append YTD, QTD, MTD outside this function.
    """

    # freq can be 'M', 'Q', 'A'
    resampled = data.resample(freq)
    periodic_val = resampled.last()
    shifted_val = periodic_val.shift(1)
    periodic_ret = np.divide(periodic_val, shifted_val) - 1

    periodic_num = resampled.count()

    if include_first_period:
        if periodic_num.iloc[0] > 1:
            periodic_ret.iloc[0] = np.divide(periodic_val.iloc[0], data.iloc[0]) - 1
        elif pd.isnull(periodic_ret.iloc[0]):
            periodic_ret = periodic_ret.iloc[1:]  # ignore the first value NaN
        else:
            raise ValueError('The first periodic return value should be null, but received {}'.format(periodic_ret.iloc[0]))
    else:
        periodic_ret = periodic_ret.iloc[1:]

    if not include_last_period:
        periodic_ret = periodic_ret.iloc[:-1]

    return periodic_ret


def get_simple_return(data):
    """Return of input pandas Series or DataFrames"""
    first_index = data.first_valid_index()
    last_index = data.last_valid_index()

    if data.loc[first_index] == 0:
        return np.inf
    else:
        return data.loc[last_index] / data.loc[first_index] - 1


def get_simple_cagr(data):
    """Compound Annual Growth Rate of input pandas Series or DataFrames"""
    if data.iloc[0] == 0:
        return np.inf
    else:
        first_index = data.first_valid_index()
        last_index = data.last_valid_index()
        num_years = get_year_frac(first_index, last_index)

        tri_ratio = np.divide(data.loc[last_index], data.loc[first_index])
        return np.power(tri_ratio, 1/num_years) - 1  # annualized return


def get_return_batting_average(data, port, bench):
    """Get batting average from return data.

    Batting average is the number of periods in which the portfolio betas or matches the benchmark
    by the total number of periods in the period being analyzed.
    Batting average measures an investment manager's ability to beat the benchmark.

    Args:
        data(pandas.DataFrame): Input return data
        port(str): Portfolio name
        bench(str): Benchmark name

    Returns:
        float
    """
    s_active = data[port] - data[bench]
    return sum(s_active > 0)/len(data)


def get_upside_capture_ratio(data, port, bench):
    """Return a portfolio's compound return divided by the benchmark's compounded return when the benchmark was up.
    The bigger the value, the better.

    Args:
        data(pandas.DataFrame): Input return data
        port(str): Portfolio name
        bench(str): Benchmark name

    Returns:
         float
    """
    bench_pos = data[bench] >= 0 
    tri = (1 + data[bench_pos]).cumprod()
    compound_ret = tri.loc[tri.last_valid_index()] - 1
    return compound_ret[port] / compound_ret[bench]


def get_downside_capture_ratio(data, port, bench):
    """Return a portfolio's compound return divided by the benchmark's compounded return when the benchmark was down.
    The smaller the value, the better.

    Args:
        data(pandas.DataFrame): Input return data
        port(str): Portfolio name
        bench(str): Benchmark name

    Returns:
         float
    """
    bench_neg = data[bench] < 0
    tri = (1 + data[bench_neg]).cumprod()
    compound_ret = tri.loc[tri.last_valid_index()] - 1
    return compound_ret[port] / compound_ret[bench]





