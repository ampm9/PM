"""Utility functions for portfolio analytics
"""

import itertools
import logging
import datetime

import pandas as pd

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




