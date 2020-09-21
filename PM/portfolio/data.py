"""Portfolio data """

import abc
import logging
import pandas as pd
import numpy as np

from .utilities import *


logger = logging.getLogger(__name__)


__all__ = [
    'PortfolioTRIBase',
    'PortfolioTRI',
    'PortfolioDataBase',
    'PortfolioData'
]


DEFAULT_INITIAL_VALUE = 1

STAT_RETURN = 'return'
STAT_VOLATILITY = 'volatility'
STAT_SHARPE = 'sharpe'


class PortfolioTRIBase(metaclass=abc.ABCMeta):
    """Portfolio Total Return Index Data abstract base class
    """

    @property
    def tri(self):
        """pandas.Series: time-series total return index
        """

    @property
    def ret(self):
        """pandas.Series: time-series total returns"""


class PortfolioDataBase(PortfolioTRIBase, metaclass=abc.ABCMeta):
    """Portfolio data with TRI and holding weights abstract base class"""

    @property
    def weight(self):
        """pandas.DataFrame: portfolio holdings weight
        """


class PortfolioTRI(PortfolioTRIBase):
    """Portfolio data with total return index only

    Args:
        tri (pandas.DataFrame or pandas.Series): total return index
        ret (pandas.DataFrame or pandas.Series): total returns
    """

    def __init__(self, tri=None, ret=None, start_date=None, end_date=None, **kwargs):
        self._tri = None
        self._ret = None

        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self._stats = None

        self.periods_per_year = 252

        if tri is not None:
            self._process_input_tri(tri)
            if ret is not None:
                self._check_tri_and_ret(ret)
                pass
        elif ret is not None:  # no input tri
            self._process_input_ret(ret, **kwargs)
        else:
            raise ValueError('One of input tri and ret must be provided.')

    @property
    def tri(self):
        return self._tri  # or self._tri.copy()

    @property
    def ret(self):
        return self._ret  # or self._ret.copy()

    def _process_input_tri(self, tri):
        # assuming initial value is not NaN
        self._tri = tri
        self._ret = tri.pct_change().dropna(axis=0, how='all')

        self.initial_date = tri.first_valid_index()
        self.initial_value = tri.loc[self.initial_date]

    def _process_input_ret(self, ret, initial_date=None, initial_value=DEFAULT_INITIAL_VALUE):
        self._ret = ret
        self._tri = return2tri(ret, initial_date=initial_date, initial_value=initial_value)

        self.initial_value = self._tri.iloc[0]
        self.initial_date = self._tri.index[0]

    def _check_tri_and_ret(self, in_ret):
        # TODO: cross-check returns implied by input tri and input time-series returns
        # use pandas.DataFrame.diff()
        pass

    def __add__(self, port2):
        if not isinstance(port2, PortfolioTRI):
            TypeError('Subtraction for PortfolioTRI object only ')

        blend_ret = self.ret + self.port2
        blend = PortfolioTRI(ret=blend_ret, initial_value=self.initial_value, initial_date=self.initial_date)
        return blend

    def __sub__(self, bench):
        if not isinstance(bench, PortfolioTRI):
            TypeError('Subtraction for PortfolioTRI object only ')

        active_ret = self.ret - bench.ret
        active = PortfolioTRI(ret=active_ret, initial_value=self.initial_value, initial_date=self.initial_date)
        return active

    def __mul__(self, scaler):
        if not isinstance(scaler, (int, float)):
            TypeError('Multiplication with scaler only')

        ret = scaler * self.ret
        out = PortfolioTRI(ret=ret, initial_value=self, initial_date=self.initial_date)
        return out

    def compute_basic_stats(self):
        out_dict = {}
        first_index = self.tri.first_valid_index()
        last_index = self.tri.last_valid_index()
        out_dict[STAT_RETURN] = self.tri.loc[last_index] / self.tri.loc[first_index] - 1
        out_dict[STAT_RETURN] = np.sqrt(self.periods_per_year) * self.ret.std()
        out_dict[STAT_SHARPE] = out_dict[STAT_RETURN] / out_dict[STAT_SHARPE]

        self._stats = out_dict
        return out_dict

    @property
    def stats(self):
        if not self._stats:
            self.compute_basic_stats()
        return self._stats

    @property
    def ret(self):
        return self.stats[STAT_RETURN]

    @property
    def volatility(self):
        return self.stats[STAT_VOLATILITY]

    @property
    def sharpe(self):
        return self.stats[STAT_SHARPE]


class PortfolioData(PortfolioTRI, PortfolioDataBase):
    """Portfolio data with TRI, holding weights etc. """

    def __init__(self, tri=None, ret=None, start_date=None, end_date=None, weight=None, **kwargs):

        super().__init__(tri=tri, ret=ret, start_date=start_date, end_date=end_date, **kwargs)

        self._weight = weight  # index = 'date', columns = security

    @property
    def weight(self):
        return self._weight




