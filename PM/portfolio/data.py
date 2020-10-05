"""Portfolio data """

import abc
import logging
import dateutil

import pandas as pd
import numpy as np

from . import constants as pc
from . import utilities as pu


logger = logging.getLogger(__name__)


__all__ = [
    'PortfolioTRIBase',
    'PortfolioTRI',
    'PortfolioDataBase',
    'PortfolioData'
]





class PortfolioTRIBase(metaclass=abc.ABCMeta):
    """Portfolio Total Return Index Data abstract base class

    TRI input can be pandas.Series (single portfolio) or pandas.DataFrame (multiple portfolio)
    """

    @property
    def tri(self):
        """pandas.Series or pandas.DataFrame: time-series total return index
        """

    @property
    def ret(self):
        """pandas.Series or pandas.DataFrame: time-series total returns"""


class PortfolioDataBase(PortfolioTRIBase, metaclass=abc.ABCMeta):
    """Portfolio data abstract base class including TRI, weight and so on

    PortfolioDataBase is for single portflio, TRI in pandas.Series, weight in 2D pandas.DataFrame"""

    @property
    def weight(self):
        """pandas.DataFrame: portfolio holdings weight
        """

    @property
    def assets(self):
        """pandas.DataFrame: portfolio asset price or total return index
        """


class PortfolioTRI(PortfolioTRIBase):
    """Portfolio data with total return index only

    Args:
        tri (pandas.DataFrame or pandas.Series): total return index
        ret (pandas.DataFrame or pandas.Series): total returns
    """

    def __init__(self, tri=None, ret=None, name=None, start_date=None, end_date=None, **kwargs):
        self._tri = None
        self._ret = None

        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)

        self._stats = None
        self._years = None
        self._periods_per_year = None

        # required input, one of tri and ret must be valid
        if tri is not None:
            self._process_input_tri(tri)
            if ret is not None:
                self._check_tri_and_ret(ret)
                pass
        elif ret is not None:  # no input tri
            self._process_input_ret(ret, **kwargs)
        else:
            raise ValueError('One of input tri and ret must be provided.')

        # Portfolio name, pandas.Series only
        if isinstance(self.tri, pd.Series):
            if name is None:
                name = self.tri.name
            else:
                self._tri.name = name
                self._ret.name = name
        else:
            if name is not None:
                raise Warning('Input data not pandas.Series, ignore input name')

        self._run_basic_stats()

    @property
    def tri(self):
        return self._tri.copy()

    @property
    def ret(self):
        return self._ret.copy()

    def _process_input_tri(self, tri):
        # assuming initial value is not NaN
        self._tri = tri
        self._ret = tri.pct_change().dropna(axis=0, how='all')

        self.initial_date = tri.first_valid_index()
        self.initial_value = tri.loc[self.initial_date]

    def _process_input_ret(self, ret, initial_date=None, initial_value=pc.DEFAULT_TRI_INITIAL_VALUE):
        self._ret = ret
        self._tri = pu.return2tri(ret, initial_date=initial_date, initial_value=initial_value)

        self.initial_value = self._tri.iloc[0]
        self.initial_date = self._tri.index[0]

    def _check_tri_and_ret(self, in_ret):
        """ TODO: cross-check returns implied by input tri and input time-series returns
        """
        pass

    def _run_basic_stats(self):
        out_dict = {}
        first_index = self.tri.first_valid_index()
        last_index = self.tri.last_valid_index()

        num_years = pu.get_year_frac(first_index, last_index)
        periods_per_year = len(self.tri) / num_years

        tri_ratio = np.divide(self.tri.loc[last_index], self.tri.loc[first_index])

        out_dict[pc.RETURN] = tri_ratio - 1  # simple return
        out_dict[pc.CAGR] = np.power(tri_ratio, 1/num_years) - 1  # annualized return
        out_dict[pc.VOLATILITY] = np.sqrt(periods_per_year) * self.ret.std()
        out_dict[pc.SHARPE] = out_dict[pc.CAGR] / out_dict[pc.VOLATILITY]

        self._stats = out_dict
        self._years = num_years
        self._periods_per_year = periods_per_year
        return out_dict

    @property
    def years(self):
        if self._years is None:
            self._run_basic_stats()
        return self._years

    @property
    def periods_per_year(self):
        """Number of periods (daily, weekly or monthly depending on underlying data) per year in integer. """
        if self._periods_per_year is None:
            self._run_basic_stats()
        return int(np.ceil(self._periods_per_year))

    @property
    def stats(self):
        if not self._stats:
            self._run_basic_stats()
        return self._stats

    @property
    def cagr(self):
        """Compound Annual Growth Rate (CAGR), geometrically annualized return"""
        return self.stats[pc.CAGR]

    @property
    def volatility(self):
        """Annualized Volatility"""
        return self.stats[pc.VOLATILITY]

    @property
    def sharpe(self):
        """Sharpe Ratio"""
        return self.stats[pc.SHARPE]

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
        out = PortfolioTRI(ret=ret, initial_value=self.initial_value, initial_date=self.initial_date)
        return out


class PortfolioData(PortfolioTRI, PortfolioDataBase):
    """Portfolio data with TRI, holding weights etc. """

    def __init__(self, tri=None, ret=None, start_date=None, end_date=None, weight=None, **kwargs):

        super().__init__(tri=tri, ret=ret, start_date=start_date, end_date=end_date, **kwargs)

        self._weight = weight  # index = 'date', columns = security

    @property
    def weight(self):
        return self._weight




