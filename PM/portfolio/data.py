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

    @property
    def risk_free_tri(self):
        """pandas.Series: risk-free rate return index, default to constant time-series with portfolio's initial value"""

    @property
    def risk_free_rate(self):
        """pandas.Series: risk-free interest rate, default to zero"""

    @property
    def excess_tri(self):
        """pandas.Series or pandas.DataFrame: time-series excess return index"""

    @property
    def excess_ret(self):
        """pandas.Series or pandas.DataFrame: time-series excess returns"""


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

    def __init__(self, tri=None, ret=None, risk_free_tri=None, risk_free_rate=None, name=None, **kwargs):

        # portfolio returns and TRI
        if tri is None and ret is None:
            raise ValueError('One of input tri and ret must be provided.')
        else:
            self._tri, self._ret, _ = pu.process_tri_or_return(tri=tri, ret=ret, **kwargs)

        # self._ret = self._ret.dropna(axis=0, how='all') # remove first NaN

        self.initial_date = tri.first_valid_index()
        self.initial_value = tri.loc[self.initial_date]

        # Portfolio name, pandas.Series only
        if isinstance(self.tri, pd.Series):
            if name is None:
                self._name = self.tri.name
            else:
                self._name = name
                self._tri.name = name
                self._ret.name = name
        else:
            self._name = ', '.join(self.tri.columns.tolist())

        # risk free rate and risk free index
        if risk_free_tri is None and risk_free_rate is None:
            self._risk_free_tri = pd.Series(1, index=self.tri.index, name='risk_free')
            self._risk_free_rate = risk_free_tri.pct_change()
        else:
            self._risk_free_tri, self._risk_free_rate = pu.process_tri_or_return(tri=risk_free_tri, ret=risk_free_rate, **kwargs)

        # excess return and TRI
        self._excess_ret = self.ret.subtract(self.risk_free_rate, axis=0)
        self._excess_tri = pu.return2tri(self.excess_ret, initial_date=self.initial_date, initial_value=self.initial_value)

        if isinstance(self.ret, pd.Series):
            self._excess_ret.name = self.ret.name
            self._excess_tri.name = self.tri.name

        # stats metrics
        self._stats = None
        self._years = None
        self._periods_per_year = None

        # run
        self._run_basic_stats()

    @property
    def tri(self):
        return self._tri.copy()

    @property
    def ret(self):
        return self._ret.copy()

    @property
    def risk_free_tri(self):
        return self._risk_free_rate.copy()

    @property
    def risk_free_rate(self):
        return self._risk_free_rate.copy()

    @property
    def excess_tri(self):
        return self._ex_tri.copy()

    @property
    def excess_ret(self):
        return self._ex_ret.copy()

    def _run_basic_stats(self):
        out_dict = {}
        first_index = self.tri.first_valid_index()
        last_index = self.tri.last_valid_index()

        num_years = pu.get_year_frac(first_index, last_index)
        periods_per_year = len(self.tri) / num_years

        tri_ratio = np.divide(self.tri.loc[last_index], self.tri.loc[first_index])
        out_dict[pc.RETURN] = tri_ratio - 1  # simple return
        out_dict[pc.CAGR] = np.power(tri_ratio, 1/num_years) - 1  # annualized return

        ex_tri_ratio = np.divide(self.excess_tri.loc[last_index], self.excess_tri.loc[first_index])
        out_dict[pc.RETURN_EXCESS] = ex_tri_ratio - 1
        out_dict[pc.CAGR_EXCESS] = np.power(ex_tri_ratio, 1/num_years) - 1

        out_dict[pc.VOLATILITY] = np.sqrt(periods_per_year) * self.ret.std()
        ex_vol = np.sqrt(periods_per_year) * self.excess_ret.std()
        out_dict[pc.SHARPE] = out_dict[pc.CAGR_EXCESS] / ex_vol

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
        ## TODO: should align two time-series returns, same to __sub__
        if not isinstance(port2, PortfolioTRI):
            TypeError('Subtraction for PortfolioTRI object only ')

        blend_ret = self.ret + port2.ret
        blend = PortfolioTRI(ret=blend_ret, initial_value=self.initial_value, initial_date=self.initial_date)
        return blend

    def __sub__(self, bench):
        if not isinstance(bench, PortfolioTRI):
            TypeError('Subtraction for PortfolioTRI object only ')

        active_ret = self.ret - bench.ret
        active = PortfolioTRI(ret=active_ret, initial_value=self.initial_value, initial_date=self.initial_date)
        return active

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            TypeError('Multiplication with scaler only')

        ret = scalar * self.ret
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




