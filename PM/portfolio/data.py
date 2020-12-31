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

    PortfolioDataBase is for single portfolio, TRI in pandas.Series, weight in 2D pandas.DataFrame"""

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

    def __init__(self, tri=None, risk_free_tri=None, name=None, start=None, end=None):
        # variable initialization
        self._tri = None
        self._ret = None

        self._risk_free_tri = None
        self._risk_free_rate = None

        self._excess_tri = None
        self._excess_ret = None

        self._name = None
        self._start = None
        self._end = None

        # for returns to tri
        self.initial_value = None
        self.initial_date = None  # may not be start date

        # stats metrics
        self._stats = None
        self._years = None
        self._periods_per_year = None

        # initialize risk free rate if input
        if risk_free_tri is not None:
            self._risk_free_tri = risk_free_tri
            self._risk_free_rate = pu.tri2return(risk_free_tri)

        if tri is None:
            return

        self.initialize_tri(tri=tri, name=name, start=start, end=end)

        self.run()

    @property
    def tri(self):
        if self._tri is None:
            return None
        return self._tri.copy()

    @property
    def ret(self):
        if self._ret is None:
            return None
        return self._ret.copy()

    @property
    def risk_free_tri(self):
        if self._risk_free_tri is None:
            return None
        return self._risk_free_tri.copy()

    @property
    def risk_free_rate(self):
        if self._risk_free_rate is None:
            return None
        return self._risk_free_rate.copy()

    @property
    def excess_tri(self):
        if self._excess_tri is None:
            return None
        return self._excess_tri.copy()

    @property
    def excess_ret(self):
        if self._excess_ret is None:
            return None
        return self._excess_ret.copy()

    @property
    def name(self):
        return self._name

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @tri.setter
    def tri(self, tri):
        self.initialize_tri(tri)

    @ret.setter
    def ret(self, ret):
        self.initialize_ret(ret)

    def initialize_tri(self, tri, start=None, end=None, name=None):
        """Set portfolio total return index
        1. check input tri type
        2. truncate time-series between start and end
        3. rename if there is an input name
        """
        if self.tri is not None:
            raise ValueError('Portfolio TRI is specified already.')

        if not isinstance(tri, pd.Series):
            raise TypeError('Invalid input type error, class(tri) = '.format(type(tri)))

        if name is not None:
            tri.name = name
        self._name = tri.name

        if self._start is None:
            self._start = tri.index[0]
        else:
            self._start = pd.to_datetime(start)
            tri = tri[tri.index >= start]

        if self._end is None:
            self._end = tri.index[-1]
        else:
            self._end = pd.to_datetime(end)
            tri = tri[tri.index <= end]

        self.initial_date = tri.first_valid_index()
        self.initial_value = tri.loc[self.initial_date]

        self._tri = tri
        self._ret = pu.tri2return(tri)

        # excess return and TRI
        if self.risk_free_rate is not None:
            self._excess_ret = self.ret.subtract(self.risk_free_rate, axis=0)
            self._excess_tri = pu.return2tri(self._excess_ret,
                                             initial_date=self.initial_date,
                                             initial_value=self.initial_value)
        else:
            self._excess_ret = self._ret
            self._excess_tri = self._tri

    def initialize_ret(self, ret, initial_value=None, initial_date=None):
        """Set portfolio total returns"""
        if self.ret is not None:
            raise ValueError('Portfolio returns are specified already.')
        tri = pu.return2tri(ret, initial_value=initial_value, initial_date=initial_date)
        self.initialize_tri(tri)

    def run(self):
        self._run_basic_stats()

    def _run_basic_stats(self):
        out_dict = {}

        # empty in, empty out
        if self.tri.dropna().empty:
            self._stats = out_dict
            return

        num_years = pu.get_year_frac(self.start, self.end)
        periods_per_year = len(self.tri) / num_years

        first_index = self.tri.first_valid_index()
        last_index = self.tri.last_valid_index()

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
        if self._stats is None:
            self._run_basic_stats()
        return self._stats

    @property
    def cagr(self):
        """Compound Annual Growth Rate (CAGR), geometrically annualized return"""
        return self.stats.get(pc.CAGR)

    @property
    def volatility(self):
        """Annualized Volatility"""
        return self.stats.get(pc.VOLATILITY)

    @property
    def sharpe(self):
        """Sharpe Ratio"""
        return self.stats.get(pc.SHARPE)

    def __add__(self, port2):
        ## TODO: should align two time-series returns, same to __sub__
        if not isinstance(port2, PortfolioTRI):
            TypeError('Addition for PortfolioTRI object only ')
        blend_ret = self.ret + port2.ret
        blend = PortfolioTRI()
        blend.initialize_ret(blend_ret, initial_value=self.initial_value, initial_date=self.initial_date)

        name = '{} {}'.format(self.name, port2.name)
        blend._name = name
        blend._start = self.start
        blend._end = self.end
        blend.run()
        return blend

    def __sub__(self, bench):
        if not isinstance(bench, PortfolioTRI):
            TypeError('Subtraction for PortfolioTRI object only ')

        active_ret = self.ret - bench.ret
        active = PortfolioTRI()
        active.initialize_ret(active_ret, initial_value=self.initial_value, initial_date=self.initial_date)

        name = '{} - {}'.format(self.name, bench.name)
        active._name = name
        active._start = self.start
        active._end = self.end
        active.run()
        return active

    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            TypeError('Multiplication with scaler only')

        ret = scalar * self.ret
        out = PortfolioTRI()
        out.initialize_ret(ret, initial_value=self.initial_value, initial_date=self.initial_date)

        name = '{:.2f} {}'.format(scalar, self.name)
        out._name = name
        out._start = self.start
        out._end = self.end
        out.run()
        return out


class PortfolioData(PortfolioTRI, PortfolioDataBase):
    """Portfolio data with TRI, holding weights etc. """

    def __init__(self, weight=None, assets=None, initial_value=None):

        super().__init__()

        self._weight = weight
        self._assets = assets

    @property
    def weight(self):
        # index = 'date', columns = security
        return self._weight

    @property
    def assets(self):
        return self._assets

    @weight.setter
    def weight(self, weight):
        self._weight = weight

    @assets.setter
    def assets(self, assets):
        self._assets = assets


