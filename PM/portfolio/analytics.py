"""Portfolio analytics"""

import abc
import collections
import logging

import numpy as np
import pandas as pd

from . import constants as pc
from . import utilities as pu
from . import data as pdata


logger = logging.getLogger(__name__)


__all__ = [
    'PortfolioAnalyticsBase',
    'PortfolioAnalyticsTRI',
    'PortfolioAnalytics'
]


class PortfolioAnalyticsBase(metaclass=abc.ABCMeta):
    """Portfolio Analytics abstract base class
    """

    @property
    def stats(self):
        """dict: performance statistics
        """

    @property
    def run(self):
        """run portfolio analytics
        """


class PortfolioAnalyticsTRI(PortfolioAnalyticsBase):
    """Portfolio Analytics with total return index abstract base class

    Args:
        tri (pandas.DataFrame or pandas.Series):

    """

    def __init__(self, port, bench=None, risk_free=None):
        self._port = port
        self._bench = bench

        # active return against benchmark
        if bench is None:
            self._active = None
        else:
            active_ret = port.ret - bench.ret
            active_ret.name = pc.ACTIVE
            self._active = pdata.PortfolioTRI(ret=active_ret, initial_value=port.initial_value, initial_date=port.initial_date)

        # excess return against risk-free rate
        if risk_free is None:
            risk_free_rate = pd.Series(1, index=self.port.tri.index, name='risk_free')
        else:
            risk_free_rate = risk_free
        self._risk_free = pdata.PortfolioTRI(tri=risk_free_rate)

        excess_ret = port.ret - self.risk_free.ret
        excess_ret.name = pc.ACTIVE
        self._excess = pdata.PortfolioTRI(ret=excess_ret, initial_value=port.initial_value, initial_date=port.initial_date)

        # base stats output
        self._stats = None

        # rolling stats
        self.rolling_years = [1, 3, 5]
        self._rolling_stats = {y: None for y in self.rolling_years}

        # period returns
        self.periodic_freqs = ['M', 'Q', 'A', 'W']
        self._periodic_returns = {x: None for x in self.periodic_freqs}

        self.run()

    @property
    def port(self):
        return self._port

    @property
    def bench(self):
        return self._bench

    @property
    def active(self):
        return self._active

    @property
    def risk_free(self):
        return self._risk_free

    @property
    def excess(self):
        return self._excess

    @property
    def stats(self):
        if self._stats is None:
            self.run()
        return self._stats

    def run(self):
        self.run_basic_stats()

        self.run_rolling_stats()

        self.run_period_returns()

    def run_basic_stats(self):
        out_stats = dict()

        out_stats[pc.RETURN] = self.port.cagr
        out_stats[pc.VOLATILITY] = self.port.volatility
        out_stats[pc.SHARPE] = self.port.sharpe

        out_stats[pc.RETURN_ACTIVE] = self.active.cagr
        out_stats[pc.TE] = self.active.volatility
        out_stats[pc.IR] = np.divide(self.active.cagr, self.active.volatility)
        self._stats = out_stats

    def compute_rolling_stats(self, yr):
        periods_per_year = self.port.periods_per_year
        window1 = 1 + yr * periods_per_year

        if not isinstance(yr, int):
            window1 = int(np.ceil(window1))
            raise Warning('Input rolling year {.2f} NOT integer, rolling window ceil to {} periods'.format(yr, window1))

        window = window1 - 1  # for return data

        # rolling objects
        rp_tri = self.port.tri.rolling(window1)
        rb_tri = self.bench.tri.rolling(window1)
        ra_tri = self.active.tri.rolling(window1)
        re_tri = self.excess.tri.rolling(window1)

        rp_ret = self.port.ret.rolling(window)  # one less
        rb_ret = self.bench.ret.rolling(window)
        ra_ret = self.active.ret.rolling(window)
        re_ret = self.excess.tri.rolling(window)

        r2_ret = pd.concat([self.port.ret, self.bench.ret], axis=1).rolling(window)

        # rolling metrics
        stats = collections.OrderedDict()

        # returns
        stats[pc.RETURN_PORT] = rp_tri.apply(pu.get_simple_return, raw=False)
        stats[pc.RETURN_BENCH] = rb_tri.apply(pu.get_simple_return, raw=False)
        stats[pc.RETURN_ACTIVE] = ra_tri.apply(pu.get_simple_return, raw=False)  # cumulative alpha
        stats[pc.RETURN_EXCESS] = re_tri.apply(pu.get_simple_return, raw=False)

        # volatility
        const4std = np.sqrt(periods_per_year)
        stats[pc.VOL_PORT] = const4std * rp_ret.std()  # volatility
        stats[pc.VOL_BENCH] = const4std * rb_ret.std()
        stats[pc.TE] = const4std * ra_ret.std()

        stats[pc.VOL_RATIO] = stats[pc.RETURN_PORT] / stats[pc.RETURN_BENCH]

        # beta
        rcov3d = r2_ret.cov().unstack()
        rcov = rcov3d[(self.port.tri.name, self.bench.tri.name)]
        rvar = rcov3d[(self.bench.tri.name, self.bench.tri.name)]
        stats[pc.BETA] = rcov.divide(rvar)

        # risk-adjusted return
        stats[pc.SHARPE] = stats[pc.RETURN_EXCESS].divide(const4std*re_ret.std())  # std of excess return
        stats[pc.IR] = (stats[pc.RETURN_PORT] - stats[pc.RETURN_BENCH]).divide(stats[pc.TE])
        stats[pc.M2] = stats[pc.SHARPE] * stats[pc.VOL_BENCH] + re_ret.mean()

        # remove NaN's
        stats = collections.OrderedDict([(k, v.dropna(axis=0, how='all')) for k, v in stats.items()])

        return stats

    def run_rolling_stats(self):
        self._rolling_stats = {y: self.compute_rolling_stats(y) for y in self._rolling_stats}

    @property
    def rolling1yr(self):
        if self._rolling_stats.get(1, None) is None:
            self.run_rolling_stats()
        self._rolling_stats.get(1, {})

    @property
    def rolling3yr(self):
        if self._rolling_stats.get(3, None) is None:
            self.run_rolling_stats()
        self._rolling_stats.get(3, {})

    @property
    def rolling5yr(self):
        if not self._rolling_stats.get(5, None) is None:
            self.run_rolling_stats()
        self._rolling_stats.get(5, {})

    def run_period_returns(self):
        dict_returns = {x: None for x in self.periodic_freqs}
        for freq in self.periodic_freqs:
            port_rets = pu.get_period_returns(self.port.tri, freq)
            bench_rets = pu.get_period_returns(self.bench.tri, freq)

            ret = pd.DataFrame([port_rets, bench_rets]).transpose()
            ret[pc.ACTIVE] = ret[self.port.tri.name] - ret[self.bench.tri.name]
            dict_returns[freq] = ret  # pandas.DataFrame
        self._periodic_returns = dict_returns

    @property
    def annual_returns(self):
        freq = 'A'
        if self._periodic_returns.get(freq, None) is None:
            self.run_period_returns()
        ret = self._periodic_returns[freq].copy()
        ret.index = ret.index.strftime(pc.FORMAT_ANNUALLY)
        return ret

    @property
    def quarterly_returns(self):
        freq = 'Q'
        if self._periodic_returns.get(freq, None) is None:
            self.run_period_returns()
        ret = self._periodic_returns[freq].copy()
        ret.index = ret.index.strftime(pc.FORMAT_MONTHLY)
        return ret

    @property
    def monthly_returns(self):
        freq = 'M'
        if self._periodic_returns.get(freq, None) is None:
            self.run_period_returns()
        ret = self._periodic_returns[freq].copy()
        ret.index = ret.index.strftime(pc.FORMAT_MONTHLY)
        return ret

    @property
    def weekly_returns(self):
        freq = 'W'
        if self._periodic_returns.get(freq, None) is None:
            self.run_period_returns()
        ret = self._periodic_returns[freq].copy()
        ret.index = ret.index.strftime(pc.FORMAT_WEEKLY)
        return ret


class PortfolioAnalytics(PortfolioAnalyticsBase):
    """Portfolio Analytics with total return index abstract base class

    Args:
        tri (pandas.DataFrame or pandas.Series):

    """

    def __init__(self, tri=None, ret=None, bench=None, bench_ret=None, weight=None):
        self._port = None
        self._bench = None

        # self._active = self._port - self._bench






