"""Portfolio analytics"""

import abc
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

    def __init__(self, port, bench=None):
        self._port = port
        self._bench = bench

        if bench is None:
            self._active = None
        else:
            active_ret = port.ret - bench.ret
            # active_ret.name = '{}-{}'.format(port.ret.name, bench.ret.name)
            self._active = pdata.PortfolioTRI(ret=active_ret, initial_value=port.initial_value, initial_date=port.initial_date)

        self._stats = None
        self.rolling_stats = None
        self.rolling_metrics = None

        self.periodic_freqs = ['M', 'Q', 'A', 'W']
        self.periodic_returns = {x: None for x in self.periodic_freqs}

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
    def stats(self):
        if self._stats is None:
            self.run()
        return self._stats

    def run(self):
        """Run portfolio analytics"""

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

    def run_rolling_stats(self):
        r_stats = dict()

        periods_per_year = int(np.ceil(self.port.periods_per_year))

        data = self.port.tri

        r_stats[pc.ROLLING_RETURN_1YR] = pu.rolling_cagr(data, periods_per_year)
        r_stats[pc.ROLLING_RETURN_3YR] = pu.rolling_cagr(data, 3*periods_per_year)
        r_stats[pc.ROLLING_RETURN_5YR] = pu.rolling_cagr(data, 5*periods_per_year)

        r_stats[pc.ROLLING_VOLATILITY_1YR] = pu.rolling_realised_vol(data, periods_per_year)
        r_stats[pc.ROLLING_VOLATILITY_3YR] = pu.rolling_realised_vol(data, 3*periods_per_year)
        r_stats[pc.ROLLING_VOLATILITY_5YR] = pu.rolling_realised_vol(data, 5*periods_per_year)

        r_stats[pc.ROLLING_SHARPE_1YR] = pu.rolling_sharpe(data, periods_per_year)
        r_stats[pc.ROLLING_SHARPE_3YR] = pu.rolling_sharpe(data, 3*periods_per_year)
        r_stats[pc.ROLLING_SHARPE_5YR] = pu.rolling_sharpe(data, 5*periods_per_year)

        if self.bench is not None:
            data = self.active.tri

            r_stats[pc.ROLLING_ACTIVE_RETURN_1YR] = pu.rolling_cagr(data, periods_per_year)
            r_stats[pc.ROLLING_ACTIVE_RETURN_3YR] = pu.rolling_cagr(data, 3*periods_per_year)
            r_stats[pc.ROLLING_ACTIVE_RETURN_5YR] = pu.rolling_cagr(data, 5*periods_per_year)

            r_stats[pc.ROLLING_ACTIVE_VOLATILITY_1YR] = pu.rolling_realised_vol(data, periods_per_year)
            r_stats[pc.ROLLING_ACTIVE_VOLATILITY_3YR] = pu.rolling_realised_vol(data, 3*periods_per_year)
            r_stats[pc.ROLLING_ACTIVE_VOLATILITY_5YR] = pu.rolling_realised_vol(data, 5*periods_per_year)

            r_stats[pc.ROLLING_ACTIVE_SHARPE_1YR] = pu.rolling_sharpe(data, periods_per_year)
            r_stats[pc.ROLLING_ACTIVE_SHARPE_3YR] = pu.rolling_sharpe(data, 3*periods_per_year)
            r_stats[pc.ROLLING_ACTIVE_SHARPE_5YR] = pu.rolling_sharpe(data, 5*periods_per_year)

            data = self.bench.tri

            r_stats[pc.ROLLING_BENCH_RETURN_1YR] = pu.rolling_cagr(data, periods_per_year)
            r_stats[pc.ROLLING_BENCH_RETURN_3YR] = pu.rolling_cagr(data, 3*periods_per_year)
            r_stats[pc.ROLLING_BENCH_RETURN_5YR] = pu.rolling_cagr(data, 5*periods_per_year)

            r_stats[pc.ROLLING_BENCH_VOLATILITY_1YR] = pu.rolling_realised_vol(data, periods_per_year)
            r_stats[pc.ROLLING_BENCH_VOLATILITY_3YR] = pu.rolling_realised_vol(data, 3*periods_per_year)
            r_stats[pc.ROLLING_BENCH_VOLATILITY_5YR] = pu.rolling_realised_vol(data, 5*periods_per_year)

            r_stats[pc.ROLLING_BENCH_SHARPE_1YR] = pu.rolling_sharpe(data, periods_per_year)
            r_stats[pc.ROLLING_BENCH_SHARPE_3YR] = pu.rolling_sharpe(data, 3*periods_per_year)
            r_stats[pc.ROLLING_BENCH_SHARPE_5YR] = pu.rolling_sharpe(data, 5*periods_per_year)

        # r_stats = {k: v.dropna() for k, v in r_stats.items()}

        self.rolling_stats = r_stats
        self.rolling_metrics = pd.DataFrame(r_stats)

    def run_period_returns(self):
        dict_returns = {x: None for x in self.periodic_freqs}
        for freq in self.periodic_freqs:
            port_rets = pu.get_period_returns(self.port.tri, freq)
            bench_rets = pu.get_period_returns(self.bench.tri, freq)

            ret = pd.DataFrame([port_rets, bench_rets]).transpose()
            ret[pc.ACTIVE] = ret[self.port.tri.name] - ret[self.bench.tri.name]
            dict_returns[freq] = ret  # pandas.DataFrame
        self.periodic_returns = dict_returns

    @property
    def annual_returns(self):
        ret = self.periodic_returns['A'].copy()
        ret.index = ret.index.strftime(pc.FORMAT_ANNUALLY)
        return ret

    @property
    def quarterly_returns(self):
        ret = self.periodic_returns['Q'].copy()
        ret.index = ret.index.strftime(pc.FORMAT_MONTHLY)
        return ret

    @property
    def monthly_returns(self):
        ret = self.periodic_returns['M'].copy()
        ret.index = ret.index.strftime(pc.FORMAT_MONTHLY)
        return ret

    @property
    def weekly_returns(self):
        ret = self.periodic_returns['W'].copy()
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






