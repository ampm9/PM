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

        # base stats output
        self._stats = None

        # rolling stats
        self.rolling_years = [1, 3, 5]
        self._rolling_stats = {y: None for y in self.rolling_years}

        self._efficiency_scores = None

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
        return self.port.risk_free_tri

    @property
    def excess(self):
        return self.port.excess_tri

    @property
    def bench_excess(self):
        return self.bench.excess_tri

    @property
    def stats(self):
        if self._stats is None:
            self.run()
        return self._stats

    def run(self):
        self.run_basic_stats()

        self.run_rolling_stats()

        self.run_period_returns()

        self.run_efficiency_scores(yr=3)

    def run_basic_stats(self):
        out_stats = dict()

        out_stats[pc.CAGR] = self.port.cagr
        out_stats[pc.VOLATILITY] = self.port.volatility
        out_stats[pc.SHARPE] = self.port.sharpe

        out_stats[pc.CAGR_BENCH] = self.bench.cagr
        out_stats[pc.VOL_BENCH] = self.bench.volatility
        out_stats[pc.SHARPE_BENCH] = self.bench.sharpe

        out_stats[pc.CAGR_ACTIVE] = self.active.cagr
        out_stats[pc.TE] = self.active.volatility
        out_stats[pc.IR] = np.divide(self.active.cagr, self.active.volatility)

        out_stats[pc.M2] = out_stats[pc.SHARPE] * out_stats[pc.VOL_BENCH] + self.port.risk_free_rate.mean()
        self._stats = out_stats

    def compute_rolling_stats(self, yr):
        periods_per_year = self.port.periods_per_year
        window = yr * periods_per_year
        window1 = window + 1

        # rolling objects
        rp_tri = self.port.tri.rolling(window1)
        rb_tri = self.bench.tri.rolling(window1)
        ra_tri = self.active.tri.rolling(window1)
        re_tri = self.port.excess_tri.rolling(window1)
        rbe_tri = self.bench.excess_tri.rolling(window1)

        rp_ret = self.port.ret.rolling(window)  # one less
        rb_ret = self.bench.ret.rolling(window)
        ra_ret = self.active.ret.rolling(window)
        re_ret = self.port.excess_ret.rolling(window)
        rbe_ret = self.bench.excess_ret.rolling(window)
        rf_rate = self.port.risk_free_rate.rolling(window)

        r2_ret = pd.concat([self.port.ret, self.bench.ret], axis=1).rolling(window)

        # rolling metrics
        stats = collections.OrderedDict()

        # returns
        # stats[pc.RETURN_PORT] = rp_tri.apply(pu.get_simple_return, raw=False)
        # stats[pc.RETURN_BENCH] = rb_tri.apply(pu.get_simple_return, raw=False)
        # stats[pc.RETURN_ACTIVE] = ra_tri.apply(pu.get_simple_return, raw=False)  # cumulative alpha
        # stats[pc.RETURN_EXCESS] = re_tri.apply(pu.get_simple_return, raw=False)

        # returns
        stats[pc.CAGR_PORT] = rp_tri.apply(pu.get_simple_cagr, raw=False)
        stats[pc.CAGR_BENCH] = rb_tri.apply(pu.get_simple_cagr, raw=False)
        stats[pc.CAGR_ACTIVE] = ra_tri.apply(pu.get_simple_cagr, raw=False)  # cumulative alpha
        stats[pc.CAGR_EXCESS] = re_tri.apply(pu.get_simple_cagr, raw=False)
        stats[pc.CAGR_BENCH_EXCESS] = rbe_tri.apply(pu.get_simple_cagr, raw=False)

        # volatility
        const4std = np.sqrt(periods_per_year)
        stats[pc.VOL_PORT] = const4std * rp_ret.std()  # volatility
        stats[pc.VOL_BENCH] = const4std * rb_ret.std()
        stats[pc.TE] = const4std * ra_ret.std()

        stats[pc.VOL_RATIO] = stats[pc.VOL_PORT] / stats[pc.VOL_BENCH]

        # beta
        rcov3d = r2_ret.cov().unstack()
        rcov = rcov3d[(self.port.name, self.bench.name)]
        rvar = rcov3d[(self.bench.name, self.bench.name)]
        stats[pc.BETA] = rcov.divide(rvar)

        # risk-adjusted return
        stats[pc.SHARPE_PORT] = stats[pc.CAGR_EXCESS].divide(const4std * re_ret.std())  # std of excess return
        stats[pc.SHARPE_BENCH] = stats[pc.CAGR_BENCH_EXCESS].divide(const4std * rbe_ret.std())  # std of excess return
        stats['sharpe_difference'] = stats[pc.SHARPE_PORT] - stats[pc.SHARPE_BENCH]

        stats[pc.IR] = (stats[pc.CAGR_ACTIVE]).divide(stats[pc.TE])
        stats[pc.M2] = stats[pc.SHARPE_PORT] * stats[pc.VOL_BENCH] + rf_rate.mean()

        # remove NaN's
        stats = collections.OrderedDict([(k, v.dropna(axis=0, how='all')) for k, v in stats.items()])

        return stats

    def run_rolling_stats(self):
        self._rolling_stats = {y: self.compute_rolling_stats(y) for y in self.rolling_years}

    @property
    def rolling_stats(self):
        if self._rolling_stats is None:
            self.run_rolling_stats()
        return self._rolling_stats

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

    @property
    def efficiency_scores(self):
        if self._efficiency_scores is None:
            self.run_efficiency_scores(yr=3)
        return self._efficiency_scores

    def run_efficiency_scores(self, yr=3):
        window = yr * pc.MONTHS_PER_YEAR
        roll_m_ret = self.monthly_returns.rolling(window)

        p_name = self.port.name
        b_name = self.bench.name
        df = self.monthly_returns.copy()

        eff_cols = ['batting_average', 'upside_capture_ratio', 'downside_capture_ratio', 'upside_downside_ratio']
        eff_scores = pd.DataFrame(np.nan, index=df.index, columns=eff_cols)

        for e in range(window - 1, len(df)):
            sub_df = df.iloc[1 + e - window:1 + e]  # exclude end_point
            sub_sc = pd.Series(index=eff_cols)
            sub_sc['batting_average'] = pu.get_return_batting_average(sub_df, p_name, b_name)
            sub_sc['upside_capture_ratio'] = pu.get_upside_capture_ratio(sub_df, p_name, b_name)
            sub_sc['downside_capture_ratio'] = pu.get_downside_capture_ratio(sub_df, p_name, b_name)
            sub_sc['upside_downside_ratio'] = np.divide(sub_sc['upside_capture_ratio'],
                                                        sub_sc['downside_capture_ratio'])
            eff_scores.iloc[e] = sub_sc

        self._efficiency_scores = eff_scores.dropna(how='all')
        return


class PortfolioAnalytics(PortfolioAnalyticsBase):
    """Portfolio Analytics with total return index abstract base class

    Args:
        tri (pandas.DataFrame or pandas.Series):

    """

    def __init__(self, tri=None, ret=None, bench=None, bench_ret=None, weight=None):
        self._port = None
        self._bench = None

        # self._active = self._port - self._bench






