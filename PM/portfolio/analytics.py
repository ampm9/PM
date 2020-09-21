"""Portfolio analytics"""

import abc
import logging

import numpy as np
import pandas as pd

from .data import *


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

    def __init__(self, tri=None, bench_tri=None):
        self._port = PortfolioData(tri=tri)
        self._bench = PortfolioData(tri=bench_tri)


class PortfolioAnalytics(PortfolioAnalyticsBase):
    """Portfolio Analytics with total return index abstract base class

    Args:
        tri (pandas.DataFrame or pandas.Series):

    """

    def __init__(self, tri=None, ret=None, bench=None, bench_ret=None, weight=None)
        self._port = None
        self._bench = None

        self._active = self._port - self._bench






