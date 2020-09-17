"""Portfolio data """

import abc
import logging
import pandas as pd


logger = logging.getLogger(__name__)


__all__ = [
    'PortfolioDataBase',
    'PortfolioDataTRI'
]


DEFAULT_INITIAL_VALUE = 1


class PortfolioDataBase(metaclass=abc.ABCMeta):
    """Portfolio Data abstract base class
    """

    @property
    def tri(self):
        """pandas.DataFrame: total return index
        """

    @property
    def ret(self):
        """pandas.DataFrame: time-series total returns"""


class PortfolioDataTRI(PortfolioDataBase):
    """Portfolio data with total return index

    Args:
        tri (pandas.DataFrame or pandas.Series): total return index
        ret (pandas.DataFrame or pandas.Series): total returns
    """

    def __init__(self, tri=None, ret=None, start_date=None, end_date=None, **kwargs):
        self._tri = None
        self._ret = None
        self._ratio = None

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

    @property
    def ratio(self):
        return self._ratio

    def _process_input_tri(self, tri):
        # assuming initial value is not NaN
        self._tri = tri
        self._ret = tri.pct_change().dropna()
        self._ratio = 1 + self._ret
        self.initial_value = tri.iloc(0)
        self.initial_date = tri.index(0)

    def _process_input_ret(self, ret, initial_date=None, initial_value=DEFAULT_INITIAL_VALUE):
        self._ret = ret
        # append first row
        self._tri = self._ratio.cumprod()

    def _check_tri_and_ret(self, in_ret):
        # TODO: cross-check returns implied by input tri and input time-series returns
        # use pandas.DataFrame.diff()
        pass
