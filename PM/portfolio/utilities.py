"""Utility functions for portfolio analytics
"""

import itertools
import logging
import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# normalize function, check first value