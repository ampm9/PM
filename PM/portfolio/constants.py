

# Date formats
FORMAT_DAILY = '%Y/%m/%d'
FORMAT_MONTHLY = '%Y/%m'
FORMAT_ANNUALLY = '%Y'

FORT_DATE_FREQ = {
    'A': FORMAT_ANNUALLY,
    'Q': FORMAT_MONTHLY,
    'M': FORMAT_MONTHLY,
    'W': FORMAT_DAILY,
    'D': FORMAT_DAILY
}


# Date relevant
DAYS_PER_YEAR = 252
DAYS_PER_YEAR_GB = 260
DAYS_PER_YEAR_365 = 365.25

DAYS_PER_MONTH = 21
DAYS_PER_QUARTER = 63

MONTHS_PER_YEAR = 12
QUARTERS_PER_YEAR = 4


# Portfolio relevant
PORT = 'portfolio'
BENCH = 'benchmark'
ACTIVE = 'active'
EXCESS = 'excess'
RISK_FREE = 'risk_free'

DEFAULT_TRI_INITIAL_VALUE = 1


# Analytic Stats
RETURN = 'return'
VOLATILITY = 'volatility'
SHARPE = 'sharpe'
CAGR = 'cagr'

RETURN_ACTIVE = 'return_active'
RETURN_EXCESS = 'return_excess'

# Benchmark relevant metrics
BETA = 'beta'
TE = 'te'
IR = 'ir'
M2 = 'm2'

# Rolling Metrics, some are same as regular metrics
ROLLING_PREFIX = 'rolling'

RETURN_PORT = 'return_port'
RETURN_BENCH = 'return_bench'

VOL_PORT = 'vol_port'
VOL_BENCH = 'vol_bench'
VOL_RATIO = 'vol_ratio'


