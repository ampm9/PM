

# Date formats
FORMAT_DATE = '%Y/%m/%d'

FORMAT_DAILY = FORMAT_DATE
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
BENCH_EXCESS = 'bench_excess'
RISK_FREE = 'risk_free'

DEFAULT_TRI_INITIAL_VALUE = 1


# Analytic Stats
RETURN = 'return'
RETURN_ACTIVE = 'return_active'
RETURN_EXCESS = 'return_excess'
RETURN_EXCESS = 'return_bench_excess'

CAGR = 'cagr'
CAGR_ACTIVE = 'cagr_active'
CAGR_EXCESS = 'cagr_excess'
CAGR_BENCH_EXCESS = 'cagr_bench_excess'

VOLATILITY = 'volatility'
VOLATILITY_EXCESS = 'volatility_excess'
SHARPE = 'sharpe'

# Benchmark relevant metrics
BETA = 'beta'
TE = 'te'
IR = 'ir'
M2 = 'm2'

# Rolling Metrics, some are same as regular metrics
ROLLING_PREFIX = 'rolling'

CAGR_PORT = 'cagr_port'
CAGR_BENCH = 'cagr_bench'

VOL_PORT = 'vol_port'
VOL_BENCH = 'vol_bench'
VOL_RATIO = 'vol_ratio'

SHARPE_PORT = 'sharpe_port'
SHARPE_BENCH = 'sharpe_bench'


