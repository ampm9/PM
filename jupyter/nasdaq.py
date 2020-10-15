import os

import pandas as pd

import pm.PM.portfolio.utilities as pu
import pm.PM.portfolio.data as pdata


COL_DATE = 'Date'
COL_CLOSE = 'Close/Last'


def read_csv_nasdaq(tickers, col_date=COL_DATE, col_close=COL_CLOSE):
    csv_dir = os.path.join('D:', 'datasets', 'nasdaq')
    dict_data = {x: None for x in tickers}

    for code in tickers:
        csv_name = '{}.csv'.format(code)
        csv_path = os.path.join(csv_dir, csv_name)

        df = pd.read_csv(csv_path, index_col=col_date, parse_dates=True)
        df.rename(columns = {x: x.strip() for x in df.columns}, inplace=True)
        ts = df[col_close]
        ts.name = code
        dict_data[code] = ts

    df = pd.DataFrame(dict_data)
    df.sort_index(axis=0, inplace=True)
    return df

tickers = ['SPX', 'VOOG', 'VOOV']

df = read_csv_nasdaq(tickers)

pd.concat([df.head(2), df.tail(2)])


(name, bench_name) = ('VOOV', 'SPX')

port = pdata.PortfolioTRI(ret=df[name].pct_change().dropna())
bench = pdata.PortfolioTRI(ret=df[bench_name].pct_change().dropna())


