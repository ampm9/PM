import datetime

import pandas as pd


def compute_holding_period(shares):
    """Compute a security average holding period"""
    if isinstance(shares, pd.DataFrame):
        return shares.apply(compute_holding_period) # by column

    if not isinstance(shares, pd.Series):
        raise TypeError('Invalid input type: {}'.format(type(shares)))

    if any(shares < 0):
        raise ValueError('Number of shares should be non-negative')

    shares_diff = shares - shares.shift()
    shares2sell = shares_diff[shares_diff < 0]  # negative values

    shares_on_hold = shares.loc[shares.last_valid_index()]
    total_shares2sell = - shares2sell.sum() + shares_on_hold

    return shares.sum() / total_shares2sell


def compute_holding_period_loop(shares):
    """Compute a security average holding period by looping each sell action. """
    if isinstance(shares, pd.DataFrame):
        return shares.apply(compute_holding_period_loop)  # by column

    if not isinstance(shares, pd.Series):
        raise TypeError('Invalid input type: {}'.format(type(shares)))

    if any(shares < 0):
        raise ValueError('Number of shares should be non-negative')

    s = shares.copy()  # s will be modified within the loop
    s_delta = s - s.shift()

    extra_date = s.index.max() + datetime.timedelta(days=1)

    delta_dates = s_delta[s_delta < 0].index
    rec_dates = list(delta_dates)
    rec_dates.append(extra_date)
    rec = pd.DataFrame(columns=['sum', 'max'], index=rec_dates)

    for dt in delta_dates:
        dt_val = s.loc[dt]

        # scrape upper area above dt_val
        idx = s.index[s.index < dt]
        area = s[idx] - dt_val
        area = area[area > 0]
        idx = area.index

        # update temporary s
        s[idx] = dt_val

        # record information
        rec.loc[dt, 'sum'] = area.sum()
        rec.loc[dt, 'max'] = area.max()

    rec.loc[extra_date, 'sum'] = s.sum()
    rec.loc[extra_date, 'max'] = s.max()

    return rec['sum'].sum() / rec['max'].sum()


def compute_holding_period_hbar(shares):
    """Compute a security average holding period by top-down approach of horizontal bars"""
    if isinstance(shares, pd.DataFrame):
        return shares.apply(compute_holding_period_hbar)  # by column

    if not isinstance(shares, pd.Series):
        raise TypeError('Invalid input type: {}'.format(type(shares)))

    if any(shares < 0):
        raise ValueError('Number of shares should be non-negative')

    s = shares.copy()

    values = s.unique().tolist()
    if min(values) > 0:
        values.append(0)
    values.sort(reverse=True)

    all_bars = []
    for i, upper in enumerate(values[:-1]):
        lower = values[i+1]
        gap = upper - lower
        assert gap > 0

        ind = s > lower
        ind_bars = ind2series_list(ind)
        gap_bars = [pd.Series(gap, index=b.index) for b in ind_bars]
        all_bars.extend(gap_bars)

        s[ind] = lower  # update s

    y_vals = [b.iloc[0] for b in all_bars]
    x_vals = [len(b.index) for b in all_bars]

    return sum([y*x for y, x in zip(y_vals, x_vals)]) / sum(y_vals)


def ind2series_list(s):
    """Change boolean indicator series to list of sub-series with continuous True-values"""
    if isinstance(s, pd.DataFrame):
        out_dict = {c: [] for c in s.columns}
        for c in s.columns:
            out_dict[c] = ind2series_list(s[c])
        return out_dict

    if not isinstance(s, pd.Series):
        raise TypeError('Invalid input type: {}'.format(type(s)))

    if s.empty:  # empty in
        return []  # empty out

    if s.dtype.name != 'bool':
        s = s.astype('bool')
        raise Warning('Input series not boolean type, coerce input to boolean type')

    ind = s.copy()
    ind_next = ind.shift(-1)
    ind_next.iloc[-1] = False
    s_last_on = ind & (~ind_next)
    s_last_on = s_last_on[s_last_on]
    list_last_index = s_last_on.index

    out_list = [None] * len(list_last_index)

    for i, last_idx in enumerate(list_last_index):
        sub_ind = ind.loc[ind.index <= last_idx]
        sub_ind = sub_ind[sub_ind]
        assert sub_ind.all()
        ind[sub_ind.index] = False  # mark the recorded sub-series
        out_list[i] = sub_ind

    return out_list
