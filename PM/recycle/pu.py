"""Portfolio utilities recycle functions"""



# No need after portfolio data exclude ret input

def process_tri_or_return(tri=None, ret=None, initial_value=None, initial_date=None):
    """Compute total return index or return if either one of them are input;
    If neither TRI and return are not None, check if they represent the same index.

    Args:
        tri(pandas.Series or pandas.DataFrame): Input total return index Time-Series or DataFrame
        ret(pandas.Series or pandas.DataFrame): Input return Time-Series or DataFrame
        initial_date(date): the initial index value to prepend (usually type of datetime)
        initial_value(float): initial value of total return index, defaults to 100.

    Returns:
        (pandas.Series or pandas.DataFrame, pandas.Series or pandas.DataFrame, boolean)
    """
    if tri is None and ret is None:
        return None, None, True

    if tri is not None and ret is None:
        if initial_date is not None:
            raise Warning('Initial_date is only required with return input')
        if initial_value is not None:
            tri = normalise(tri, initial_value=initial_value)
        ret = tri.pct_change()
        return tri, ret, True

    if tri is None and ret is not None:
        tri = return2tri(ret, initial_value=initial_value, initial_date=initial_date)
        return tri, ret, True

    if tri is not None and ret is not None:
        if all(isinstance(x, pd.Series) for x in [tri, ret]):
            if tri.empty or ret.empty:
                return tri, ret, tri.empty and ret.empty

            ret2 = tri.pct_change()
            if len(ret.index) != len(ret2.index):
                raise Warning('Input TRI and return data index length mismatch ')
                return tri, ret, False
            is_equal = np.isclose(ret.iloc[1:].to_numpy(), ret2.iloc[1:].to_numpy())
            return tri, ret, all(is_equal)

        elif all(isinstance(x, pd.DataFrame) for x in [tri, ret]):
            if len(tri.columns.symmetric_difference(ret.columns)) > 1:
                raise Warning('Input TRI and return DataFrame columns mismatch')
                return tri, ret, False

            is_equal_dict = {x: False for x in tri.columns}
            for c in tri.columns:
                _, _, is_equal = process_tri_or_return(tri=tri[c], ret=ret[c])
                is_equal_dict[c] = is_equal
            is_equal = all(is_equal_dict.values())
            return tri, ret, False

        else:
            raise Warning('Input TRI and return type mismatch ')
            return tri, ret, False


