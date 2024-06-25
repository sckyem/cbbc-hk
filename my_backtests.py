import numpy as np
import pandas as pd
from my_script import *
import pandas_ta as ta

def cleaning(y, x):
    if isinstance(y, pd.Series) and isinstance(x, pd.Series):
        df = pd.concat([y, x], axis=1).dropna()
        return {'df': df, 'price':df[y.name], 'indicator':df[x.name], 'price_name': y.name, 'indicator_name': x.name}

def get_zscore(series, window=0):
    if window:
        return series.rolling(window).apply(  lambda x: (x.iloc[-1] - x.mean()) / x.std() if len(x) == window else np.nan  )
    else: 
        return (series - series.mean()) / series.std()

def get_minmax(series, window=0):
    if window:
        return series.rolling(window).apply(  lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min())   )
    else:
        return (series - series.min()) / (series.max() - series.min())

def get_ohlcefficency(ohlc, window=10):
    df = ohlc.copy()
    df.columns = [str(i).lower() for i in df.columns]
    df['log_return'] = np.log(df['close'] / df['close'].shift())
    df['cp'] = df['close'] - df['open']
    df['hl'] = df['high'] - df['low']
    df['ratio'] = df['cp'] / df['hl']
    return df['ratio'].rolling(window).mean().rename('ohlcefficency')

def get_ohlcefficency2(ohlc, window=10):
    df = ohlc.copy()
    df.columns = [str(i).lower() for i in df.columns]
    df['log_return'] = np.log(df['close'] / df['close'].shift())
    df['log_return_abs_sum'] = df['log_return'].abs().rolling(window).sum()
    df['log_return_long'] = np.log(df['close'] / df['close'].shift(window))
    df['ohlcefficency'] = df['log_return_long'] / df['log_return_abs_sum']
    return df['ohlcefficency']

def get_adx(ohlc, window=10):
    df = ohlc.copy() 
    df.columns = [str(i).lower() for i in df.columns]
    return ta.adx(df['high'], df['low'], df['close'], window)

def get_kama(close, window=10):
    return ta.kama(close, window)

def get_percentb(series, window=20):
    return (series.iloc[-1] - series.rolling(window).min()) / (series.rolling(window).max() - series.rolling(window).min()   )

def get_shift_updowndiff(price_series, periods):
    df = price_series.copy().shift()
    if isinstance(df, pd.DataFrame):
        if len(df.columns) > 3:
            df = df[df.columns[3]]
        else:            
            df = df[df.columns[0]]            
    if isinstance(df, pd.Series):        
        df = df.to_frame()
    
    df['log_return'] = np.log(df / df.shift())
    results = []
    for j in periods:
        df['log_return_abs'] = df['log_return'].abs()
        df['up'] = np.where(df['log_return'] > 0, df['log_return_abs'], 0)
        df['down'] = np.where(df['log_return'] < 0, df['log_return_abs'], 0)
        df[['up', 'down']] = df[['up', 'down']].rolling(j).sum()
        df[j] = df['up'] - df['down']
        results.append(  df[j]  )
    results = pd.concat(results, axis=1)
    results.columns.name = 'updowndiff_ma'
    return results

def classify_only(indicators, shift_period=1, threshold_is_mean_or_median=''):
    df = indicators.copy()
    if isinstance(df, pd.Series):
        name = [df.name]
        df = df.to_frame()
    else:
        name = df.columns.names

    if isinstance(df, pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            pass
        else:            
            df.columns.names = name

    if shift_period:
        df = df.shift(shift_period)

    if threshold_is_mean_or_median == 'mean':
        df = (df > df.mean()).astype(int)
    elif threshold_is_mean_or_median == 'median':
        df = (df > df.median()).astype(int)
    else:
        df = (df > 0).astype(int)
    return df

def classify_by_double_ma(indicators, fast_periods=0, slow_periods=[], shift_period=1):
    df = indicators.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if isinstance(fast_periods, int):
        fast_periods = [fast_periods]
    
    if isinstance(slow_periods, int):
        slow_periods = [slow_periods]
    
    if shift_period:
        df = df.shift(shift_period)

    results = []    
    for fast in fast_periods:
        for slow in slow_periods:
            if fast < slow:
                for i in df.columns:
                    if len(df.columns) > 1:
                        name = (i, fast, slow)
                    else:
                        name = (fast, slow)
                    
                    if fast == 0:
                        results.append(  (df[i] - df[i].rolling(slow).mean()).rename(name)  )
                    else:
                        results.append(  (df[i].rolling(fast).mean() - df[i].rolling(slow).mean()).rename(name)  )
    results = pd.concat(results, axis=1).dropna()

    if not results.empty:
        if len(df.columns) > 1:
            names = ['indicator', 'fast_period', 'slow_period']
        else:
            names = ['fast_period', 'slow_period']
        results.columns = pd.MultiIndex.from_tuples(  results.columns, names=names  )        
        return results.map(lambda x: 1 if x > 0 else 0)

def get_shift_cumsum(indicators):
    df = indicators.copy().cumsum()
    if isinstance(df, pd.Series):
        name = [df.name]
        df = df.to_frame()
    else:
        name = df.columns.names
    if isinstance(df, pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            pass
        else:            
            df.columns.names = name
    return df

def get_shift_count_positive_pct(indicators, periods):
    df = indicators.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    results = []
    for i in df.columns:
        for j in periods:
            if len(df.columns) > 1:
                name = (i, j)
            else:
                name = j
            results.append(  df[i].pct_change().rolling(j).apply(lambda x: (x > 0).sum(), raw=True).shift().rename(  name  )  )
            
    results = pd.concat(results, axis=1)

    if len(df.columns) > 1:
        results.columns = pd.MultiIndex.from_tuples(  results.columns, names=['indicator', 'count_positive_pct']  )
    else:
        results.columns.name = 'count_positive_pct'
    return results

def get_shift_count_of_consecutive_positive_pct(indicators):
    df = indicators.copy()
    mask = df > 0
    df = mask.groupby((mask != mask.shift()).cumsum()).cumsum().shift()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if isinstance(df, pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            pass
        else:
            df.columns.names = ['shift']
    return df            

def get_shift_rolling_sum(indicators, periods):
    df = indicators.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    results = []
    for i in df.columns:
        for j in periods:
            if len(df.columns) > 1:
                name = (i, j)
            else:
                name = j
            results.append(  df[i].rolling(  window=j  ).sum().shift().rename(  name  )  )
            
    results = pd.concat(results, axis=1)

    if len(df.columns) > 1:
        results.columns = pd.MultiIndex.from_tuples(  results.columns, names=['indicator', 'sum_period']  )
    else:
        results.columns.name = 'sum_period'
    return results

def get_shift_pct(indicators, periods):
    df = indicators.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    results = []
    for i in df.columns:
        for j in periods:
            if len(df.columns) > 1:
                name = (i, j)
            else:
                name = j
            results.append(  df[i].pct_change(j).shift().rename(  name  )  )
    
    results = pd.concat(results, axis=1)

    if len(df.columns) > 1:
        results.columns = pd.MultiIndex.from_tuples(  results.columns, names=['indicator', 'pct']  )
    else:
        results.columns.name = 'pct'
    return results

def get_shift_pct_ma(indicators, periods):
    df = indicators.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    results = []
    for i in df.columns:
        for j in periods:
            if len(df.columns) > 1:
                name = (i, j)
            else:
                name = j
            results.append(  df[i].pct_change().rolling(  window=j  ).mean().shift().rename(  name  )  )
            
    results = pd.concat(results, axis=1)

    if len(df.columns) > 1:
        results.columns = pd.MultiIndex.from_tuples(  results.columns, names=['indicator', 'pct_ma_period']  )
    else:
        results.columns.name = 'pct_ma_period'
    return results

def get_shift_minmax(indicators, periods):
    dfs = []
    for period in periods:
        df = get_minmax(indicators, period)
        dfs.append(  df.shift().rename(  period  )  )
    df = pd.concat(dfs, axis=1)
    df.columns.names = ['minmax']
    return df

def get_shift_zscore(indicators, periods=[0]):
    dfs = []
    for period in periods:
        df = get_zscore(indicators, period)
        
        dfs.append(  df.shift().rename(  period  )  )
    df = pd.concat(dfs, axis=1).dropna()
    df = df.applymap(lambda x: math.ceil(x) if x > 0 else math.floor(x))
    df.columns.names = ['zscore']
    return df

def get_signals(indicators, step=0, is_gt_threshold=True):
    df = indicators.copy().fillna(0)
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if not step:
        step = (df.max().max() - df.min().min()) / 20

    round_digi = -int(math.log10(get_the_last_position(step)))
    min_threshold = math.floor(df.min().min() * 10**round_digi) / 10**round_digi
    max_threshold = math.ceil(df.max().max() * 10**round_digi) / 10**round_digi
    thresholds = [  round(i, round_digi) for i in np.arange(min_threshold, max_threshold, step)]

    results = []
    for i in df.columns:
        for j in thresholds:

            if len(df.columns) > 1:
                if len(thresholds) > 1:
                    name = (i, j)
                else:
                    name = i
            else:
                if len(thresholds) > 1:
                    name = j
                else:
                    name = i
            
            if is_gt_threshold:
                result = ((df[i] > j) & (df[i].shift() < j))
            else:
                result = ((df[i] < j) & (df[i].shift() > j))

            results.append(  result.astype(int).rename(name)  )
    results = pd.concat(results, axis=1)

    if len(df.columns) > 1:
        if len(thresholds) > 1:
            results.columns = pd.MultiIndex.from_tuples(  results.columns, names=[df.columns.name, 'threshold']  )
        else:
            results.columns.name = df.columns.name
    else:
        if len(thresholds) > 1:
            results.columns.name = 'threshold'
        else:
            results.columns.name = df.columns.name
    return results

def get_extended_signals(signals, holding_periods):
    df = signals.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()

    results = []
    for i in df.columns:
        for j in holding_periods:
            if len(df.columns) > 1:
                if isinstance(i, tuple):
                    name = (*i, j)
                else:
                    name = (i, j)     
            else:
                name = j
            results.append(  df[i].replace(0, np.nan).ffill(limit=j).replace(np.nan, 0).rename(  name  )  )
    results = pd.concat(results, axis=1)

    if len(df.columns) > 1:        
        if isinstance(df.columns, pd.MultiIndex):
            names = [*df.columns.names, 'holding_period']
        else:
            names = [df.columns.name, 'holding_period']
        results.columns = pd.MultiIndex.from_tuples(  results.columns, names=names  )
    else:
        results.columns.name = 'holding_period'
    return results

def get_volume_profile(ohlcv, window=365, is_summary=True):

    start = ohlcv.index[0]
    end = ohlcv.index[-1]
    h = ohlcv.columns[1]
    l = ohlcv.columns[2]
    v = ohlcv.columns[4]
    dfs = []

    sub_start = start
    while sub_start <= end - datetime.timedelta(days=window):

        sub_end = sub_start + datetime.timedelta(days=window)
        sub_ohlcv = ohlcv.loc[sub_start:sub_end]
        sub_max = sub_ohlcv[h].max()
        sub_min = sub_ohlcv[l].min()

        prices = np.linspace(sub_min, sub_max, num=100)
        prices = np.round(prices, 2)
        
        volumes = np.zeros_like(prices)
    
        for date, row in sub_ohlcv.iterrows():
            levels = np.where((prices >= row[l]) & (prices <= row[h]))
            # Distribute volume evenly across these price levels if levels are found
            if len(levels[0]) > 0:
                volumes[levels] += row[v] / len(levels[0])

        if is_summary:
            mean_price = np.average(prices, weights=volumes)
            variance = np.average((prices - mean_price)**2, weights=volumes)
            volume_profile_sd = np.sqrt(variance)
            dfs.append(  pd.Series(  [mean_price, volume_profile_sd], index=['mean_price', 'volume_profile_sd'], name=date  )  )
        else:
            dfs.append(  pd.Series(  volumes, name=(date, sub_min, sub_max)  )  )

        sub_start += datetime.timedelta(days=1)
        
    if is_summary:
        return pd.concat(dfs, axis=1).T
    else:
        df = pd.concat(dfs, axis=1)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=['date', 'low', 'high'])        
        return df.T

def print_performance(  pnls, anualized_factor=252, is_first_column_benchmark=False  ):
    df = pnls.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()

    if is_first_column_benchmark:
        benchmark = df[df.columns[0]]
        cut = int(len(benchmark) * 0.75)
        train = benchmark.iloc[:cut]
        test = benchmark.iloc[cut:]

        train_positive = train[train > 0].sum()
        train_negative = train[train < 0].sum()
        test_positive = test[test > 0].sum()
        test_negative = test[test < 0].sum()       

    results = []
    for i in df.columns:
        c = df[i]
        num_of_day = len(c)
        trade = np.where(  (c != 0) & (c.shift() == 0), 1, 0  ).sum()

        c = c.replace(0, np.nan)
        c_dropna = c.dropna()

        explorsure = len(c_dropna)
        explorsure_pct = len(c_dropna) / num_of_day

        mean = c_dropna.mean()
        std = c_dropna.std()
        sum = c_dropna.sum()
        sharpe = mean / std * math.sqrt(anualized_factor)

        equity = c_dropna.cumsum().add(1)
        drawdown = equity / equity.cummax() - 1
        mdd = drawdown.min()

        values = [sum, mean, std, sharpe, mdd, explorsure, explorsure_pct, trade]
        indexes = ['sum', 'mean', 'std', 'sharpe', 'mdd', 'explorsure', 'explorsure_pct', 'trade']

        if is_first_column_benchmark:
            this_train = c.iloc[:cut]
            this_test = c.iloc[cut:]

            this_train_positive = this_train[this_train > 0].sum()
            this_train_negative = this_train[this_train < 0].sum()
            this_test_positive = this_test[this_test > 0].sum()
            this_test_negative = this_test[this_test < 0].sum()

            train_positive_rate = this_train_positive / train_positive
            train_negative_rate = this_train_negative / train_negative
            test_positive_rate = this_test_positive / test_positive
            test_negative_rate = this_test_negative / test_negative

            values += [train_positive_rate, train_negative_rate, test_positive_rate, test_negative_rate]
            indexes += ['train_positive_rate', 'train_negative_rate', 'test_positive_rate', 'test_negative_rate']

        results.append(  pd.Series(  values, index=indexes, name=i  )  )
    results = pd.concat(results, axis=1)
    print(results)
    return results

def get_ma(dataframe, period=0, diff_times=0, is_classify=False):
    df = diffs(dataframe, diff_times)
    df = df.rolling(period).mean()
    if is_classify:
        return classify(df)
    else:
        return df

def get_double_ma(dataframe, fast=0, slow=10, diff_times=0, is_classify=False):
    df = diffs(dataframe, diff_times)
    dfs = []
    for i in dataframe.columns:
        fast_df = dataframe[i]
        if fast > 0:
            fast_df = fast_df.rolling(fast).mean()
        df = fast_df - dataframe[i].rolling(slow).mean()
        dfs.append(df)
    df = pd.concat(dfs, axis=1).dropna()
    if is_classify:
        return classify(df)
    else:
        return df

def get_cut(dataframe, num_of_groups=5, rolling_window=0, diff_times=0, is_count_distribution=False):
    df = diffs(dataframe, diff_times)
    def func(x):
        return pd.cut(x, bins=num_of_groups, labels=False)
    df = apply_rolling_func(df, rolling_window, func)
    if is_count_distribution:
        return df.apply(pd.value_counts)
    else:
        return df

def get_rolling_qcut(dataframe, num_of_groups=5, rolling_window=0, diff_times=0):
    df = diffs(dataframe, diff_times)
    def func(x):
        return pd.qcut(x, q=num_of_groups, labels=False)
    return apply_rolling_func(df, rolling_window, func)

def get_rolling_z(dataframe, rolling_window=0, diff_times=0, bins=[-2, -1, 1, 2]):
    df = diffs(dataframe, diff_times)
    def func(x):
        return (x - x.mean()) / x.std()
    df = apply_rolling_func(df, rolling_window, func)
    if bins:
        bins = [-float('inf')] + bins + [float('inf')]
        return pd.cut(df, bins=bins, labels=False)
    else:
        return df

def apply_rolling_func(dataframe, rolling_window, func):
    df = dataframe.copy()
    if rolling_window:
        return df.rolling(rolling_window).apply(lambda x: func(x).iloc[-1]).dropna()
    else:
        return func(df)

def classify(dataframe):
    dfs = []
    for i in dataframe.columns:
        dfs.append(np.sign(dataframe[i]))
    return pd.concat(dfs, axis=1).dropna()

def diffs(dataframe, diff_times=0, is_classify=True):
    df = dataframe.copy()
    if diff_times > 0:
        for i in range(diff_times):
            df = df.diff()
    if is_classify:
        return classify(df)
    else:
        return df

def speed(dataframe, is_classify=True):
    return diffs(dataframe, 1, is_classify)

def acceleration(dataframe, is_classify=True):
    return diffs(dataframe, 2, is_classify)

def sma(dataframe, windows=[10], is_above_sma=False, is_uptrend=False, is_accelerate=False, is_classify=True):
    df = dataframe.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if isinstance(windows, int):
        windows = [windows]    
    dfs = []
    for i in df:
        for w in windows:
            tmp = df[i].rolling(w).mean()
            if is_above_sma:
                if isinstance(i, tuple):
                    name = (*i, f"above_sma{w}")
                else:
                    name = (i, f"above_sma{w}")
                dfs.append(  (df[i] - tmp).rename(  name  )  )
            elif is_uptrend:
                if isinstance(i, tuple):
                    name = (*i, f"uptrend_sma{w}")
                else:
                    name = (i, f"uptrend_sma{w}")
                dfs.append(  tmp.diff().rename(  name  )  )
            elif is_accelerate:
                if isinstance(i, tuple):
                    name = (*i, f"accelerate_sma{w}")
                else:
                    name = (i, f"accelerate_sma{w}")
                dfs.append(  tmp.rename(  name  ).diff().diff()  )
            else:
                if isinstance(i, tuple):
                    name = (*i, f"sma{w}")
                else:
                    name = (i, f"sma{w}")
                dfs.append(tmp)
    dfs = pd.concat(dfs, axis=1).dropna()
    if is_classify:
        return classify(dfs)
    else:
        return dfs

def pnls(returns, indicators, labels={}, is_add_returns=False):    
    if returns is not None and indicators is not None:
        if isinstance(returns, pd.Series):
            returns = returns.to_frame()
        if isinstance(indicators, pd.Series):
            indicators = indicators.to_frame()
        
        levels = list(range(1, len(indicators.columns) + 1))
        pnls = []
        for r in returns:
            df = pd.concat([returns[r], indicators], axis=1).dropna()
            pnl = df.set_index(indicators.columns.to_list(), append=True).unstack(levels, fill_value=0)            
            pnls.append(  pnl  )
            # if len(returns.columns) == 1:
            #     result = pnl
            # else:
            #     pnl['return'] = r
            #     
        if pnls:
        #     result = pd.concat(pnls).set_index('return', append=True).unstack(1, fill_value=0)
            pnls = pd.concat(pnls, axis=1)
            if labels:
                for k,v in labels.items():
                    pnls[k] = v
                    pnls = pnls.set_index(k, append=True).unstack(1, fill_value=0)
            
            if is_add_returns:
                pnls = pd.concat([returns, pnls], axis=1).dropna()
            return pnls


def cumsums(pnls, lowest_pnl=0, to_period='Q'):
    def func(x):
        std = x.std()
        return x.mean() / std if std != 0 else 0
    if isinstance(pnls, pd.DataFrame):
        df = pnls.cumsum()
        index = df.index.to_series().dt.to_period(to_period)
        orders = df.groupby(index).agg(func).mean().abs().sort_values(ascending=False).index
        df = df[orders]
        if lowest_pnl:
            return df.loc[:, df.iloc[-1] > lowest_pnl]
        else:
            return df

if __name__ == '__main__':

    pass