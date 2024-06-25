import yfinance as yf
from my_io import path, read_parquet, write_parquet
from my_script import *
from default_modules import * 

def request_by(self, manu):

    df_list = []        

    for request_func in manu.get('request_funcs'):
        
        df_list.append(  getattr(self, request_func)(manu.get('symbols'))  )
        
    return pd.concat(df_list, axis=1)

def set_ticker(self, symbols ):
        
    try:
        if type(symbols ) == str:
            self.Ticker = yf.Ticker(symbols )

        elif (type(symbols ) == list) and (len(symbols ) == 1):
            self.Ticker = yf.Ticker(symbols [0]).ticker

        elif (type(symbols ) == list) and (len(symbols ) > 1):
            self.Ticker = yf.Tickers(symbols )

        time.sleep(2)
    except Exception as error:
        print(f"Cant set {symbols } Ticker")

def get_history(self, symbols):        
    self.set_ticker(symbols)   
    df = self.Ticker.history(period='10y')
    df.index = df.index.tz_localize(None)
    df = df.rename_axis(None)
    return df.dropna()

def get_close(self, symbols ):
    return self.get_history(symbols )['Close']

def get_close_with_3m10y_yeild_spread(symbols, **args):
    symbols.extend(['^IRX', '^TNX'])
    df = get_close(['QQQ', '^IRX', '^TNX'], **args)
    df['spread_3m10y'] = df['^irx_close'] - df['^tnx_close']
    df = df.drop(columns=['^irx_close', '^tnx_close'], axis=1)
    print_result(df)
    return df

def get_ohlcv(symbol, from_time, to_time, from_time_delta=0, to_time_delta=0, is_adj_close=False, is_drop_hl=False, is_download=True):
    if isinstance(symbol, list):
        file_path = path('yfinance', 'all')
    else:
        file_path = path('yfinance', symbol)

    if not is_download:
        dfs = read_parquet(file_path)
    else:
        from_datetime = datetime.datetime.strptime(from_time, '%Y-%m-%d') if isinstance(from_time, str) else from_time        
        to_datetime = datetime.datetime.strptime(to_time, '%Y-%m-%d') if isinstance(to_time, str) else to_time
        if from_time_delta: from_datetime -= datetime.timedelta(days=from_time_delta)
        if to_time_delta: to_datetime += datetime.timedelta(days=to_time_delta)
        from_str = from_datetime.strftime('%Y-%m-%d')
        to_str = to_datetime.strftime('%Y-%m-%d')

        if isinstance(symbol, list):
            step = 5
            dfs = []
            for i in range(0, len(symbol), step):
                if i > 0:
                    time.sleep(5)
                dfs.append(  yf.download(symbol[i:i+step], start=from_str, end=to_str)  )
            dfs = pd.concat(dfs, axis=1)
        else:
            dfs = yf.download(symbol, start=from_str, end=to_str)

        write_parquet(dfs, file_path)
        time.sleep(1)
    if is_drop_hl:
        dfs = dfs.drop(columns=['High', 'Low'])
    if is_adj_close:
        return dfs.drop(columns=['Close']).rename(columns={'Adj Close': 'Close'})
    else:
        return dfs.drop(columns=['Adj Close'])

if __name__ == '__main__':
    
    pass