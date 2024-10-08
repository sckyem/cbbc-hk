import pandas as pd
from default_modules import *
from my_io import download_zip, write_csv, path, write_parquet, remove
from shutil import move
from my_mongodb import Mongodb
from my_yfinance import get_ohlcv

root = 'hkex'
save_file = path(root, 'cbbc', 'cbbc')
download_folder = path(root, 'download')
unzip_folder = path(root, 'unzip')
unzip_files = path(root, 'unzip', is_file_paths=True)
last_files = unzip_files[-2:]
dates_filter_out = ['2023-07-17']
neglected_symbols = [  'HSTEC'  ]

def write_symbol():
    df = data(last_files)
    symbols = df.groupby('Underlying')['Turnover'].sum().sort_values(ascending=False).index.to_list()
    symbols = [  i for i in symbols if i not in neglected_symbols ]
    symbols = pd.Series(symbols, name='symbol')
    doc = Mongodb('cbbc', 'symbol')
    doc.delete()
    doc.write(symbols)

def symbol():
    symbols = Mongodb('cbbc', 'symbol').read()
    return [  i.get('symbol') for i in symbols  ]

def yfinance_symbol(symbols):
    def func(symbol):
        match symbol:
            case 'HSTEC':
                return 'HSTECH.HK'
            case 'HSCEI':
                return '^HSCE'
        if str(symbol).isalpha():
            return f'^{symbol}'
        elif str(symbol).isnumeric() and len(str(symbol)) > 4:
            return f'{symbol[-4:]}.HK'
        else:
            return symbol

    if isinstance(symbols, list):
        return [  func(i) for i in symbols  ]
    else:
        return func(symbols)

def download():
    def func(year, month):
        month = str(month).zfill(2)
        file_name = f"CBBC{month}"
        url = f"https://www.hkex.com.hk/eng/cbbc/download/{file_name}.zip"
        result = download_zip(url, download_folder, unzip_folder)
        if result:
            old = path(unzip_folder, f"{file_name}.csv")
            check = data(old)
            if year == check['Trade_Date'].iloc[-1].year:
                year = str(year % 100)
                new = path(unzip_folder, f"CBBC{year}{month}.csv")
                move(old, new)
                print(f"Renamed {file_name}.csv to CBBC{year}{month}.csv")
            else:
                remove(old)
                return False
        return result
    today = datetime.datetime.now()
    result = func(today.year, today.month)

    if not result:
        time.sleep(1)
        prev_month = today.replace(day=1) - datetime.timedelta(days=1)
        result = func(prev_month.year, prev_month.month)
    return result

def data(files=[]):
    if isinstance(files, str):
        files = [files]
    if not files:
        files = unzip_files
    if files:
        dfs = []
        for f in files:
            try: 
                dfs.append(  pd.read_csv(f, encoding='utf-16le', engine='python', skipfooter=3, delimiter="\t")  )
            except: pass
        if dfs:
            df = pd.concat(dfs).reset_index(drop=True)
            df.columns = [  str(i).replace(' *', '').replace('**', '').replace('^', '').replace('.', '').replace('%', 'Pct').replace(' ', '_').replace('/', '_') for i in df.columns  ]
            
            df['Bull_Bear'] = df['Bull_Bear'].str.rstrip()
            df['CBBC_Type'] = df['CBBC_Type'].str.rstrip()
            df = df[~df['Trade_Date'].isin(dates_filter_out)]
            df['Trade_Date'] = pd.to_datetime(df['Trade_Date'])
            df = df.sort_values(by=['Trade_Date'])
            df[['No_of_CBBC_still_out_in_market', 'No_of_CBBC_Bought', 'Average_Price_per_CBBC_Bought', 'No_of_CBBC_Sold', 'Average_Price_per_CBBC_Sold', 'Volume', 'Turnover']] = df[['No_of_CBBC_still_out_in_market', 'No_of_CBBC_Bought', 'Average_Price_per_CBBC_Bought', 'No_of_CBBC_Sold', 'Average_Price_per_CBBC_Sold', 'Volume', 'Turnover']].replace('-', 0).astype(float)
            df = df.rename(columns={'No_of_CBBC_still_out_in_market':'CBBC'})
            return df.reset_index(drop=True)

def ungrouped_cbbc(files=[]):
    df = data(files)
    df['Bought'] = df['No_of_CBBC_Bought'] * df['Average_Price_per_CBBC_Bought']
    df['Sold'] = df['No_of_CBBC_Sold'] * df['Average_Price_per_CBBC_Sold']
    df['Share'] = df['CBBC'] / df['Ent_Ratio']
    df = df.set_index(['Trade_Date', 'Underlying', 'Bull_Bear', 'MCE', 'Call_Level'])
    df = df[['Bought', 'Sold', 'Volume', 'CBBC', 'Turnover', 'Share']]
    df = df[~(df == 0).all(axis=1)]
    return df.sort_index()

def cbbc(files=[]):
    df = ungrouped_cbbc(files)
    skew = df.loc[  df.index.get_level_values('MCE') == 'N'  ].groupby(level=[0, 1, 2, 3]).agg('skew').assign(Skew='Skew').set_index('Skew', append=True).unstack(level=[1, 2, 3, 4]).dropna(axis=1, how='all').fillna(0)
    sum = df.groupby(level=[0, 1, 2, 3]).agg('sum').assign(Sum='Sum').set_index('Sum', append=True).unstack(level=[1, 2, 3, 4]).dropna(axis=1, how='all').fillna(0)
    df = pd.concat([sum, skew], axis=1)
    df = df.swaplevel(0, 1, axis=1)
    return df

# def sum_cbbc(files=[]):
#     df = ungrouped_cbbc(files)
#     sum = df.groupby(level=[0, 1, 2, 3]).agg('sum').unstack(level=[1, 2, 3]).dropna(axis=1, how='all').fillna(0)
#     return sum.swaplevel(0, 1, axis=1)

def sum_cbbc(files=[]):
    df = ungrouped_cbbc(files)
    sum = df.groupby(level=[0, 1, 2, 3]).agg('sum').unstack(level=[1, 2, 3]).dropna(axis=1, how='all').fillna(0)
    return sum.swaplevel(0, 1, axis=1)

def skew_cbbc(files=[]):
    df = ungrouped_cbbc(files)
    skew = df.loc[  df.index.get_level_values('MCE') == 'N'  ].groupby(level=[0, 1, 2, 3])#.agg('skew').unstack(level=[1, 2, 3]).dropna(axis=1, how='all').fillna(0)
    #return skew.swaplevel(0, 1, axis=1)
    for name, group in skew:
        print(name)
        print(group)
        print()

def is_update(time):
    if time is None:
        return True
    else:
        current = datetime.datetime.now()
        current_weekday = current.weekday()
        delta = current_weekday - time.weekday()
        
        if current.hour < 9:
            if current_weekday == 0:
                if delta > 3:
                    return True
            elif current_weekday != 6:
                if delta > 1:
                    return True
        elif current.hour > 20:
            if delta:
                return True
        return False

def update_cbbc_sum(  is_check=True, is_write=True, is_replace=False, files=unzip_files  ):
    doc = Mongodb('cbbc', 'sum')
    last = doc.last_id()
    if not is_check or is_update(last):
        downloaded = download()
        if downloaded:
            df = sum_cbbc(files)
            if df is not None:
                if is_write:
                    doc.update(df, is_replace)
                    print(df.tail())

def update_yf(  is_check=True, is_write=True  ):    
    count = 0
    for i in symbol():
        i = yfinance_symbol(i)
        doc = Mongodb('yfinance',)
        last = doc.last_id()
        if last is None:
            last = string_to_datetime('2014-01-01')
            
        if not is_check or is_update(last):
            if count > 0:
                time.sleep(5)
            count += 1
            if is_write:
                df = get_ohlcv(i, last, datetime.datetime.now(), 10)
                doc.update(df)
                print(df.tail())

def print_cbbc():
    doc = Mongodb('cbbc', 'sum')
    df = doc.read(is_dataframe=True)
    print(df)

if __name__ == '__main__':
    
    for f in unzip_files: update_cbbc_sum(False, True, True, files=f)
    update_yf(False)
    write_symbol()
    print_cbbc()
    pass