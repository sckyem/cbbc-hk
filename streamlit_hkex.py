from my_backtests import sma, pnls, classify
from my_yfinance import get_ohlcv
from my_io import download_zip, write_csv, read_csv, path, read_parquet, write_parquet
import streamlit as st
from shutil import move
from default_modules import *
import itertools

root = 'hkex'
download_folder = path(root, 'download')
unzip_folder = path(root, 'unzip')
unzip_files = path(root, 'unzip', is_file_paths=True)
data_folder = path(root, 'data')
pnl_file = path(root, 'pnl', 'pnl')
data_file = path(data_folder, 'data')
pnl_file = 'pnl'

def yfinance_symbol(symbols):
    if not isinstance(symbols, list):
        symbols = [symbols]
    y = []
    for symbol in symbols:
        match symbol:
            case 'HSTEC':
                y.append('HSTECH.HK') 
            case 'HSCEI':
                y.append('^HSCE')  
        if str(symbol).isalpha():
            y.append(f'^{symbol}')   
        elif str(symbol).isnumeric() and len(str(symbol)) > 4:
            y.append(f'{symbol[-4:]}.HK')    
        else:
            y.append(symbol)
    return y

def download():
    def func(year, month):
        month = str(month).zfill(2)
        file_name = f"CBBC{month}"
        url = f"https://www.hkex.com.hk/eng/cbbc/download/{file_name}.zip"
        result = download_zip(url, download_folder, unzip_folder)
        if result:
            year = str(year % 100)
            old = path(unzip_folder, f"{file_name}.csv")
            new = path(unzip_folder, f"CBBC{year}{month}.csv")
            move(old, new)
            print(f"Renamed {file_name}.csv to CBBC{year}{month}.csv")
        return result
    today = datetime.datetime.now()
    result = func(today.year, today.month)

    if not result:
        time.sleep(1)
        prev_month = today.replace(day=1) - datetime.timedelta(days=1)
        result = func(prev_month.year, prev_month.month)
    return result

def data(files=[]):
    dfs = []
    for u in files if files else unzip_files:
        try: 
            dfs.append(  pd.read_csv(u, encoding='utf-16le', engine='python', skipfooter=3, delimiter="\t")  )
        except: pass
    df = pd.concat(dfs).reset_index(drop=True)
    df.columns = [  str(i).replace(' *', '').replace('**', '').replace('^', '').replace('.', '').replace('%', 'Pct').replace(' ', '_').replace('/', '_') for i in df.columns  ]
    
    df['Bull_Bear'] = df['Bull_Bear'].str.rstrip()
    df['CBBC_Type'] = df['CBBC_Type'].str.rstrip()
    df['Trade_Date'] = pd.to_datetime(df['Trade_Date'])
    df = df.sort_values(by=['Trade_Date'])
    df[['No_of_CBBC_still_out_in_market', 'No_of_CBBC_Bought', 'Average_Price_per_CBBC_Bought', 'No_of_CBBC_Sold', 'Average_Price_per_CBBC_Sold', 'Volume', 'Turnover']] = df[['No_of_CBBC_still_out_in_market', 'No_of_CBBC_Bought', 'Average_Price_per_CBBC_Bought', 'No_of_CBBC_Sold', 'Average_Price_per_CBBC_Sold', 'Volume', 'Turnover']].replace('-', 0).astype(float)
    df = df.rename(columns={'No_of_CBBC_still_out_in_market':'CBBC'})
    return df.reset_index(drop=True)

def symbols():
    df = data(files=unzip_files[-1:])
    return df.groupby('Underlying')['Turnover'].sum().sort_values(ascending=False).index.to_list()

def cbbc():
    df = data()
    df['Bought_Amount'] = df['No_of_CBBC_Bought'] * df['Average_Price_per_CBBC_Bought']
    df['Sold_Amount'] = df['No_of_CBBC_Sold'] * df['Average_Price_per_CBBC_Sold']
    df = df[['Trade_Date', 'Underlying', 'Bull_Bear', 'Bought_Amount', 'Sold_Amount', 'Volume', 'CBBC', 'Turnover']]
    df = df.groupby(  ['Trade_Date', 'Underlying', 'Bull_Bear']  ).sum().unstack(level=[1, 2], fill_value=0)
    df = df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    return df

def read_cbbc(is_read_csv=False, is_read_parquet=False, is_read_mongodb=False, is_write_mongodb=False, is_write_csv=False, is_write_parquet=False, query={}, projection={}):
    if is_read_mongodb or is_write_mongodb:
        document = Mongodb('test', 'cbbc')
    file = path(root, 'cbbc', 'cbbc')    
    if is_read_mongodb:
        df = document.read(query, projection, is_dataframe=True)
    elif is_read_csv:
        df = read_csv(file)
    elif is_read_parquet:
        df = read_parquet(file)
    else:
        df = cbbc()
    if df is not None:
        if is_write_mongodb:
            document.update(df)
        if is_write_csv:
            write_csv(df, file)
        if is_write_parquet:
            write_parquet(df, file)
    return df

def load_from(source):
    match source:
        case "HKEX":
            download()
            df = cbbc()
        case "Parquet":
            df = read_parquet(root, 'cbbc', 'cbbc')
        case "CSV":
            df = read_csv(root, 'cbbc', 'cbbc')
        case "MongoDB":
            #query = st.text_input("query", {})
            #projection = st.text_input("projection", {})
    return df

def save_to(data, save_format):
    match save_format:
        case "Parquet":
            result = write_parquet(data, root, 'cbbc', 'cbbc')
        case "CSV":
            result = write_csv(data, root, 'cbbc', 'cbbc')
    return result

def show_line_charts(dataframe, lines_per_tab, lines_per_chart, is_show_close, is_download=False):
    df = dataframe.copy()
    prev_symbol = ''
    tab_names = [  str(i) for i in range(0, math.ceil(len(df.columns) / lines_per_tab))]
    for i, tab in enumerate(st.tabs(tab_names)):
        with tab:
            tab_df = df[  df.columns[i*lines_per_tab:i*lines_per_tab+lines_per_tab]  ]
            for j in range(  0, lines_per_tab, lines_per_chart  ):
                chart_df = tab_df[  tab_df.columns[j:j+lines_per_chart]  ]

                tab.write(  ' '.join(chart_df)  )

                if is_show_close:
                    if not chart_df.empty:
                        symbol = str(chart_df.columns[0]).split(',')[0]
                        symbol = yfinance_symbol(symbol)
                        if symbol != prev_symbol:
                            ohlcv = get_ohlcv(symbol, chart_df.index[0], chart_df.index[-1], is_download=is_download)
                            if is_download:
                                time.sleep(2)
                        close = get_log_returns(ohlcv[ohlcv.columns[3]]).cumsum()
                        close_scaled = get_scaled_df(close, chart_df.min().min(), chart_df.max().max())
                        tmp = pd.concat([chart_df, close_scaled], axis=1)
                        tab.line_chart(  tmp  )
                else:
                    tab.line_chart(  chart_df  )

def app():

    st.set_page_config(layout="wide")

    if "page" not in st.session_state:
        st.session_state.page = 0

    def next(): 
        st.session_state.page += 1
    def back(): 
        st.session_state.page -= 1
    def restart(): 
        st.session_state.page = 0

    content = st.empty()

    # Load Data
    if st.session_state.page == 0:
        
        with content.container():
            # is_load_pnls = st.sidebar.selectbox('is_load_pnls', [True, False, ], index=1)
            # if is_load_pnls:
            #     pass
            # else:            
                source = st.sidebar.selectbox('Load data from', [False, "HKEX", "Parquet", "CSV"])
                if source:
                    cbbc = load_from(source)
                    st.session_state.cbbc = cbbc

                    if cbbc is not None:
                        st.dataframe(cbbc)
                        save_format = st.sidebar.selectbox('save_format', [False, "Parquet", "CSV", "MongoDB"])

                        if save_format:
                            result = save_to(cbbc, save_format)
                            st.write(result)
                    
                    else:
                        st.write(f"No data from {source}")

    # show_line_chart
    if st.session_state.page == 1:
        
        with content.container():
            df = st.session_state.cbbc

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = columns_to_strings(df.columns)
            elements = [  list(set(t)) for t in zip(*[  str(i).split(',') for i in df.columns  ])  ]
            elements_selected = [  st.sidebar.multiselect(f"level{i} elements", ['All'] + e, default='HSI' if i == 0 else 'All') for i, e in enumerate(elements) ]
            elements_selected = [  elements[i] if 'All' in e else e for i, e in enumerate(elements_selected)  ]
            elements_selected = [  ','.join(list(i)) for i in itertools.product(*elements_selected)  ]

            is_update_yfinance = st.sidebar.selectbox("is_update_yfinance", [True, False], 1)
            if is_update_yfinance:
                for symbol in elements[0]:
                    get_ohlcv(yfinance_symbol(symbol), df.index[0], datetime.datetime.now(), 5, 5, is_download=is_update_yfinance)
                    time.sleep(2)

            is_show_close = st.sidebar.selectbox("is_show_close", [True, False], 1)
            drop_nlargest_rows = st.sidebar.number_input("drop_nlargest_rows", step=1)
            drop_nsmallest_rows = st.sidebar.number_input("drop_nsmallest_rows", step=1)
            save_format = st.sidebar.selectbox('save_format', [False, "Parquet", "CSV", "MongoDB"])

            lines_per_chart = st.sidebar.select_slider("No of Lines per Chart", list(range(1, 11)), 1)
            charts_per_tab = st.sidebar.select_slider("No of Charts per Tab", list(range(10, 101)), 10)
            lines_per_tab = lines_per_chart * charts_per_tab

            if elements_selected:
                df = df.loc[:, df.columns.isin(elements_selected)]
            if df is not None and not df.empty:

                if drop_nlargest_rows:
                    df = df.drop(df.nlargest(drop_nlargest_rows, columns=df.columns).index.tolist())
                if drop_nsmallest_rows:
                    df = df.drop(df.nsmallest(drop_nsmallest_rows, columns=df.columns).index.tolist())
                if save_format:
                    result = save_to(df, save_format)
                    st.write(result)

                prev_symbol = ''
                tab_names = [  str(i) for i in range(0, math.ceil(len(df.columns) / lines_per_tab))]
                for i, tab in enumerate(st.tabs(tab_names)):
                    with tab:
                        tab_df = df[df.columns[i*lines_per_tab:i*lines_per_tab+lines_per_tab]]
                        for j in range(  0, lines_per_tab, lines_per_chart  ):
                            chart_df = tab_df[tab_df.columns[j:j+lines_per_chart]]

                            tab.write(  ' '.join(chart_df)  )

                            if is_show_close:
                                if not chart_df.empty:
                                    symbol = str(chart_df.columns[0]).split(',')[0]
                                    symbol = yfinance_symbol(symbol)
                                    if symbol != prev_symbol:
                                        ohlcv = get_ohlcv(symbol, chart_df.index[0], chart_df.index[-1], is_download=False)
                                        close = get_log_returns(ohlcv[ohlcv.columns[3]]).cumsum()
                                        close_scaled = get_scaled_df(close, chart_df.min().min(), chart_df.max().max())
                                    if is_show_close:
                                        tmp = pd.concat([chart_df, close_scaled], axis=1).dropna()
                                    tab.line_chart(  tmp  )
                            else:
                                tab.line_chart(  chart_df  )
            st.session_state.selected_cbbc = df
            

    # TA
    if st.session_state.page == 2:

        with content.container():
            df = st.session_state.selected_cbbc
            ta_name = st.sidebar.selectbox("TA", ["No", "SMA"])
                
            if ta_name == "SMA":
                windows = st.sidebar.multiselect("Windows", [5, 10, 20])
                if windows:                    
                    ta_method = st.sidebar.selectbox("SMA_ta_method", ["", "is_above_sma", "is_uptrend", "is_accelerate"])
                    if ta_method:
                        df = sma(df, windows, **{ta_method:True}, is_classify=False)


            is_show_close = st.sidebar.selectbox("is_show_close", [True, False], 1)
            lines_per_chart = st.sidebar.select_slider("No of Lines per Chart", list(range(1, 11)), 1)
            charts_per_tab = st.sidebar.select_slider("No of Charts per Tab", list(range(10, 101)), 10)
            lines_per_tab = lines_per_chart * charts_per_tab
            df.columns = columns_to_strings(df.columns)
            show_line_charts(df, lines_per_tab, lines_per_chart, is_show_close)
            st.session_state.ta = df

    # Combination
    if st.session_state.page == 3:
        
        with content.container():
            df = st.session_state.ta
            df.columns = [','.join(str(v) for v in c) if isinstance(c, tuple) else str(c) for c in df.columns]
            elements = [set(t) for t in zip(*[  str(i).split(',') for i in df.columns  ])]

            elements_limit = [  st.sidebar.select_slider(f"level{i} elements_limit", list(range(1, len(set) +1)), 1) if len(set) > 1 else 1 for i, set in enumerate(elements) ]
            combination_limit = max(elements_limit)
            combinations = combination(df.columns, combination_limit)
            combinations = [c for c in combinations if all(a <= b for a, b in zip([len(set(t)) for t in zip(*[  str(i).split(',') for i in c])], elements_limit))]

            start_time_delta = st.sidebar.select_slider("start_time_delta", list(range(1, 6)), 1)
            duration = st.sidebar.select_slider("duration", list(range(1, 3)), 1)
            is_show_last = st.sidebar.selectbox("is_show_last", [True, False], 1)
            is_show_pnl = st.sidebar.selectbox("is_show_pnl", [True, False], 1)
            is_compare_benchmark = st.sidebar.selectbox("is_compare_benchmark", [True, False], 1)
            if is_show_pnl or is_compare_benchmark:
                chart_height = st.sidebar.select_slider("chart_height", list(range(500, 1001, 50)), 500)

            prev_symbol = ''
            tasks = []
            for c in combinations:
                symbol = c[0][0]
                if symbol != prev_symbol:                    
                    ohlcv = get_ohlcv(yfinance_symbol(symbol), df.index[0], df.index[-1], 1, 1, is_drop_hl=True, is_download=False)

                        # windows = st.sidebar.multiselect("Windows", [5, 10, 20])
                        
                        # ta_method = st.sidebar.selectbox("SMA_ta_method", ["", "is_above_sma", "is_uptrend", "is_accelerate"])
                        # if ta_method:
                        #     ohlcv = sma(ohlcv, windows, **{ta_method:True}, is_classify=False)

                    returns = get_log_returns(ohlcv, start_time_delta, duration)
                tasks.append([returns, df[c]])
                prev_symbol = symbol

            if tasks:
                tab_names = [  str(i) for i in range(0, len(tasks))  ]
                for i, tab in enumerate(st.tabs(tab_names)):
                    with tab:
                        task = tasks[i]
                        tab.dataframe(task[1].columns)
                        task[1] = classify(task[1])
                        last = task[1].iloc[-1].tolist()
                        for r in task[0]:
                            pnl = pnls(  task[0][r], task[1]  )
                            if is_show_last:
                                pnl = pnl[[  (*list(c)[:1], *list(c)[1:]) if isinstance(c, tuple) else c for c in pnl.columns if last == list(c)[1:]  ]]
                            pnl.columns = columns_to_strings(pnl.columns)



                            if is_show_pnl:
                                cpnl = pd.concat([task[0][r], pnl], axis=1).dropna().cumsum()
                                cpnl.iloc[0] = 0
                                tab.line_chart(cpnl, height=chart_height)
                            elif is_compare_benchmark:
                                compare = cpnl.apply(lambda row: row[1:].subtract(row[0]), axis=1)
                                tab.line_chart(compare, height=chart_height)
                            else:
                                tab.dataframe(pd.concat(task, axis=1).dropna())
                            
            else:
                st.write("No task")

            #         if False:
            #             d['ohlcv_sma'] = sma(d['ohlcv'], sma_windows)
            #             d['ohlcv_sma'].columns = columns_to_strings(d['ohlcv_sma'].columns)
            #             d['ohlcv_combination'] = combination(d['ohlcv_sma'].columns.to_list(), ohlcv_combination_length)
            #             for i in d['ohlcv_combination']:              
            #                 tasks.append(  [d['returns'], pd.concat([d['ohlcv_sma'][i], indicator], axis=1)   ]  )
            #         else:
            #             tasks.append(  [returns, indicator]  )
                
    #                             d['task_length'] = len(tasks)  
    #                             d['task0'] = tasks[0]
    #                             d['is_process_pnls'] = is_process_pnls
    #                             if is_process_pnls:
    #                                 result = multi_process(pnls, tasks)
    #                                 d['label'] = {  i:list(r.columns.names) for i, r in enumerate(result)  }
    #                                 d['pnls'] = pd.concat([  r.assign(label_no=i).set_index('label_no', append=True).unstack(1, fill_value=0) for i, r in enumerate(result)  ], axis=1)

    #                                 if is_process_cumsum:            
    #                                     d['cumsum'] = cumsums(d['pnls'], lowest_pnl)                            
    #                             write_parquet(d, pnl_file)
            

    st.sidebar.button("Back", on_click=back, disabled=(st.session_state.page > 3))
    st.sidebar.button("Next", on_click=next, disabled=(st.session_state.page > 3))
    st.sidebar.button("Restart", on_click=restart)

if __name__ == '__main__':

    app()
