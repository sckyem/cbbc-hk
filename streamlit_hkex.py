from my_yfinance import get_ohlcv
from my_io import read_csv, read_parquet
from my_mongodb import Mongodb
import streamlit as st
from default_modules import *
import itertools

root = 'hkex'

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

def load_from(source):
    match source:
        case "Parquet":
            df = read_parquet(root, 'cbbc', 'cbbc')
        case "CSV":
            df = read_csv(root, 'cbbc', 'cbbc')
        case "MongoDB":
            #query = st.text_input("query", {})
            #projection = st.text_input("projection", {})
            document = Mongodb('test', 'cbbc')
            df = document.read(query={}, projection={}, is_dataframe=True)
    return df

def show_line_charts(dataframe, lines_per_tab=0, lines_per_chart=0, chart_height=0, is_show_close=False):
    if not lines_per_chart:
        lines_per_chart = st.sidebar.select_slider("No of Lines per Chart", list(range(1, 11)), 1)

    if not lines_per_tab:
        if not lines_per_chart:         
            lines_per_chart = st.sidebar.select_slider("No of Lines per Chart", list(range(1, 11)), 1)
        charts_per_tab = st.sidebar.select_slider("No of Charts per Tab", list(range(10, 101)), 10)
        lines_per_tab = lines_per_chart * charts_per_tab

    if not chart_height:
        chart_height = st.sidebar.select_slider("chart_height", list(range(200, 1001, 50)), 500)
    
    df = dataframe.copy()
    prev_symbol = ''
    tab_names = [  str(i) for i in range(0, math.ceil(len(df.columns) / lines_per_tab))]
    for i, tab in enumerate(st.tabs(tab_names)):
        with tab:
            tab_df = df[  df.columns[i*lines_per_tab:i*lines_per_tab+lines_per_tab]  ]
            for j in range(  0, lines_per_tab, lines_per_chart  ):
                chart_df = tab_df[  tab_df.columns[j:j+lines_per_chart]  ]

                if is_show_close:
                    if not chart_df.empty:
                        symbol = str(chart_df.columns[0]).split(',')[0]
                        symbol = yfinance_symbol(symbol)
                        if symbol != prev_symbol:
                            ohlcv = get_ohlcv(symbol, chart_df.index[0], chart_df.index[-1], is_download=False)
                        close = get_log_returns(ohlcv[ohlcv.columns[3]]).cumsum()
                        close_scaled = get_scaled_df(close, chart_df.min().min(), chart_df.max().max())
                        tab.line_chart(  pd.concat([chart_df, close_scaled], axis=1), height=chart_height  )
                else:
                    tab.line_chart(  chart_df, height=chart_height  )

def app():

    st.set_page_config(layout='wide')
    content = st.empty()

    with content.container():
        df = load_from("Parquet")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = columns_to_strings(df.columns)

        element_names = ['Underlyings', 'Data name', 'Market', 'MCE', 'Statistic']

        elements = [  list(set(t)) for t in zip(*[  str(i).split(',') for i in df.columns  ])  ]
        elements = [  sorted(e) for e in elements  ]
        elements_selected = [  st.sidebar.multiselect(f"{element_names[i]}", ['All'] + e, default='HSI' if i == 0 else 'All') for i, e in enumerate(elements) ]
        elements_selected = [  elements[i] if 'All' in e else e for i, e in enumerate(elements_selected)  ]
        elements_selected = [  ','.join(list(i)) for i in itertools.product(*elements_selected)  ]

        is_show_close = st.sidebar.selectbox("is_show_close", [True, False], 1)

        lines_per_chart = st.sidebar.select_slider("No of Lines per Chart", list(range(1, 11)), 1)
        charts_per_tab = st.sidebar.select_slider("No of Charts per Tab", list(range(10, 101)), 10)
        chart_height = st.sidebar.select_slider("chart_height", list(range(200, 1001, 50)), 500)
        lines_per_tab = lines_per_chart * charts_per_tab

        if elements_selected:
            df = df.loc[:, df.columns.isin(elements_selected)]

        if df is not None and not df.empty:
            prev_symbol = ''
            tab_names = [  str(i) for i in range(0, math.ceil(len(df.columns) / lines_per_tab))]
            for i, tab in enumerate(st.tabs(tab_names)):
                with tab:
                    tab_df = df[df.columns[  i*lines_per_tab:i*lines_per_tab+lines_per_tab  ]]
                    for j in range(  0, lines_per_tab, lines_per_chart  ):
                        chart_df = tab_df[tab_df.columns[j:j+lines_per_chart]]

                        if is_show_close:
                            if not chart_df.empty:
                                symbol = str(chart_df.columns[0]).split(',')[0]
                                symbol = yfinance_symbol(symbol)
                                if symbol != prev_symbol:
                                    ohlcv = get_ohlcv(symbol, chart_df.index[0], chart_df.index[-1], is_download=False)
                                    close = get_log_returns(ohlcv[ohlcv.columns[3]]).cumsum()
                                    close_scaled = get_scaled_df(close, chart_df.min().min(), chart_df.max().max())
                                    tab.line_chart(  pd.concat([chart_df, close_scaled], axis=1).dropna(), height=chart_height  )
                        else:
                            tab.line_chart(  chart_df, height=chart_height  )
        
if __name__ == '__main__':

    app()