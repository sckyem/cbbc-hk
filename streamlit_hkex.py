from my_mongodb import Mongodb
import streamlit as st
from default_modules import *
import itertools
from my_hkex import yfinance_symbol, symbol

ELEMENT_NAMES = [  'Symbol', 'Data Name', 'Market', 'MCE'  ]
SYMBOLS = symbol()

@st.cache_data(ttl=28800)
def symbol_close(symbols, start, end):
    result = {}
    for i in symbols:
        symbol = yfinance_symbol(i)
        document = Mongodb('yfinance', symbol)
        close = document.read({'_id': {'$gte': start, '$lte': end}}, {'_id': 1, 'Close': 1}, is_dataframe=True)        
        if close is not None:
            result[i] = close
    return result

@st.cache_data(ttl=28800)
def cbbc():
    document = Mongodb('cbbc', 'sum')
    return document.read(is_dataframe=True)   

def app():
    st.set_page_config(
        layout='wide',
        page_title="Historical Data of CBBC", 
        page_icon="📈",
        menu_items={            
            'About': "Created by CKY"
            }
        )

    df = cbbc()
    
    if df is not None and not df.empty:
        
        from_time = st.sidebar.radio(  "Date Range", ["3M", "1Y", "All"], 1, horizontal=True  )
        if from_time != 'All':
            start = df.index[-1] - interval_to_timedelta(from_time)
        else:
            start = df.index[0]
        end = df.index[-1]
        df = df.loc[df.index[df.index.isin(pd.date_range(start, end, freq='D'))]]            
        
        symbol_selected = st.sidebar.selectbox("Symbol", SYMBOLS, 0)
        elements = [  list(set(t)) for t in zip(*[  str(i).split(',') for i in df.columns  ])  ][1:]
        elements_selected = [  st.sidebar.multiselect(f"{ELEMENT_NAMES[i]}", e) for i, e in enumerate(elements) ]    
        elements_selected = [[symbol_selected]] + [  e if e else elements[i] for i, e in enumerate(elements_selected)  ]
        
        columns_filtered = [  ','.join(list(i)) for i in itertools.product(*elements_selected)  ]
        if columns_filtered:
            df = df[columns_filtered]   
        
        symbols_closes = symbol_close(elements_selected[0], start, end)
        
        charts_per_tab = st.sidebar.select_slider("No of Charts per Tab", list(range(10, 101)), 10)
        height = st.sidebar.select_slider("chart_height", list(range(200, 1001, 50)), 300)

        st.title("Historical Data of CBBC", anchor=False)
        st.write(f"Last update: {end.strftime('%Y-%m-%d (%a)')}")

        tab_names = [  str(i+1) for i in range(0, len(df.columns) // charts_per_tab + 1)  ]
        for i, tab in enumerate(st.tabs(tab_names)):
        
            with tab:
                cols = df.columns[  i*charts_per_tab:(i+1)*charts_per_tab  ].to_list()
                if cols:
                    for j in cols:
                        st.write(j)
                        chart = df[j].dropna()
                        symbol = str(j).split(',')[0]

                        close = symbols_closes.get(symbol)
                        if close is not None:
                            scaled = get_scaled_df(close, chart.min(), chart.max())

                            if scaled is not None:
                                scaled = scaled.loc[scaled.index.isin(chart.index)]
                                st.line_chart(  pd.concat([chart, scaled], axis=1), height=height  )     
                            else:
                                st.line_chart(  chart, height=height  )
                                
if __name__ == '__main__':

    app()