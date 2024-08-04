from my_mongodb import Mongodb, list_collection_names
import streamlit as st
from default_modules import *
import itertools

root = 'hkex'
element_names = [  'Symbol', 'Data Name', 'Market', 'MCE', 'Aggregate'  ]
symbols = ['HSI']

@st.cache_data(ttl=28800)
def symbol_close(start, end):
    result = {}    
    for symbol in symbols:
        document = Mongodb('yfinance', symbol)
        result[symbol] = document.read({'_id': {'$gte': start, '$lte': end}}, {'_id':1, 'Close':1}, is_dataframe=True)
    return result

@st.cache_data(ttl=28800)
def cbbc():
    document = Mongodb('cbbc', 'cbbc')
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
        
        elements = [  sorted(list(set(t))) for t in zip(*[  str(i).split(',') for i in df.columns  ])  ]
        elements[0] = symbols

        elements_selected = [  st.sidebar.multiselect(f"{element_names[i]}", e) for i, e in enumerate(elements) ]    
        elements_selected = [  e if e else elements[i] for i, e in enumerate(elements_selected)  ]

        columns_filtered = [  ','.join(list(i)) for i in itertools.product(*elements_selected)  ]
        if columns_filtered:
            df = df.loc[:, df.columns.isin(columns_filtered)]
        df = df[sorted(df.columns)]

        symbols_closes = symbol_close(start, end)
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
