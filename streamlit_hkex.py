from my_io import read_csv, read_parquet
from my_mongodb import Mongodb
import streamlit as st
from default_modules import *
import itertools

root = 'hkex'
coffee = "https://buymeacoffee.com/sckyem"

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
    
@st.cache_data(ttl=28800)
def load_from(source, collection='test', document='test', query={}, projection={}):
    match source:
        case "Parquet":
            df = read_parquet(root, 'cbbc', 'cbbc')
        case "CSV":
            df = read_csv(root, 'cbbc', 'cbbc')
        case "MongoDB":
            document = Mongodb(collection, document)
            df = document.read(query, projection, is_dataframe=True)            
    return df

def app():

    st.set_page_config(
        layout='wide',
        page_title="Historical Data of CBBC", 
        page_icon="📈",
        menu_items={            
            'About': "Created by CKY"
            }
        )
    content = st.empty()

    with content.container():
        df = load_from("MongoDB", 'cbbc', 'cbbc')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = columns_to_strings(df.columns)

        from_time = st.sidebar.radio(  "Date Range", ["3M", "1Y", "All"], 1, horizontal=True  )
        
        if from_time != 'All':
            df = df.loc[df.index[-1] - interval_to_timedelta(from_time):]

        element_names = [  'Underlyings', 'Data name', 'Market', 'MCE', 'Statistic'  ]

        elements = [  sorted(list(set(t))) for t in zip(*[  str(i).split(',') for i in df.columns  ])  ]
        elements_selected = [  st.sidebar.multiselect(f"{element_names[i]}", ['All'] + e, default='HSI' if i == 0 else 'All') for i, e in enumerate(elements) ]
        elements_selected = [  elements[i] if 'All' in e else e for i, e in enumerate(elements_selected)  ]
        elements_selected = [  ','.join(list(i)) for i in itertools.product(*elements_selected)  ]

        if elements_selected:
            df = df.loc[:, df.columns.isin(elements_selected)]

        is_show_charts = st.sidebar.toggle("Show Charts", True)
        is_show_close = st.sidebar.toggle("Show Close", True)

        lines_per_chart = st.sidebar.select_slider("No of Lines per Chart", list(range(1, 11)), 1)
        charts_per_tab = st.sidebar.select_slider("No of Charts per Tab", list(range(10, 101)), 10)
        lines_per_tab = lines_per_chart * charts_per_tab

        chart_height = st.sidebar.select_slider("chart_height", list(range(200, 1001, 50)), 300)
        st.sidebar.link_button("☕ Buy Me a Coffee", coffee)

        if df is not None and not df.empty:
            st.title("Historical Data of CBBC", anchor=False)
            st.write(f"Last update: {df.index[-1].strftime('%Y-%m-%d (%a)')}")

            tab_names = [  str(i+1) for i in range(0, math.ceil(len(df.columns) / lines_per_tab))]
            for i, tab in enumerate(st.tabs(tab_names)):
                
                symbols = list(set(  [str(i).split(',')[0] for i in df.columns]  ))
                symbols_closes = {  i:load_from("MongoDB", 'yfinance', yfinance_symbol(i), {'_id': {'$gte': df.index[0], '$lte': df.index[-1]}}, {'_id':1, 'Close':1}) for i in symbols  }

                with tab:
                    tab_cols = df.columns[  i*lines_per_tab:i*lines_per_tab+lines_per_tab  ].to_list()
                    for j in range(  0, lines_per_tab, lines_per_chart  ):
                        chart_cols = tab_cols[  j:j+lines_per_chart  ]

                        if chart_cols:      
                            symbol = str(chart_cols[0]).split(',')[0]
                            close = symbols_closes[symbol]
                            chart_df = df[chart_cols]

                            if is_show_charts:
                                if is_show_close:
                                    if close is not None:
                                        close = get_scaled_df(close, chart_df.min().min(), chart_df.max().max())
                                        chart_df = pd.concat([close, chart_df], axis=1)                      
                                tab.line_chart(  chart_df, height=chart_height  )
                            else:
                                if is_show_close:
                                    chart_df = pd.concat([close, chart_df], axis=1)
                                tab.dataframe(  chart_df  )

if __name__ == '__main__':

    app()