import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from my_module.my_script import *
from my_module.my_mongodb import MongodbReaders, MongodbReader
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class StCbbc:
    def __init__(self):
        file_path = Path('/home/sckyem/.streamlit/secrets.toml')
        if file_path.exists():
            self.ADDRESS = f'mongodb+srv://{st.secrets["user"]}:{st.secrets["pwd"]}@cluster0.xovoill.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
        else:
            self.ADDRESS = ''
        self.INDICATORS_NAMES = ['hkex_cbbc','hkex_cbbc_ratio']
        self.PAGE_TITLE = ''
        self.CONTENT_TITLE = 'Streamlit Cbbc'
        self.CHART_TYPES = ['Line Chart', 'Table']
        self.N_COMPONENTS = 1
    def select_item(self, option_name=str, options=list):
        if option_name in st.query_params:
            index = options.index(st.query_params[option_name])
        else:
            index = 0
        db_name = st.sidebar.selectbox(f'Choose {str(option_name).capitalize()}', options, index)
        st.query_params[option_name] = db_name
        return db_name
    def select_indicator_name(self):
        return self.select_item('indicator', self.INDICATORS_NAMES)
    def select_underlying(self, db_name):
        underlyings = MongodbReaders(self.ADDRESS, db_name).list_collection_names()
        return self.select_item('underlying', underlyings)
    def get_collection(self, db_name, collection_name):
        if db_name not in st.session_state:
            st.session_state[db_name] = {}
        if collection_name in st.session_state[db_name]:
            collection = st.session_state[db_name][collection_name]
        else:
            collection = MongodbReader(self.ADDRESS, db_name, collection_name).collection_to_dataframe()
            if collection is not None:
                collection = collection.fillna(0)
            st.session_state[db_name][collection_name] = collection
        return collection
    def select_ma(self, dataframe=pd.DataFrame or pd.Series):
        if 'ma' in st.query_params:
            value = int(st.query_params['ma'])
        else:
            value = 0
        ma = st.sidebar.number_input('Moving Average', min_value=0, value=value, step=5)
        st.query_params['ma'] = ma
        if ma:
            return dataframe.rolling(int(ma)).mean()
        else:
            return dataframe
    def get_pca(self, dataframe=pd.DataFrame):
        scaled_data = StandardScaler().fit_transform(dataframe)
        pca = PCA(n_components=self.N_COMPONENTS)
        pca_result = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=pca_result, columns=[f'PCA{i}' for i in list(range(1, self.N_COMPONENTS+1))], index=dataframe.index)
        return pca_df
    def select_chart_type(self):
        return self.select_item('chart_type', self.CHART_TYPES)
    def show_chart(self, chart_type, collection):
        match chart_type:
            case 'Line Chart':
                for i in collection:
                    st.write(i)
                    st.line_chart(collection[i])
            case 'Table':
                st.write(collection)
    def run(self):
        st.set_page_config(self.PAGE_TITLE, layout="wide")
        st.title(self.CONTENT_TITLE)
        indicator_name = self.select_indicator_name()
        underlying = self.select_underlying(indicator_name)
        indicator = self.get_collection(indicator_name, underlying)
        indicator_ma = self.select_ma(indicator)
        chart_type = self.select_chart_type()
        self.show_chart(chart_type, indicator_ma)

class StOhlcv(StCbbc):
    def __init__(self):
        super().__init__()
        self.PAGE_TITLE = ''
        self.CONTENT_TITLE = 'Streamlit Ohlcv'
        self.OHLCV_NAME = 'yfinance'
    def get_ohlcv(self, underlying):
        symbol = cbbc_underlying_to_yf_symbol(underlying)
        return self.get_collection(self.OHLCV_NAME, symbol)
    def show_candlestick(self, chart_type, ohlcv):
        if ohlcv is not None:
            match chart_type:
                case 'Line Chart':
                    fig = go.Figure(
                        go.Candlestick(
                            x=ohlcv.index,
                            open=ohlcv['Open'],
                            high=ohlcv['High'],
                            low=ohlcv['Low'],
                            close=ohlcv['Close'],
                            )
                        )
                    fig.update_layout(xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig)
                case 'Table':
                    st.dataframe(ohlcv, use_container_width=True)
    def run(self):
        st.set_page_config(self.PAGE_TITLE, layout="wide")
        st.title(self.CONTENT_TITLE)
        indicator_name = self.select_indicator_name()
        underlying = self.select_underlying(indicator_name)
        ohlcv = self.get_ohlcv(underlying)
        chart_type = self.select_chart_type()
        self.show_candlestick(chart_type, ohlcv)

class StReturns(StOhlcv):
    def __init__(self):
        super().__init__()
        self.PAGE_TITLE = ''
        self.CONTENT_TITLE = 'Streamlit Returns'
    def get_sign(self, indicator, ma=0):
        df = deepcopy(indicator)
        if ma > 1:
            df = indicator.rolling(ma).mean()
        df['ma'] = ma
        df = df.set_index('ma', append=True).unstack(level=-1)
        df.columns = columns_to_strings(df.columns)
        return np.sign(df)
    def get_signs(self, indicator):
        windows = list(range(1, 21)) #[1, 2, 3, 4, 5, 10, 15, 20]
        return pd.concat([self.get_sign(indicator, ma) for ma in windows], axis=1).iloc[max(windows):]
    def get_signs_of_last(self, indicator):
        signs = self.get_signs(indicator)
        return signs.apply(lambda x: np.where(x == x.iloc[-1], 1, 0))
    def get_benchmark(self, ohlcv):
        return np.log(ohlcv['Open']).diff()
    # def get_cat_pnl(self, benchmark, cat):
    #     dfs = []
    #     for i in cat:
    #         df = pd.concat([benchmark, cat[i].shift(2)], axis=1)
    #         df = df.set_index(i, append=True).unstack(level=-1)
    #         df.columns = columns_to_strings(df.columns)
    #         df.columns = [f'{i},{j}' for j in df.columns]
    #         dfs.append(df)
    #     return pd.concat(dfs, axis=1).fillna(0)
    def get_pnl(self, benchmark, signs):
        df = pd.concat([benchmark, signs.shift(2)], axis=1).dropna()
        r = pd.DataFrame()
        for i in signs:
            r[i] = df[i] * df[df.columns[0]]
        return r
    def get_cpnl(self, pnl):
        return pnl.cumsum()
    def get_score(self, pnl):
        return pnl.sum()
    def get_trade(self, signs):
        trades = np.where((signs != 0) & (signs.shift() == 0), 1, 0)
        trades = pd.DataFrame(trades, columns=signs.columns, index=signs.index)
        return trades
    def get_num_trade(self, trades):
        return trades.sum()
    def get_adjust_score(self, score, num_trades):
        df = pd.concat([score, num_trades], axis=1)
        df.columns = ['pnl', 'trade']
        df['adjust'] = np.where(df['pnl'] >= 0, df['trade'] * -0.01, df['trade'] * 0.01)
        df['adjust_pnl'] = df['pnl'] + df['adjust']
        df['adjust_pnl'] = np.where(df['pnl'] / df['adjust_pnl'] >= 0, df['adjust_pnl'], 0)
        return df
    def get_result(self, adjust_score, benchmark=None):
        df = deepcopy(adjust_score['adjust_pnl'])
        df.index = strings_to_columns(df.index)
        df = df.unstack(level=-1)
        df.columns = df.columns.astype(int)
        df = df.T.sort_index()        
        if benchmark is not None:
            df['benchmark'] = benchmark.sum()
        return df
    def run(self):
        st.set_page_config(self.PAGE_TITLE, layout="wide")
        st.title(self.CONTENT_TITLE)
        indicator_name = self.select_indicator_name()
        underlying = self.select_underlying(indicator_name)
        ohlcv = self.get_ohlcv(underlying)
        if ohlcv is not None:
            chart_type = 'Line Chart'
            self.show_candlestick(chart_type, ohlcv)
            benchmark = self.get_benchmark(ohlcv)
            indicator = self.get_collection(indicator_name, underlying)
            st.write('This is CBBC data, postitive values mean Bulls are more, negative values mean Bears are more')
            st.dataframe(indicator)
            indicator = self.get_pca(indicator)
            # signs = self.get_signs(indicator)
            # st.dataframe(signs)
            last = self.get_signs_of_last(indicator)
            trade = self.get_trade(last)
            num_trade = self.get_num_trade(trade)
            pnl = self.get_pnl(benchmark, last)
            # st.write(f'This is Pnl if values equal to the last row')
            st.dataframe(pnl, use_container_width=True)
            score = self.get_score(pnl)
            adjust_score = self.get_adjust_score(score, num_trade)
            st.dataframe(adjust_score, use_container_width=True)
            result = self.get_result(adjust_score, benchmark)
            st.line_chart(result, use_container_width=True)

if __name__ == '__main__':
    
    i = StReturns()
    j = i.run()