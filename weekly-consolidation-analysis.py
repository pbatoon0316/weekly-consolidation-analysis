import pandas as pd
import yfinance as yf
from finta import TA
import warnings
import streamlit as st
import streamlit.components.v1 as components

######################
# Set the display option to show 2 decimal places
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title='Weekly Consolidation Analysis',
                   page_icon='🙏', 
                   layout="wide")

@st.cache_data(ttl='1d')
def download_metadata():
    url = 'metadata_squeeze.csv'
    metadata = pd.read_csv(url)
    return metadata

@st.cache_data(ttl='12hr')
def download_data_wk():
    url = 'metadata_squeeze.csv'
    stocks = pd.read_csv(url)
    tickers = stocks['Symbol'].tolist()
    data = yf.download(tickers, period='6mo', interval='1wk', auto_adjust=True, progress=True)
    return data

@st.cache_data(ttl='12hr')
def scanner_wk(data):
    tickers = list(data.columns.get_level_values(1).unique())
    squeezes = pd.DataFrame()

    for ticker in tickers:
        df = data.loc[:, (slice(None), ticker)].copy()
        df.columns = df.columns.droplevel(1)
        df.columns = df.columns.str.lower()
        df['ticker'] = ticker

        # Volume
        df['volume_average'] = df['volume'].mean()
        df['volume_zscore'] = (df['volume'] - df['volume'].shift(1).mean()) / df['volume'].shift(1).std()
        # KC
        KC = TA.KC(df, period=20, atr_period=20, kc_mult=1.5)
        # BB
        BB = TA.BBANDS(df, period=20, std_multiplier=2)
        # AO
        AO = TA.AO(df, slow_period=20, fast_period=10)
        # Concatenate
        df = pd.concat([df, KC, BB, AO], axis=1)

        # Conditions
        df['SQUEEZE'] = (df.BB_LOWER >= df.KC_LOWER) | (df.BB_UPPER <= df.KC_UPPER)
        df['ASCENDING'] = df.AO.iloc[-1] > df.AO.iloc[-2]

        df.dropna(inplace=True)

        if not df.empty and df.SQUEEZE.iloc[-1] and df.ASCENDING.iloc[-1]:
            squeezes = pd.concat([squeezes, df.iloc[[-1]]])
        else:
            pass

    squeezes = squeezes.sort_values('volume_average', ascending=False)

    return squeezes

@st.cache_data(ttl='1hr')
def download_data_day(tickers):
    data = yf.download(tickers, period='3mo', interval='1d', auto_adjust=True, progress=True)
    return data

@st.cache_data(ttl='1hr')
def scanner_day(data):
    tickers = list(data.columns.get_level_values(1).unique())
    squeezes = pd.DataFrame()

    for ticker in tickers:
        df = data.loc[:, (slice(None), ticker)].copy()
        df.columns = df.columns.droplevel(1)
        df.columns = df.columns.str.lower()
        df['ticker'] = ticker

        # Volume
        df['volume_average'] = df['volume'].mean()
        df['volume_zscore'] = (df['volume'] - df['volume'].shift(1).mean()) / df['volume'].shift(1).std()
        # KC
        KC = TA.KC(df, period=20, atr_period=20, kc_mult=1.5)
        # BB
        BB = TA.BBANDS(df, period=20, std_multiplier=2)
        # AO
        AO = TA.AO(df, slow_period=20, fast_period=10)
        # Concatenate
        df = pd.concat([df, KC, BB, AO], axis=1)

        # Conditions
        df['SQUEEZE'] = (df.BB_LOWER >= df.KC_LOWER) | (df.BB_UPPER <= df.KC_UPPER)
        df['ASCENDING'] = df.AO.iloc[-1] > df.AO.iloc[-2]

        df.dropna(inplace=True)

        if not df.empty and df.SQUEEZE.iloc[-1] and df.ASCENDING.iloc[-1]:
            squeezes = pd.concat([squeezes, df.iloc[[-1]]])
        else:
            pass

    squeezes = squeezes.sort_values('volume_average', ascending=False)

    return squeezes


def plot_ticker_html(ticker):
    st.markdown(f'''{ticker} - [[Finviz]](https://finviz.com/quote.ashx?t={ticker}&p=d) [[Profitviz]](https://profitviz.com/{ticker})''')
    
    fig_html = f'''
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {{
        "height": "290",
        "symbol": "{ticker}",
        "interval": "W",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "backgroundColor": "rgba(255, 255, 255, 1)",
        "gridColor": "rgba(0, 0, 0, 0.06)",
        "hide_top_toolbar": true,
        "allow_symbol_change": false,
        "save_image": false,
        "calendar": false,
        "studies": [
            "STD;Bollinger_Bands"],
        "support_host": "https://www.tradingview.com"
        }}
        </script>
    </div>
    <!-- TradingView Widget END -->
    '''
    return fig_html

######################

##### Data download & Calculations #####

# 1 Download & process weekly data
st.session_state.metadata = download_metadata()
metadata = st.session_state.metadata
st.session_state.data_wk = download_data_wk()
data_wk = st.session_state.data_wk
st.session_state.squeezes_wk = scanner_wk(data_wk)
squeezes_wk = st.session_state.squeezes_wk

# 2 Download & process daily data
st.session_state.tickers_wk = squeezes_wk.ticker.tolist()
tickers_day = st.session_state.tickers_wk
st.session_state.data_day = download_data_day(tickers_day)
data_day = st.session_state.data_day
st.session_state.squeezes_day = scanner_day(data_day)
squeezes_day = st.session_state.squeezes_day

left_datacontainer, right_resultcontainer = st.columns([1,2])

with left_datacontainer:
    st.markdown('Daily Squeezes')
    st.dataframe(squeezes_day, hide_index=True)

    with st.expander('Weekly Squeezes'):
        st.dataframe(squeezes_wk, hide_index=True)

##### Plotting charts in Mid & Right columns #####
with right_resultcontainer:
    daily_tab, weekly_tab = st.tabs(['Daily Results', 'Weekly Results'])

    with daily_tab:
        num_plots_day = st.number_input('Display Num. Plots', min_value=1, max_value=len(squeezes_day), value=int(0.1*len(squeezes_day)))
        left_resultsplot, right_resultsplot = st.columns([1,1])

        i = 0
        for ticker in squeezes_day[:num_plots_day].ticker.tolist():
            if i % 2 == 0:
                with left_resultsplot:
                    try:
                        fig = plot_ticker_html(ticker)
                        components.html(fig, height=300)
                    except:
                        st.markdown(f'{ticker} - [[Finviz]](https://finviz.com/quote.ashx?t={ticker}&p=d) [[Profitviz]](https://profitviz.com/{ticker})')
                    i += 1
            else:
                with right_resultsplot:
                    try:
                        fig = plot_ticker_html(ticker)
                        components.html(fig, height=300)
                    except:
                        st.markdown(f'{ticker} - [[Finviz]](https://finviz.com/quote.ashx?t={ticker}&p=d) [[Profitviz]](https://profitviz.com/{ticker})')
                    i += 1

    with weekly_tab:
        num_plots_wk = st.number_input('Display Num. Plots', min_value=1, max_value=len(squeezes_wk), value=int(0.1*len(squeezes_wk)))
        left_resultsplot, right_resultsplot = st.columns([1,1])

        i = 0
        for ticker in squeezes_day[:num_plots_wk].ticker.tolist():
            if i % 2 == 0:
                with left_resultsplot:
                    try:
                        fig = plot_ticker_html(ticker)
                        components.html(fig, height=300)
                    except:
                        st.markdown(f'{ticker} - [[Finviz]](https://finviz.com/quote.ashx?t={ticker}&p=d) [[Profitviz]](https://profitviz.com/{ticker})')
                    i += 1
            else:
                with right_resultsplot:
                    try:
                        fig = plot_ticker_html(ticker)
                        components.html(fig, height=300)
                    except:
                        st.markdown(f'{ticker} - [[Finviz]](https://finviz.com/quote.ashx?t={ticker}&p=d) [[Profitviz]](https://profitviz.com/{ticker})')
                    i += 1
           
