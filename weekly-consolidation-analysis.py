import warnings
import time
import math

import pandas as pd
import yfinance as yf
import streamlit as st
import streamlit.components.v1 as components

#%% Set the display option to show 2 decimal places

pd.set_option('display.float_format', '{:.2f}'.format)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title='Weekly Consolidation Screener',
                   page_icon='📇', 
                   layout="wide")

#%% Obtain tickers from raw nasdaq table. With data cleanup

@st.cache_resource(ttl='12hr')
def clean_metadata(metadata_csv):
    metadata = pd.read_csv(metadata_csv)
    metadata.dropna(subset=['Market Cap'], inplace=True)
    metadata = metadata.sort_values(by='Market Cap', ascending=False)
    metadata['Symbol'] = metadata['Symbol'].str.replace('/', '-')
    metadata['ticker'] = metadata['Symbol']
    metadata.reset_index(drop=True, inplace=True)
    return metadata 

@st.cache_resource(ttl='12hr')
def get_tickers(metadata, minval=0, maxval=2000):
    tickers = metadata['Symbol'][minval:maxval].tolist()
    return tickers

@st.cache_resource(ttl='12hr')
#% Download stock data from yfinance
def download_data_wk(tickers):
    data = yf.download(tickers, period='1y', interval='1wk', auto_adjust=True, progress=True)
    return data

@st.cache_data(ttl='12hr')
#% Input stock data into squeeze screener
def scanner_wk(data):
    tickers = list(data.columns.get_level_values(1).unique())
    squeezes = pd.DataFrame()

    for ticker in tickers:
        df = data.loc[:, (slice(None), ticker)].copy()
        
        df.dropna(inplace=True)
        df.columns = df.columns.droplevel(1)
        df.columns = df.columns.str.lower()
        df['ticker'] = ticker
        
        if df.empty or len(df)<5:
            continue

        # Volume
        df['volume_average'] = df['volume'].mean()
        df['volume_zscore'] = (df['volume'] - df['volume'].shift(1).mean()) / df['volume'].shift(1).std()
        
        #Exponential Moving Averages
        df['EMA50'] = df['close'].ewm(50).mean()
        df['EMA20'] = df['close'].ewm(20).mean()
        df['EMA10'] = df['close'].ewm(10).mean()
        
        # Keltner Channel
        df['ATR'] = (df['high'] - df['low']).rolling(20).mean()
        df['KC_UPPER'] = df['EMA20'] + 1.5*df['ATR']
        df['KC_LOWER'] = df['EMA20'] - 1.5*df['ATR']
        # Bollinger Bands
        df['BB_UPPER'] = df['EMA20'] + 2*df['close'].rolling(20).std()
        df['BB_LOWER'] = df['EMA20'] - 2*df['close'].rolling(20).std()
        
        # Awesome Oscillator
        df['AO'] =  df['EMA10'] - df['EMA20']

        # Conditions
        df['SQUEEZE'] = (df.BB_LOWER >= df.KC_LOWER) | (df.BB_UPPER <= df.KC_UPPER)
        df['ASCENDING'] = (df.close.iloc[-1] > df.EMA50.iloc[-1]) & (df.AO.iloc[-1] > df.AO.iloc[-2]) #& (df.AO.iloc[-2] > df.AO.iloc[-3]) & (df.AO.iloc[-3] > df.AO.iloc[-4])

        df.dropna(inplace=True)

        if not df.empty and df.SQUEEZE.iloc[-1] and df.ASCENDING.iloc[-1]:
            squeezes = pd.concat([squeezes, df.iloc[[-1]]])
        else:
            pass

    squeezes = squeezes.sort_values('volume_average', ascending=False)

    return squeezes

# Plot TradingView charts for each ticker
def plot_ticker_html(ticker='SPY',interval='W'):
    time.sleep(1)
    st.markdown(f'''{ticker} - [[Finviz]](https://finviz.com/quote.ashx?t={ticker}&p=d) [[Profitviz]](https://profitviz.com/{ticker})''')
    
    fig_html = f'''

    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
      <div class="tradingview-widget-container__widget"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
      {{
        "autosize": true,
        "height": "290",
        "symbol": "{ticker}",
        "interval": "{interval}",
        "timezone": "Etc/UTC",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "hide_top_toolbar": true,
        "allow_symbol_change": false,
        "save_image": false,
        "calendar": false,
        "studies": [
          "STD;Bollinger_Bands"
        ],
        "support_host": "https://www.tradingview.com"
      }}
      </script>
    </div>
    <!-- TradingView Widget END -->
    '''
    return fig_html

#%% Load & download data

metadata_csv = 'nasdaq_screener_1727589550419.csv'
metadata = clean_metadata(metadata_csv)
tickers = get_tickers(metadata, minval=0, maxval=2200) 
data_wk = download_data_wk(tickers)

#%% Process & screen
squeezes_wk = scanner_wk(data_wk)
squeezes_wk = squeezes_wk.merge(metadata[['ticker','Name','Market Cap','Sector','Industry']], how='left', on='ticker')
squeezes_wk = squeezes_wk.loc[squeezes_wk['volume_average']>250000]
squeezes_wk = squeezes_wk.sort_values(by=['Market Cap','volume_average'], ascending=False).dropna()


#%% Sidebar Layout

sector_filter = st.sidebar.selectbox(label='Filter by Sector', 
                                         options=squeezes_wk['Sector'].unique(),
                                         index=None)

if sector_filter == None:
    filter_squeezes_wk = squeezes_wk
else:
    filter_squeezes_wk = squeezes_wk.loc[squeezes_wk['Sector']==sector_filter]


num_plots_day = st.sidebar.number_input(f'Display Num. Plots (max = {len(filter_squeezes_wk)})', 
                                        min_value=1, max_value=len(filter_squeezes_wk), 
                                        value=math.ceil(0.10*len(filter_squeezes_wk)))
st.sidebar.divider()

st.sidebar.expander('Weekly Squeeze Reults').dataframe(filter_squeezes_wk[['ticker','Name','close','Market Cap','volume_average','Sector','Industry']])

sector_counts = squeezes_wk[['Sector','ticker']].groupby('Sector').count()
st.sidebar.text(sector_counts)


#%% Result Layout

mobile_view = st.toggle('Mobile View', value=False)
result_cols = st.columns(3)

if mobile_view == False:
    i = 0
    for ticker in filter_squeezes_wk[:num_plots_day].ticker.tolist():
        with result_cols[i]:
            try:
                fig = plot_ticker_html(ticker=ticker,interval='W')
                components.html(fig, height=300)
            except:
                st.markdown(f'{ticker} - [[Finviz]](https://finviz.com/quote.ashx?t={ticker}&p=d) [[Profitviz]](https://profitviz.com/{ticker})')
            i += 1
        if i == 3:
            i = 0
        else:
            pass

if mobile_view == True:
    for ticker in filter_squeezes_wk[:num_plots_day].ticker.tolist():
        try:
            fig = plot_ticker_html(ticker=ticker,interval='W')
            components.html(fig, height=300)
        except:
            st.markdown(f'{ticker} - [[Finviz]](https://finviz.com/quote.ashx?t={ticker}&p=d) [[Profitviz]](https://profitviz.com/{ticker})')

