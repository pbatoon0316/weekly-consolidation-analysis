import pandas as pd
import yfinance as yf
import mplfinance as mpf
import warnings
import streamlit as st
import streamlit.components.v1 as components


######################
# Set the display option to show 2 decimal places
pd.set_option('display.float_format', '{:.2f}'.format)
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title='Volatility-Momentum Surge',
                   page_icon='⚡', 
                   layout="wide")

@st.cache_data(ttl='1d')
def download_metadata():
    url = 'metadata.csv'
    metadata = pd.read_csv(url)
    return metadata

@st.cache_data(ttl='1hr')
def download_data():
    url = 'metadata.csv'
    stocks = pd.read_csv(url)
    tickers = stocks['Symbol'].tolist()
    data = yf.download(tickers, period='90d', auto_adjust=True, progress=True)
    return data

@st.cache_data(ttl='1hr')
def scanner(data,threshold=2):
    tickers = list(data.columns.get_level_values(1).unique())
    breakouts = pd.DataFrame()
    period = 20

    len_tickers = len(tickers)
    idx_ticker = 1
    for ticker in tickers:

        df = data.loc[:, (slice(None), ticker)].copy()
        df.columns = df.columns.droplevel(1)
        df['ticker'] = ticker

        df['%change'] = df['Close'].pct_change()
        df['%change_zscore'] = (df['%change'] - df['%change'].shift(1).rolling(period).mean()) / df['%change'].shift(1).std()

        df['Volume(M)'] = df['Volume'] / 1000000
        df['volume_average'] = df['Volume(M)'].mean()
        df['volume_zscore'] = (df['Volume(M)'] - df['Volume(M)'].shift(1).mean()) / df['Volume(M)'].shift(1).std()

        df = df[['ticker','%change_zscore','volume_zscore','volume_average','Volume(M)']]

        threshold = threshold

        if (df['%change_zscore'].iloc[-1] > threshold) & (df['volume_zscore'].iloc[-1] > threshold):
            breakouts = pd.concat([breakouts, df.iloc[[-1]]])
        else:
            pass

    breakouts = breakouts.sort_values('volume_average', ascending=False)

    return breakouts

def plot_ticker(ticker):
    df = data.loc[:, (slice(None), ticker)].copy()
    df.columns = df.columns.droplevel(1)

    # Create the plot
    fig, axes = mpf.plot(df, type='candle', volume=True, ylabel='Price', volume_panel=1, style='yahoo', returnfig=True, panel_ratios=(2,1), figsize=(8, 5), tight_layout=True)

    # Add watermark as text annotation
    watermark_text = ticker
    company_name = metadata[metadata['Symbol']==ticker]['Name'].item()
    company_sector = metadata[metadata['Symbol']==ticker]['Sector'].item()
    avg_vol = breakouts[breakouts['ticker']==ticker]['volume_average'].item()
    
    watermark_position = (0.3, 0.65)  # Adjust the position as per your preference
    fig.text(watermark_position[0], watermark_position[1], watermark_text, fontsize=80, color='black', alpha=0.1)

    st.markdown(f'''{round(avg_vol,2)}M - {ticker} - {company_sector} [[Finviz]](https://finviz.com/quote.ashx?t={ticker}&p=d) [[Profitviz]](https://profitviz.com/{ticker})''')

    return fig

def plot_ticker_html(ticker):

    df = data.loc[:, (slice(None), ticker)].copy()
    df.columns = df.columns.droplevel(1)

    company_name = metadata[metadata['Symbol']==ticker]['Name'].item()
    company_sector = metadata[metadata['Symbol']==ticker]['Sector'].item()
    avg_vol = breakouts[breakouts['ticker']==ticker]['volume_average'].item()

    st.markdown(f'''{round(avg_vol,2)}M - {ticker} - {company_sector} [[Finviz]](https://finviz.com/quote.ashx?t={ticker}&p=d) [[Profitviz]](https://profitviz.com/{ticker})''')
    
    fig_html =f'''
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container">
        <div class="tradingview-widget-container__widget"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {{
        "height": "290",
        "symbol": "{ticker}",
        "interval": "D",
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
        "support_host": "https://www.tradingview.com"
        }}
        </script>
    </div>
    <!-- TradingView Widget END -->
    '''
    return fig_html

######################

left_datacontainer, right_resultcontainer = st.columns([1,2])

##### Data download & Calculations #####

with left_datacontainer:
    with st.expander('Metadata'):
        metadata = download_metadata()
        st.text(f'{len(metadata)} tickers inputted')
        st.dataframe(metadata, hide_index=True)

    with st.expander('Data (initial download takes roughly 5 minutes)'):
        data = download_data()
        st.dataframe(data)

    ##### User Input for Z score threshold and Lookback period #####
    threshcol, lookbackcol =  st.columns([1,1])
    with threshcol:
        threshold = st.number_input('Z-Score (default=2)', value=2.0)
    with lookbackcol:
        lookback = st.number_input('Lookback (default=0)', value=0, min_value=0, max_value=50)
        lookback = lookback*-1

    if lookback == 0:
        breakouts = scanner(data,threshold)

    ##### Special condition if a lookback period is added. Loop through lookback += 1 with combined list #####
    else:
        breakouts = pd.DataFrame()
        while lookback < 0:
            breakouts_temp = scanner(data[:lookback],threshold)
            breakouts = pd.concat([breakouts, breakouts_temp])
            lookback += 1
        breakouts_temp = scanner(data,threshold)
        breakouts = pd.concat([breakouts, breakouts_temp])

    st.markdown('Breakouts')
    breakouts = breakouts.reset_index()
    breakouts = breakouts.sort_values(by=['Date','volume_average'], ascending=False)
    breakouts = breakouts.set_index('Date')
    st.dataframe(breakouts, hide_index=False)



##### Plotting charts in Mid & Right columns #####

with right_resultcontainer:

    left_resultsplot, right_resultsplot = st.columns([1,1])

    i = 0
    for ticker in breakouts.ticker.unique():
        if i % 2 == 0:
            with left_resultsplot:
                fig = plot_ticker_html(ticker)
                components.html(fig, height=300)
                i += 1
        else:
            with right_resultsplot:
                fig = plot_ticker_html(ticker)
                components.html(fig, height=300)
                i += 1
