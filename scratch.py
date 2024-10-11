import pandas as pd
import yfinance as yf
import os
import time

os.chdir(r'C:\Users\pabatoon\Downloads\weekly-consolidation-analysis-main')
print("Current working directory:", os.getcwd())

#% Obtain tickers from raw nasdaq table. With data cleanup
def clean_metadata(metadata_csv):
    metadata = pd.read_csv(metadata_csv)
    metadata.dropna(subset=['Market Cap'], inplace=True)
    metadata = metadata.sort_values(by='Market Cap', ascending=False)
    metadata['Symbol'] = metadata['Symbol'].str.replace('/', '-')
    metadata.reset_index(drop=True, inplace=True)
    return metadata 

def get_tickers(metadata, minval=0, maxval=2000):
    tickers = metadata['Symbol'][minval:maxval].tolist()
    return tickers


#% Download stock data from yfinance
def download_data_wk(tickers):
    data = yf.download(tickers, period='1y', interval='1wk', auto_adjust=True, progress=True)
    return data

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

#%% Sidebar Layout

with st.Sidebar




#%%
metadata_csv = 'nasdaq_screener_1727589550419.csv'
metadata = clean_metadata(metadata_csv)
tickers = get_tickers(metadata, minval=0, maxval=2000) 
#tickers = ['WBD','CMCSA','PBR','CNH','LUV']
data_wk = download_data_wk(tickers)

#%%
start = time.time()
sqeezes_wk = scanner_wk(data_wk)
stop = time.time()
print(f'{stop-start} sec')
