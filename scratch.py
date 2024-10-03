import pandas as pd
import yfinance as yf
from finta import TA
import math
import os

os.chdir(r'C:\Users\pbato\OneDrive\Documents\Python-Streamlit\weekly-consolidation-analysis-main')
print("Current working directory:", os.getcwd())

#%% Obtain tickers from raw nasdaq table. With data cleanup
def clean_metadata(metadata_csv):
    metadata = pd.read_csv(metadata_csv)
    metadata.dropna(subset=['Market Cap'], inplace=True)
    metadata = metadata.sort_values(by='Market Cap', ascending=False)
    metadata['Symbol'] = metadata['Symbol'].str.replace('/', '.')
    metadata.reset_index(drop=True, inplace=True)
    return metadata 

def get_tickers(metadata, minval=0, maxval=2000):
    tickers = metadata['Symbol'][minval:maxval].tolist()
    return tickers

metadata_csv = 'nasdaq_screener_1727589550419.csv'
metadata = clean_metadata(metadata_csv)
tickers =  get_tickers(metadata, minval=0, maxval=2000) ###['WBD','CMCSA','PBR','CNH','LUV']


#%% Download stock data from yfinance

def download_data_wk(tickers):
    data = yf.download(tickers, period='6mo', interval='1wk', auto_adjust=True, progress=True)
    return data

data_wk = download_data_wk(tickers)

#%% Input stock data into squeeze screener

def scanner_wk(data):
    tickers = list(data.columns.get_level_values(1).unique())
    squeezes = pd.DataFrame()

    for ticker in tickers:
        print(ticker)
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
        df['ASCENDING'] = (df.AO.iloc[-1] > df.AO.iloc[-2]) & (df.AO.iloc[-2] > df.AO.iloc[-3]) & (df.AO.iloc[-3] > df.AO.iloc[-4])

        df.dropna(inplace=True)

        if not df.empty and df.SQUEEZE.iloc[-1] and df.ASCENDING.iloc[-1]:
            squeezes = pd.concat([squeezes, df.iloc[[-1]]])
        else:
            pass

    squeezes = squeezes.sort_values('volume_average', ascending=False)

    return squeezes

sqeezes_wk = scanner_wk(data_wk)
