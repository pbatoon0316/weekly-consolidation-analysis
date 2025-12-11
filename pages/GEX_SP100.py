import streamlit as st
import yfinance as yf
import pandas as pd
from math import log, sqrt
from scipy.stats import norm
import plotly.express as px

# ------------------------------------------------------
# Sidebar config (title removed per request)
# ------------------------------------------------------
st.sidebar.header("⚙️ Screener Settings")

# ------------------------------------------------------
# 1. Universe definition (from metadata CSV)
# ------------------------------------------------------
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

metadata_csv = 'nasdaq_screener_1758565298874.csv'
metadata = clean_metadata(metadata_csv)
tickers = get_tickers(metadata, minval=0, maxval=100) 

SP100_TICKERS = tickers

# Optional: show universe size
st.sidebar.write(f"Universe size: {len(SP100_TICKERS)} tickers")

# ------------------------------------------------------
# 2. Shared Greeks: risk-free rate, Gamma, Vanna
# ------------------------------------------------------
# Uses SR1 futures as an estimate for the risk-free rate
try:
    sr1_close = yf.Ticker("SR1=F").info.get("previousClose", None)
    if sr1_close is None:
        raise ValueError("No previousClose in SR1=F.info")
    r = (100 - sr1_close) / 100
except Exception:
    # Fallback if futures info fails
    r = 0.05

def compute_gamma(row, S: float, r: float) -> float:
    """
    Black-Scholes gamma for a single option row.
    Expects: row['strike'], row['impliedVolatility'], row['expiration']
    """
    try:
        K = float(row["strike"])
        sigma = float(row["impliedVolatility"])
        if pd.isna(sigma) or sigma <= 0:
            return 0.0

        T = max(
            (pd.to_datetime(row["expiration"]) - pd.Timestamp.now()).total_seconds()
            / (365 * 24 * 60 * 60),
            1e-6,
        )

        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * sqrt(T))
        return gamma
    except Exception:
        return 0.0

def compute_vanna(row, S: float, r: float) -> float:
    """
    Simple Vanna proxy (dVega/dS), consistent with your existing page:
    vanna = -d1 * N'(d1) * T / sigma
    """
    try:
        K = float(row["strike"])
        sigma = float(row["impliedVolatility"])
        if pd.isna(sigma) or sigma <= 0:
            return 0.0

        T = max(
            (pd.to_datetime(row["expiration"]) - pd.Timestamp.now()).total_seconds()
            / (365 * 24 * 60 * 60),
            1e-6,
        )

        d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
        vanna = -d1 * norm.pdf(d1) * T / sigma
        return vanna
    except Exception:
        return 0.0

# ------------------------------------------------------
# 3. Core screener logic
#    - Per-ticker function is cached
# ------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def compute_ticker_gex_vex(symbol: str) -> dict | None:
    """
    Compute GEX/VEX aggregates for a single ticker (nearest expiration),
    normalized to $M per 1% move in the underlying.
    """
    try:
        tkr = yf.Ticker(symbol)

        # 1) Nearest expiration
        expirations = tkr.options
        if not expirations:
            return None

        expirations_sorted = sorted(
            expirations, key=lambda d: pd.to_datetime(d)
        )
        nearest_exp = expirations_sorted[0]

        # 2) Option chain for nearest expiration
        chain = tkr.option_chain(nearest_exp)
        calls_raw = chain.calls
        puts_raw = chain.puts

        if (calls_raw is None or calls_raw.empty) and (
            puts_raw is None or puts_raw.empty
        ):
            return None

        # 3) Underlying price (avoid full history if possible)
        underlying_price = None
        try:
            fast_info = getattr(tkr, "fast_info", None)
            if fast_info is not None:
                underlying_price = fast_info.get("last_price", None)
        except Exception:
            underlying_price = None

        if underlying_price is None or pd.isna(underlying_price) or underlying_price <= 0:
            hist = tkr.history(period="1d")
            if hist.empty:
                return None
            underlying_price = float(hist["Close"].iloc[-1])

        # 4) Normalize calls/puts dataframes
        def _prep(df: pd.DataFrame, opt_type: str) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame()

            cols = ["strike", "openInterest", "impliedVolatility"]
            cols = [c for c in cols if c in df.columns]
            df = df[cols].copy()

            if "openInterest" in df.columns:
                df.rename(columns={"openInterest": "oi"}, inplace=True)

            df["type"] = opt_type
            df["expiration"] = nearest_exp

            # We only care about contracts with open interest
            if "oi" in df.columns:
                df = df[df["oi"] > 0]

            return df

        calls = _prep(calls_raw, "call")
        puts = _prep(puts_raw, "put")

        if calls.empty and puts.empty:
            return None

        opt_df = pd.concat([calls, puts], ignore_index=True)

        # Ensure required columns exist
        required = {"strike", "oi", "impliedVolatility", "type", "expiration"}
        if not required.issubset(opt_df.columns):
            return None

        # 5) Greeks for each option
        opt_df["gamma"] = opt_df.apply(
            lambda row: compute_gamma(row, underlying_price, r), axis=1
        )
        opt_df["vanna"] = opt_df.apply(
            lambda row: compute_vanna(row, underlying_price, r), axis=1
        )

        # 6) GEX and VEX per contract, in $M per $1 move
