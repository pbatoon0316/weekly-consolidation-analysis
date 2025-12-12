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
try:
    sr1_close = yf.Ticker("SR1=F").info.get("previousClose", None)
    if sr1_close is None:
        raise ValueError("No previousClose in SR1=F.info")
    r = (100 - sr1_close) / 100
except Exception:
    r = 0.05

def compute_gamma(row, S: float, r: float) -> float:
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
# 3. Expiration dropdown (YYYY-MM-DD) from yfinance
# ------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def get_expirations_for_ticker(symbol: str) -> list[str]:
    try:
        opts = yf.Ticker(symbol).options
        return sorted(opts, key=lambda d: pd.to_datetime(d)) if opts else []
    except Exception:
        return []

ref_ticker = SP100_TICKERS[0] if SP100_TICKERS else "SPY"
expirations_list = get_expirations_for_ticker(ref_ticker)

if expirations_list:
    selected_expiration = st.sidebar.selectbox(
        "Options expiration (YYYY-MM-DD)",
        expirations_list,
        index=0,
        help=f"Pulled from {ref_ticker}. Applied to all tickers (tickers without this expiry are skipped)."
    )
else:
    selected_expiration = None
    st.sidebar.warning(f"No expirations found from reference ticker {ref_ticker}. Using nearest expiry per ticker.")

# ------------------------------------------------------
# 4. Core screener logic
#    - Per-ticker function is cached
#    - Cache key includes expiration
# ------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def compute_ticker_gex_vex(symbol: str, expiration: str | None) -> dict | None:
    """
    Compute GEX/VEX aggregates for a single ticker,
    normalized to $M per 1% move in the underlying.

    If expiration is provided, uses that chain date.
    If not, uses the nearest available expiration per ticker.
    """
    try:
        tkr = yf.Ticker(symbol)

        expirations = tkr.options
        if not expirations:
            return None

        if expiration is None:
            expirations_sorted = sorted(expirations, key=lambda d: pd.to_datetime(d))
            chosen_exp = expirations_sorted[0]
        else:
            if expiration not in expirations:
                return None
            chosen_exp = expiration

        chain = tkr.option_chain(chosen_exp)
        calls_raw = chain.calls
        puts_raw = chain.puts

        if (calls_raw is None or calls_raw.empty) and (puts_raw is None or puts_raw.empty):
            return None

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

        def _prep(df: pd.DataFrame, opt_type: str) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame()

            cols = ["strike", "openInterest", "impliedVolatility"]
            cols = [c for c in cols if c in df.columns]
            df = df[cols].copy()

            if "openInterest" in df.columns:
                df.rename(columns={"openInterest": "oi"}, inplace=True)

            df["type"] = opt_type
            df["expiration"] = chosen_exp

            if "oi" in df.columns:
                df = df[df["oi"] > 0]

            return df

        calls = _prep(calls_raw, "call")
        puts = _prep(puts_raw, "put")

        if calls.empty and puts.empty:
            return None

        opt_df = pd.concat([calls, puts], ignore_index=True)

        required = {"strike", "oi", "impliedVolatility", "type", "expiration"}
        if not required.issubset(opt_df.columns):
            return None

        opt_df["gamma"] = opt_df.apply(lambda row: compute_gamma(row, underlying_price, r), axis=1)
        opt_df["vanna"] = opt_df.apply(lambda row: compute_vanna(row, underlying_price, r), axis=1)

        direction = opt_df["type"].map({"call": 1, "put": -1})

        # $M per $1 move
        opt_df["gex"] = (opt_df["gamma"] * opt_df["oi"] * 100.0 * underlying_price * direction) / 1_000_000.0
        opt_df["vex"] = (opt_df["vanna"] * opt_df["oi"] * 100.0 * underlying_price) / 1_000_000.0

        # Normalize to $M per 1% move
        scale = 0.01 * underlying_price
        opt_df["gex_1pct"] = opt_df["gex"] * scale
        opt_df["vex_1pct"] = opt_df["vex"] * scale

        call_oi = float(opt_df.loc[opt_df["type"] == "call", "oi"].sum())
        put_oi = float(opt_df.loc[opt_df["type"] == "put", "oi"].sum())
        total_oi = call_oi + put_oi
        if total_oi == 0:
            return None

        net_gex = float(opt_df["gex_1pct"].sum())
        net_vex = float(opt_df["vex_1pct"].sum())
        put_call_ratio = (put_oi / call_oi) if call_oi > 0 else None

        try:
            info = tkr.info or {}
            name = info.get("shortName", symbol)
        except Exception:
            name = symbol

        return {
            "Ticker": symbol,
            "Name": name,
            "Call_OI": call_oi,
            "Put_OI": put_oi,
            "Total_OI": total_oi,
            "PutCall": put_call_ratio,
            "GEX_M": net_gex,  # $M per 1% move
            "VEX_M": net_vex,  # $M per 1% move
            "Expiration": chosen_exp,
        }

    except Exception:
        return None

# ------------------------------------------------------
# 5. Scatter helper
# ------------------------------------------------------
def vex_vs_gex_scatter(df: pd.DataFrame, x_col: str, y_col: str,
                       x_label: str, y_label: str):
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size="Total_OI",
        hover_name="Ticker",
        hover_data={
            "Name": True,
            "Expiration": True,
            "Call_OI": ":,.0f",
            "Put_OI": ":,.0f",
            "Total_OI": ":,.0f",
            "PutCall": ".2f",
        },
        labels={x_col: x_label, y_col: y_label},
    )
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
        legend_title="",
    )
    return fig

# ------------------------------------------------------
# 6. Session state for results (avoid recomputation)
# ------------------------------------------------------
if "screener_df" not in st.session_state:
    st.session_state["screener_df"] = None

# ------------------------------------------------------
# 7. Sidebar cache clear button + Run button (reverted flow)
# ------------------------------------------------------
clear_cache_btn = st.sidebar.button("Clear cached GEX/VEX computations")

if clear_cache_btn:
    compute_ticker_gex_vex.clear()
    get_expirations_for_ticker.clear()
    st.session_state["screener_df"] = None
    st.sidebar.success("Cleared cached computations. Click 'Run' to recompute.")

run_btn = st.sidebar.button("Run SP100 GEX/VEX Screener")

if run_btn:
    universe = tuple(sorted(set(SP100_TICKERS)))
    n = len(universe)

    status = st.empty()

    with st.spinner("Running GEX/VEX screener..."):
        results = []
        for i, symbol in enumerate(universe, start=1):
            status.write(f"Processing {symbol} ({i}/{n})")
            row = compute_ticker_gex_vex(symbol, selected_expiration)
            if row is not None:
                results.append(row)

    status.write("Processing complete.")

    if not results:
        st.warning("No options data returned for the current universe (or selected expiration not available).")
        st.session_state["screener_df"] = None
    else:
        screener_df = (
            pd.DataFrame(results)
            .sort_values("GEX_M", ascending=False)
            .reset_index(drop=True)
        )
        st.session_state["screener_df"] = screener_df

# ------------------------------------------------------
# 8. Display section: uses session_state dataframe only
# ------------------------------------------------------
screener_df = st.session_state["screener_df"]

if screener_df is not None and not screener_df.empty:
    mc_df = metadata[["Symbol", "Market Cap"]].rename(columns={"Symbol": "Ticker"})
    screener_df = screener_df.merg
