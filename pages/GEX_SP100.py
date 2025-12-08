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
#    - Outer loop can report active ticker
# ------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def compute_ticker_gex_vex(symbol: str) -> dict | None:
    """
    Compute GEX/VEX aggregates for a single ticker (nearest expiration).
    Returns a dict with summary metrics or None if no usable data.
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

        # 6) GEX and VEX per contract, in $M
        direction = opt_df["type"].map({"call": 1, "put": -1})
        opt_df["gex"] = (
            opt_df["gamma"] * opt_df["oi"] * 100.0 * underlying_price * direction
        ) / 1_000_000.0

        opt_df["vex"] = (
            opt_df["vanna"] * opt_df["oi"] * 100.0 * underlying_price
        ) / 1_000_000.0

        # 7) Aggregations (ticker-level)
        call_oi = float(opt_df.loc[opt_df["type"] == "call", "oi"].sum())
        put_oi = float(opt_df.loc[opt_df["type"] == "put", "oi"].sum())
        total_oi = call_oi + put_oi
        if total_oi == 0:
            return None

        net_gex = float(opt_df["gex"].sum())
        net_vex = float(opt_df["vex"].sum())
        put_call_ratio = (put_oi / call_oi) if call_oi > 0 else None

        # 8) Company name (optional; 1 extra ping per ticker)
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
            "GEX_M": net_gex,
            "VEX_M": net_vex,
        }

    except Exception:
        return None

# ------------------------------------------------------
# 4. Scatter: VEX vs GEX, bubble size = total contracts
# ------------------------------------------------------
def vex_vs_gex_scatter(df: pd.DataFrame):
    fig = px.scatter(
        df,
        x="GEX_M",
        y="VEX_M",
        size="Total_OI",
        hover_name="Ticker",
        hover_data={
            "Name": True,
            "Call_OI": ":,.0f",
            "Put_OI": ":,.0f",
            "Total_OI": ":,.0f",
            "PutCall": ".2f",
        },
        labels={
            "GEX_M": "Net GEX ($M)",
            "VEX_M": "Net VEX ($M)",
        },
    )
    fig.update_layout(
        xaxis_title="Net GEX ($M)",
        yaxis_title="Net VEX ($M)",
        legend_title="",
    )
    return fig

# ------------------------------------------------------
# 5. UI wiring
#    - No main title
#    - Report active ticker
#    - Simple dataframe (no Styler, no matplotlib)
#    - 2-column layout: left = table, right = Plotly scatter
# ------------------------------------------------------
if not SP100_TICKERS:
    st.warning(
        "No tickers found in SP100_TICKERS / metadata. Please verify the metadata file and ticker slice."
    )
else:
    run_btn = st.sidebar.button("Run SP100 GEX/VEX Screener")

    if run_btn:
        universe = tuple(sorted(set(SP100_TICKERS)))
        n = len(universe)

        status = st.empty()  # Shows actively processed ticker

        with st.spinner("Running GEX/VEX screener on nearest expirations..."):
            results = []
            for i, symbol in enumerate(universe, start=1):
                # Report which ticker is actively being processed
                status.write(f"Processing {symbol} ({i}/{n})")
                row = compute_ticker_gex_vex(symbol)
                if row is not None:
                    results.append(row)

        status.write("Processing complete.")

        if not results:
            st.warning("No options data returned for the current universe.")
        else:
            screener_df = (
                pd.DataFrame(results)
                .sort_values("GEX_M", ascending=False)
                .reset_index(drop=True)
            )

            # Prepare a cleaner display table
            display_df = screener_df.copy()
            display_df["Put/Call"] = display_df["PutCall"]
            display_df["GEX ($M)"] = display_df["GEX_M"]
            display_df["VEX ($M)"] = display_df["VEX_M"]
            display_df = display_df[
                [
                    "Ticker",
                    "Name",
                    "Call_OI",
                    "Put_OI",
                    "Total_OI",
                    "Put/Call",
                    "GEX ($M)",
                    "VEX ($M)",
                ]
            ]

            # 2-column layout: left = dataframe, right = scatter plot
            col1, col2 = st.columns(2, gap="medium")

            with col1:
                st.subheader("Ticker-level GEX/VEX (Nearest Expiration)")
                st.dataframe(display_df, use_container_width=True)

            with col2:
                st.subheader("VEX vs GEX (bubble size = total contracts)")
                fig = vex_vs_gex_scatter(screener_df)
                st.plotly_chart(fig, use_container_width=True)
