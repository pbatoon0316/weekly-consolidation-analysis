import streamlit as st
import yfinance as yf
import pandas as pd
from math import log, sqrt
from scipy.stats import norm
import plotly.express as px

# ------------------------------------------------------
# Config / Title
# ------------------------------------------------------
st.title("SP100 Options GEX/VEX Screener (Nearest Expiration)")

st.sidebar.header("⚙️ Screener Settings")

# ------------------------------------------------------
# 1. Universe definition (you provide this list)
# ------------------------------------------------------
# TODO: Replace this placeholder with your real SP100 list.
# Example: SP100_TICKERS = ["AAPL", "MSFT", "GOOGL", ...]

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
#    (mirrors your existing options page)
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
# 3. Core screener logic (heavily cached to reduce pings)
# ------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def run_gex_vex_screener(tickers: tuple[str, ...]) -> pd.DataFrame:
    """
    For each ticker:
      - Find nearest expiration
      - Pull option chain (calls + puts)
      - Compute per-contract gamma, vanna
      - Compute GEX, VEX in $M
      - Aggregate to ticker-level:
          Call OI, Put OI, Total OI, Put/Call, Net GEX, Net VEX
    Returns a DataFrame with one row per ticker.
    """
    rows = []

    for symbol in tickers:
        try:
            tkr = yf.Ticker(symbol)

            # 1) Nearest expiration
            expirations = tkr.options
            if not expirations:
                continue

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
                continue

            # 3) Underlying price (cheap-ish; avoids full history if possible)
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
                    continue
                underlying_price = float(hist["Close"].iloc[-1])

            # 4) Normalize calls/puts dataframes
            def _prep(df: pd.DataFrame, opt_type: str) -> pd.DataFrame:
                if df is None or df.empty:
                    return pd.DataFrame()

                cols = ["strike", "openInterest", "impliedVolatility"]
                # keep only available columns
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
                continue

            opt_df = pd.concat([calls, puts], ignore_index=True)

            # Ensure required columns exist
            required = {"strike", "oi", "impliedVolatility", "type", "expiration"}
            if not required.issubset(opt_df.columns):
                continue

            # 5) Greeks for each option
            opt_df["gamma"] = opt_df.apply(
                lambda row: compute_gamma(row, underlying_price, r), axis=1
            )
            opt_df["vanna"] = opt_df.apply(
                lambda row: compute_vanna(row, underlying_price, r), axis=1
            )

            # 6) GEX and VEX per contract, in $M
            # GEX = gamma * OI * 100 * S * direction(call +, put -)
            direction = opt_df["type"].map({"call": 1, "put": -1})
            opt_df["gex"] = (
                opt_df["gamma"] * opt_df["oi"] * 100.0 * underlying_price * direction
            ) / 1_000_000.0

            # VEX = vanna * OI * 100 * S
            opt_df["vex"] = (
                opt_df["vanna"] * opt_df["oi"] * 100.0 * underlying_price
            ) / 1_000_000.0

            # 7) Aggregations (ticker-level)
            call_oi = float(opt_df.loc[opt_df["type"] == "call", "oi"].sum())
            put_oi = float(opt_df.loc[opt_df["type"] == "put", "oi"].sum())
            total_oi = call_oi + put_oi
            if total_oi == 0:
                continue

            net_gex = float(opt_df["gex"].sum())
            net_vex = float(opt_df["vex"].sum())
            put_call_ratio = (put_oi / call_oi) if call_oi > 0 else None

            # 8) Company name (optional; 1 extra ping per ticker)
            try:
                info = tkr.info or {}
                name = info.get("shortName", symbol)
            except Exception:
                name = symbol

            rows.append(
                {
                    "Ticker": symbol,
                    "Name": name,
                    "Call_OI": call_oi,
                    "Put_OI": put_oi,
                    "Total_OI": total_oi,
                    "PutCall": put_call_ratio,
                    "GEX_M": net_gex,
                    "VEX_M": net_vex,
                }
            )

        except Exception:
            # Skip tickers that blow up
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Sort by GEX descending (as requested)
    df.sort_values("GEX_M", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ------------------------------------------------------
# 4. Styling: color GEX (green→red), |VEX| (green→red)
# ------------------------------------------------------
def styled_screener_table(df: pd.DataFrame):
    working = df.copy()

    # Pretty columns for display
    working["Put/Call"] = working["PutCall"]
    working["GEX ($M)"] = working["GEX_M"]
    working["VEX ($M)"] = working["VEX_M"]
    working["abs_vex"] = working["VEX_M"].abs()

    display_cols = [
        "Ticker",
        "Name",
        "Call_OI",
        "Put_OI",
        "Put/Call",
        "GEX ($M)",
        "VEX ($M)",
        "abs_vex",  # used only for coloring
    ]

    working = working[display_cols]

    max_abs_gex = float(working["GEX ($M)"].abs().max() or 1.0)
    max_abs_vex = float(working["abs_vex"].max() or 1.0)

    styler = (
        working.style
        # GEX: green for high positive, red for large negative
        .background_gradient(
            cmap="RdYlGn",
            subset=["GEX ($M)"],
            vmin=-max_abs_gex,
            vmax=max_abs_gex,
        )
        # |VEX|: lowest abs → green, highest abs → red
        .background_gradient(
            cmap="RdYlGn_r",
            subset=["abs_vex"],
            vmin=0,
            vmax=max_abs_vex,
        )
        .hide(axis="columns", subset=["abs_vex"])
        .format(
            {
                "Call_OI": "{:,.0f}",
                "Put_OI": "{:,.0f}",
                "Put/Call": "{:.2f}",
                "GEX ($M)": "{:.2f}",
                "VEX ($M)": "{:.2f}",
            }
        )
    )

    return styler

# ------------------------------------------------------
# 5. Scatter: VEX vs GEX, bubble size = total contracts
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
# 6. UI wiring
# ------------------------------------------------------
if not SP100_TICKERS:
    st.warning(
        "Please edit `SP100_TICKERS` in `options_screener.py` to include your SP100 universe."
    )
else:
    run_btn = st.sidebar.button("Run SP100 GEX/VEX Screener")

    if run_btn:
        with st.spinner("Running GEX/VEX screener on nearest expirations..."):
            universe = tuple(sorted(set(SP100_TICKERS)))
            screener_df = run_gex_vex_screener(universe)

        if screener_df.empty:
            st.warning("No options data returned for the current universe.")
        else:
            st.subheader("Ticker-level GEX/VEX (Nearest Expiration)")

            # First output: colored table
            styled = styled_screener_table(screener_df)
            st.dataframe(styled, use_container_width=True)

            # Second output: VEX vs GEX scatter
            st.subheader("VEX vs GEX (bubble size = total contracts)")
            fig = vex_vs_gex_scatter(screener_df)
            st.plotly_chart(fig, use_container_width=True)
