import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title="Options GEX Dashboard", layout="wide")

# Sidebar Inputs
with st.sidebar:
    st.header("⚙️ Options Data Settings")

    # Step 1: Ticker Input
    symbol = st.text_input("Enter a symbol (e.g., ^SPX, SPY, AAPL):", "^SPX").strip().upper()

    selected_expirations = []
    if symbol:
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            # Filter: 1-year worth of expirations
            one_year_expirations = [d for d in expirations if pd.to_datetime(d) <= pd.Timestamp.today() + pd.Timedelta(days=365)]
            default_selection = one_year_expirations[:3] if len(one_year_expirations) >= 3 else one_year_expirations

            # Step 2: Expiration Dates Multiselect
            selected_expirations = st.multiselect(
                "Select expiration dates:",
                options=expirations,
                default=default_selection
            )

        except Exception as e:
            st.error(f"Could not fetch data for {symbol}. Please check the symbol. Error: {e}")


# download and process data
@st.cache_data(ttl=3600)
def get_options_chain(_ticker, expirations):
    all_options = []

    for exp in expirations:
        try:
            opt_chain = _ticker.option_chain(exp)
            calls = opt_chain.calls.copy()
            puts = opt_chain.puts.copy()

            calls["type"] = "call"
            puts["type"] = "put"
            calls["expiration"] = exp
            puts["expiration"] = exp

            all_options.append(calls)
            all_options.append(puts)

        except Exception as e:
            st.warning(f"Could not load options for {exp}: {e}")

    if all_options:
        options_df = pd.concat(all_options, ignore_index=True)
        options_df = options_df[["expiration", "type", "strike", "openInterest", "impliedVolatility"]].dropna()
        options_df.rename(columns={"openInterest": "oi"}, inplace=True)
        options_df = options_df[options_df["oi"] > 0]
        return options_df.sort_values(by="strike")
    else:
        return pd.DataFrame()

    
# calculate gamma
def compute_gamma(row, S):
    K = row['strike']
    T = (pd.to_datetime(row['expiration']) - pd.Timestamp.today()).days / 365
    IV = row['impliedVolatility']
    sigma = IV
    q = 0.0
    r = 0.0

    if T <= 0 or sigma <= 0 or S <= 0:
        return np.nan

    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    except:
        return np.nan


# show data
if symbol and selected_expirations:
    options_df = get_options_chain(_ticker=ticker, expirations=selected_expirations)
    
    # Add underlying price for gamma calc
    underlying_price = ticker.history(period="1d")['Close'].iloc[-1]
    options_df["gamma"] = options_df.apply(lambda row: compute_gamma(row, underlying_price), axis=1)


    if options_df.empty:
        st.sidebar.warning("No options data found for selected expirations.")
    else:
        st.sidebar.badge(f"Loaded {len(options_df)} contracts across {len(selected_expirations)} expiration(s).", icon=":material/check:", color="green")
        #st.dataframe(options_df.head())


# Only run if gamma column exists
if not options_df.empty and "gamma" in options_df.columns:
    
    # Step 1: Calculate GEX
    options_df["gex"] = options_df["gamma"] * options_df["oi"] * 100
    agg = options_df.groupby("strike")[["oi", "gex"]].sum().reset_index()
    agg = agg.sort_values("strike", ascending=True)
    call_gex = options_df.loc[options_df["type"] == "call", "gex"].sum()
    put_gex  = options_df.loc[options_df["type"] == "put", "gex"].sum()
    net_gex  = call_gex - put_gex
    
    # Calculate cumulative GEX by strike to find the "zero gamma" flip point
    cumulative = agg.sort_values("strike").copy()
    cumulative["cum_gex"] = cumulative["gex"].cumsum()
    
    # Find the strike where GEX crosses zero (closest to zero)
    zero_cross_idx = cumulative["cum_gex"].abs().idxmin()
    zero_gamma_strike = cumulative.loc[zero_cross_idx, "strike"]




    # Step 2: Download price data & flatten multilevel columns
    period_selection = st.sidebar.selectbox(label='Time Period', options=['5d','2wk','1mo','2mo','3mo','6mo','1yr'], index=4)
    price_data = yf.download(symbol, period=period_selection, interval="1d")
    price_data.dropna(inplace=True)
    current_price = float(price_data["Close"].iloc[-1])


    if isinstance(price_data.columns, pd.MultiIndex):
        price_data = price_data.xs(symbol, axis=1, level=1)

    st.sidebar.metric('Neg Gamma Exposure', value=f'${round(net_gex,2)}')

    
    # Step 3: Create composite Plotly figure
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.6, 0.2, 0.2],
        shared_yaxes=False,
        horizontal_spacing=0.03,
        subplot_titles=(f"{symbol} OHLC", "Gamma Exposure", "Open Interest"),
        specs=[[{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]]
    )
    
    # Plot 1: OHLC Candlestick
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data["Open"],
            high=price_data["High"],
            low=price_data["Low"],
            close=price_data["Close"],
            name="OHLC"
        ),
        row=1, col=1
    )
    
    # Plot 2: Gamma Exposure
    fig.add_trace(
        go.Bar(
            x=-1*agg["gex"],
            y=agg["strike"],
            orientation="h",
            marker_color="orange",
            name="GEX",
            hovertemplate="Strike: %{y}<br>GEX: %{customdata}",
            customdata=agg["gex"].abs().round(1)
        ),
        row=1, col=2
    )
    
    fig.add_trace(go.Scatter(
        x=[0],
        y=[zero_gamma_strike],
        mode="markers+text",
        marker=dict(symbol="triangle-left", color="black", size=12),
        text=["Zero Gamma"],
        textposition="middle right",
        showlegend=False
    ), row=1, col=2)

    
    
    # Plot 3: Open Interest
    fig.add_trace(
        go.Bar(
            x=-1*agg["oi"],
            y=agg["strike"],
            orientation="h",
            marker_color="blue",
            name="OI",
            hovertemplate="Strike: %{y}<br>OI: %{customdata}",
            customdata=agg["oi"].abs().round(1)
        ),
        row=1, col=3
    )
    
    # Determine common y-range from stock price
    price_min = price_data["Low"].min()
    price_max = price_data["High"].max()
    padding = 0.1 * (price_max - price_min)
    y_min = price_min - 2*padding
    y_max = price_max + 5*padding

    
    # Update layout and sync all y-axes
    fig.update_layout(
        height=800,
    #    width=1200,
        showlegend=False,
    #    title_text=f"Options Dashboard for {symbol}",
        xaxis_rangeslider_visible=False,
    )
    


    # Candlestick Y-axis (force to match strike range)
    fig.update_yaxes(title_text="Price / Strike", range=[y_min, y_max], row=1, col=1)
    
    # GEX and OI use the same y range and strike values
    fig.update_yaxes(title_text="Strike", range=[y_min, y_max], side="right", row=1, col=2)
    fig.update_yaxes(title_text="Strike", range=[y_min, y_max], side="right", row=1, col=3)

    
    fig.update_xaxes(title_text="Gamma Exposure", row=1, col=2)
    fig.update_xaxes(title_text="Open Interest", row=1, col=3)
    
    # Horizontal dashed line for latest price, on GEX (col 2) and OI (col 3)
    for col in [2, 3]:
        fig.add_shape(
            type="line",
            x0=0, x1=1,  # full subplot width
            y0=current_price, y1=current_price,
            xref=f"x{col} domain",
            yref=f"y{col}",
            line=dict(color="black", width=1.5, dash="dash"),
        )
    
        fig.add_annotation(
            x=1.0,
            y=current_price,
            xref=f"x{col} domain",
            yref=f"y{col}",
            text=f"${current_price:.2f}",
            showarrow=False,
            font=dict(color="black", size=10),
            xanchor="left",
            yanchor="bottom"
        )

    # 🚀 Display chart
    st.plotly_chart(fig, use_container_width=True)

