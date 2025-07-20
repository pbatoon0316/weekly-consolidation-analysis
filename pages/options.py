import streamlit as st
import yfinance as yf
import pandas as pd
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from py_vollib.black.greeks.analytical import gamma as bsm_gamma


st.set_page_config(page_title="Options GEX Dashboard", layout="wide")

# Sidebar Inputs
with st.sidebar:
    st.header("⚙️ Options Data Settings")
    symbol_input = st.text_input("Enter a symbol (e.g., ^SPX, SPY, AAPL):", value="")

# Sanitize and validate ticker input
symbol = symbol_input.strip().upper()
ticker = None
selected_expirations = []

# If no ticker, warn in the main app window
if not symbol:
    st.warning("Please input a stock ticker.")
else:
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        print(f'Loading ${symbol} data')

        # Filter: 1-year worth of expirations
        one_year_expirations = [
            d for d in expirations if pd.to_datetime(d) <= pd.Timestamp.today() + pd.Timedelta(days=365)
        ]
        default_selection = one_year_expirations[:3] if len(one_year_expirations) >= 3 else one_year_expirations

        # Step 2: Expiration Dates Multiselect (in sidebar)
        with st.sidebar:
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
            chain = _ticker.option_chain(exp)
            for df, opt_type in [(chain.calls, "call"), (chain.puts, "put")]:
                if not df.empty:
                    df = df.copy()
                    df["type"] = opt_type
                    df["expiration"] = exp
                    all_options.append(df)
        except Exception as e:
            st.warning(f"Could not load options for {exp}: {e}")
        time.sleep(1)

    if not all_options:
        return pd.DataFrame()

    options_df = pd.concat(all_options, ignore_index=True)
    options_df.rename(columns={"openInterest": "oi"}, inplace=True)

    # Filter for required columns
    cols = ["expiration", "type", "strike", "oi", "impliedVolatility"]
    options_df = options_df[[col for col in cols if col in options_df.columns]].copy()

    return options_df[options_df["oi"] > 0].sort_values("strike")


    
# calculate gamma
sr1_close = yf.Ticker('SR1=F').info['previousClose'] 
r = (100 - sr1_close) / 100

def compute_gamma(row, S, r):
    try:
        K = float(row['strike'])
        sigma = float(row['impliedVolatility'])

        if sigma <= 0 or pd.isna(sigma):
            return 0.0

        T = max((pd.to_datetime(row['expiration']) - pd.Timestamp.now()).total_seconds() / (365 * 24 * 60 * 60), 1e-6)
        option_type = 'c' if row['type'] == 'call' else 'p'

        return bsm_gamma(option_type, S, K, T, r, sigma)

    except Exception:
        return 0.0


# show data
if symbol and selected_expirations:
    options_df = get_options_chain(_ticker=ticker, expirations=selected_expirations)
    
    # Add underlying price for gamma calc
    underlying_price = ticker.history(period="1d")['Close'].iloc[-1]
    options_df["gamma"] = options_df.apply(lambda row: compute_gamma(row, underlying_price, r), axis=1)


    if options_df.empty:
        st.sidebar.warning("No options data found for selected expirations.")
    else:
        st.sidebar.badge(f"Loaded {len(options_df)} contracts across {len(selected_expirations)} expiration(s).", icon=":material/check:", color="green")
        #st.dataframe(options_df.head())


# Only run if gamma column exists
if not options_df.empty and "gamma" in options_df.columns:
    
    # Step 1: Calculate GEX
    options_df["gex"] = (options_df["gamma"] * options_df["oi"] * 100 * underlying_price) / 1000000
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
    period_selection = st.sidebar.selectbox(label='Time Period', options=['5d','2wk','1mo','2mo','3mo','6mo','1yr'], index=3)
    price_data = yf.download(symbol, period=period_selection, interval="1d")
    price_data.dropna(inplace=True)
    current_price = underlying_price


    if isinstance(price_data.columns, pd.MultiIndex):
        price_data = price_data.xs(symbol, axis=1, level=1)

    st.sidebar.metric('Net Gamma Exposure', value=f'${round(net_gex,2)} M')

    
    # Step 3: Create composite Plotly figure
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.6, 0.2, 0.2],
        shared_yaxes=False,
        horizontal_spacing=0.03,
        subplot_titles=(f"{symbol} OHLC", "Gamma Exposure ($M)", "Open Interest"),
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
            x=1*agg["gex"],
            y=agg["strike"],
            orientation="h",
            marker_color="orange",
            name="GEX",
            hovertemplate="Strike: %{y}<br>GEX: $%{customdata} M",
            customdata=agg["gex"].abs().round(1)
        ),
        row=1, col=2
    )
           
    
    # Plot 3: Open Interest
    fig.add_trace(
        go.Bar(
            x=1*agg["oi"],
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

    
    fig.update_xaxes(title_text="Gamma Exposure ($M)", row=1, col=2)
    fig.update_xaxes(title_text="Open Interest", row=1, col=3)
    
    # Horizontal lines for current price and zero gamma strike
    for col in [2, 3]:
        # --- Current Price Line (black dashed) ---
        fig.add_shape(
            type="line",
            x0=0, x1=1,
            y0=current_price, y1=current_price,
            xref=f"x{col} domain",
            yref=f"y{col}",
            line=dict(color="black", width=2, dash="dash"),
        )
    
        fig.add_annotation(
            x=1.0,
            y=current_price,
            xref=f"x{col} domain",
            yref=f"y{col}",
            text=f"${current_price:.2f}",
            showarrow=False,
            font=dict(color="black", size=16),
            xanchor="left",
            yanchor="bottom"
        )


    # 🚀 Display chart
    st.plotly_chart(fig, use_container_width=True)
    
with st.expander('Raw Data'):
    st.dataframe(options_df)
