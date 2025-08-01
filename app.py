import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard")

# Auto-refresh every 5 minutes
st_autorefresh(interval=300000, key="data_refresh")

@st.cache_data(ttl=300)
def fetch_data(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Data fetch error for {ticker}: {e}")
        return pd.DataFrame()

# Fetch SPX and VIX data
spx = fetch_data("^GSPC", period="6mo", interval="1d")
vix = fetch_data("^VIX", period="6mo", interval="1d")

# Price column
price_col = 'Adj Close' if 'Adj Close' in spx.columns else 'Close'

if spx.empty or price_col not in spx.columns:
    st.error(f"❌ Failed to fetch SPX data or '{price_col}' column is missing.")
    st.write("Debug - Columns received:", list(spx.columns))
    st.stop()

if vix.empty or 'Close' not in vix.columns:
    st.error("❌ Failed to fetch VIX data or 'Close' column is missing.")
    st.write("Debug - Columns received:", list(vix.columns))
    st.stop()

# Calculate Drawdown
spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1

# RSI
spx['RSI'] = ta.momentum.RSIIndicator(spx[price_col], window=14).rsi()

# Moving Averages
spx['SMA50'] = spx[price_col].rolling(window=50).mean()
spx['SMA200'] = spx[price_col].rolling(window=200).mean()
spx['EMA20'] = spx[price_col].ewm(span=20, adjust=False).mean()

# MACD
macd = ta.trend.MACD(spx[price_col])
spx['MACD'] = macd.macd()
spx['MACD_signal'] = macd.macd_signal()

# MACD crossover bullish signal
spx['MACD_Bull'] = ((spx['MACD'] > spx['MACD_signal']) & (spx['MACD'].shift(1) <= spx['MACD_signal'].shift(1))).astype(int)

# Bollinger Bands
bb = ta.volatility.BollingerBands(spx[price_col])
spx['BB_high'] = bb.bollinger_hband()
spx['BB_low'] = bb.bollinger_lband()

# Stochastic Oscillator
stoch = ta.momentum.StochasticOscillator(spx['High'], spx['Low'], spx[price_col], window=14, smooth_window=3)
spx['Stoch'] = stoch.stoch()

# Volume (OBV manual calculation)
def calculate_obv(close, volume):
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * volume).cumsum()
    return obv

spx['OBV'] = calculate_obv(spx[price_col], spx['Volume'])

# VIX conditions
vix['VIX_falling'] = vix['Close'].diff() < 0
vix_latest = vix.iloc[-1]

# Bearish sentiment spike (placeholder, random for demo)
import random
bearish_sentiment_spike = random.choice([0,1])  # helyettesíthető API-val vagy valós adattal

# Macro catalyst (placeholder boolean)
macro_catalyst = random.choice([0,1])  # helyettesíthető valós adatból

# Score calculation max 6 pont
spx['Score'] = 0

# Conditions to add points
spx['Score'] += (spx['Drawdown'] < -0.10).astype(int)  # drawdown >10%
spx['Score'] += (spx['RSI'] < 30).astype(int)          # RSI oversold
spx['Score'] += ((vix_latest['Close'] > 25) & vix_latest['VIX_falling']) * 1
spx['Score'] += spx['MACD_Bull']
spx['Score'] += bearish_sentiment_spike
spx['Score'] += macro_catalyst

# Visualization
st.subheader("📉 SPX Price with SMA50, SMA200, EMA20")
fig = go.Figure()
fig.add_trace(go.Scatter(x=spx.index, y=spx[price_col], mode='lines', name='Price'))
fig.add_trace(go.Scatter(x=spx.index, y=spx['SMA50'], mode='lines', name='SMA50'))
fig.add_trace(go.Scatter(x=spx.index, y=spx['SMA200'], mode='lines', name='SMA200'))
fig.add_trace(go.Scatter(x=spx.index, y=spx['EMA20'], mode='lines', name='EMA20'))
st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
st.line_chart(spx['RSI'])

st.subheader("📈 MACD and Signal")
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=spx.index, y=spx['MACD'], mode='lines', name='MACD'))
fig_macd.add_trace(go.Scatter(x=spx.index, y=spx['MACD_signal'], mode='lines', name='Signal'))
st.plotly_chart(fig_macd, use_container_width=True)

st.subheader("📈 Stochastic Oscillator")
st.line_chart(spx['Stoch'])

st.subheader("📊 On-Balance Volume (OBV)")
st.line_chart(spx['OBV'])

st.subheader("🟢 Market Score (0-6)")
st.line_chart(spx['Score'])

st.subheader("📋 Latest Scores & Market Conditions")
st.write({
    "Drawdown > 10%": spx['Drawdown'].iloc[-1] < -0.10,
    "RSI < 30 (oversold)": spx['RSI'].iloc[-1] < 30,
    "VIX > 25 & falling": (vix_latest['Close'] > 25) & vix_latest['VIX_falling'],
    "MACD bullish crossover": bool(spx['MACD_Bull'].iloc[-1]),
    "Bearish sentiment spike": bool(bearish_sentiment_spike),
    "Macro catalyst": bool(macro_catalyst),
})

st.subheader("📊 Raw SPX Data (last 30 rows)")
st.dataframe(spx.tail(30))

