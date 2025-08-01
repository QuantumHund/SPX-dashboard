import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import numpy as np
import altair as alt

st.set_page_config(layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard")

@st.cache_data(ttl=300)
def fetch_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    # Ha MultiIndex az oszlop, csak a legbelső szintet tartjuk meg
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    return df

spx = fetch_data("^GSPC")
vix = fetch_data("^VIX")

# Debug: oszlopok listázása
st.write("SPX oszlopok:", spx.columns.tolist())
st.write("VIX oszlopok:", vix.columns.tolist())

# Price column kiválasztása
if 'Adj Close' in spx.columns:
    price_col = 'Adj Close'
elif 'Close' in spx.columns:
    price_col = 'Close'
else:
    st.error("Nem található 'Close' vagy 'Adj Close' oszlop az SPX adatok között.")
    st.stop()

# SPX Drawdown
spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1

# RSI számítás
spx['RSI'] = ta.momentum.RSIIndicator(spx[price_col], window=14).rsi()

# Mozgóátlagok (SMA, EMA)
spx['SMA50'] = spx[price_col].rolling(window=50).mean()
spx['SMA200'] = spx[price_col].rolling(window=200).mean()
spx['EMA50'] = spx[price_col].ewm(span=50, adjust=False).mean()

# MACD indikátor
macd = ta.trend.MACD(spx[price_col])
spx['MACD'] = macd.macd()
spx['MACD_signal'] = macd.macd_signal()

# Bollinger Bands
bb = ta.volatility.BollingerBands(spx[price_col])
spx['BB_High'] = bb.bollinger_hband()
spx['BB_Low'] = bb.bollinger_lband()

# Stochastic Oscillator
stoch = ta.momentum.StochasticOscillator(spx['High'], spx['Low'], spx[price_col])
spx['Stoch'] =
