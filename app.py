import streamlit as st
import pandas as pd
import yfinance as yf
import ta
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.momentum import StochasticOscillator
from ta.volume import OnBalanceVolume
from streamlit_autorefresh import st_autorefresh
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard")

# Automatikus frissítés 5 percenként
st_autorefresh(interval=300000, key="data_refresh")

@st.cache_data(ttl=300)
def fetch_data(ticker):
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False)
        if df.empty:
            return pd.DataFrame()
        # MultiIndex oszlopok kezelése
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)
        return df.dropna()
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

# SPX és VIX adatok
spx = fetch_data("^GSPC")
vix = fetch_data("^VIX")

price_col = 'Adj Close' if 'Adj Close' in spx.columns else 'Close'

if spx.empty or price_col not in spx.columns:
    st.error(f"❌ Failed to fetch SPX data or '{price_col}' column is missing.")
    st.stop()

if vix.empty or 'Close' not in vix.columns:
    st.error("❌ Failed to fetch VIX data or 'Close' column is missing.")
    st.stop()

# Indikátorok számítása
spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1
spx['RSI'] = RSIIndicator(close=spx[price_col], window=14).rsi()

macd_indicator = MACD(close=spx[price_col])
spx['MACD'] = macd_indicator.macd()
spx['MACD_signal'] = macd_indicator.macd_signal()

bb_indicator = BollingerBands(close=spx[price_col])
spx['BB_high'] = bb_indicator.bollinger_hband()
spx['BB_low'] = bb_indicator.bollinger_lband()

stoch = StochasticOscillator(high=spx['High'], low=spx['Low'], close=spx[price_col])
spx['Stoch'] = stoch.stoch()
spx['Stoch_signal'] = stoch.stoch_signal()

spx['OBV'] = OnBalanceVolume(close=spx[price_col], volume=spx['Volume']).on_balance_volume()

# VIX trend
vix['VIX_diff'] = vix['Close'].diff()

# Score számítás (0-6 max)
spx['Buy_Score'] = (
    (spx['RSI'] < 30).astype(int) + 
    (spx['Drawdown'] < -0.10).astype(int) +
    (spx['MACD'] > spx['MACD_signal']).astype(int) + 
    (spx['Stoch'] < 20).astype(int) +
    (spx['OBV'].diff() > 0).astype(int) + 
    ((vix['Close'] > 25) & (vix['VIX_diff'] < 0)).reindex(spx.index, method='ffill').fillna(False).astype(int)
)

spx['Sell_Score'] = (
    (spx['RSI'] > 70).astype(int) + 
    (spx['Drawdown'] > -0.01).astype(int) +
    (spx['MACD'] < spx['MACD_signal']).astype(int) + 
    (spx['Stoch'] > 80).astype(int) +
    (spx['OBV'].diff() < 0).astype(int) + 
    ((vix['Close'] < 15) & (vix['VIX_diff'] > 0)).reindex(spx.index, method='ffill').fillna(False).astype(int)
)

# Grafikonok
st.subheader("📉 SPX Price Chart")
st.line_chart(spx[price_col])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
st.line_chart(spx['RSI'])

st.subheader("📈 MACD and Signal")
st.line_chart(spx[['MACD', 'MACD_signal']])

st.subheader("📈 Bollinger Bands")
st.line_chart(spx[[price_col, 'BB_high', 'BB_low']])

st.subheader("📈 Stochastic Oscillator and Signal")
st.line_chart(spx[['Stoch', 'Stoch_signal']])

st.subheader("📊 On-Balance Volume")
st.line_chart(spx['OBV'])

st.subheader("🟢 Buy & 🔴 Sell Scores (0-6)")
import altair as alt

df_scores = spx[['Buy_Score', 'Sell_Score']].reset_index()
df_scores_melt = df_scores.melt('Date', var_name='ScoreType', value_name='Score')

base = alt.Chart(df_scores_melt).encode(
    x='Date:T',
    y='Score:Q',
    color=alt.condition(
        alt.datum.ScoreType == 'Buy_Score',
        alt.value("green"),
        alt.value("red")
    )
)

points = base.mark_circle(size=60).encode(
    size='Score:Q',
    tooltip=['Date:T', 'ScoreType:N', 'Score:Q']
)

lines = base.mark_line()

st.altair_chart(lines + points, use_container_width=True)

st.subheader("📊 Raw Data Table (last 30 rows)")
st.dataframe(spx.tail(30))
