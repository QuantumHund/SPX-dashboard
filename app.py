import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

st.set_page_config(layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard")

# Fetch SPX data
@st.cache_data(ttl=300)
def fetch_spx_data():
    try:
        df = yf.download("^GSPC", period="6mo", interval="1d", progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.dropna(how='all', axis=1)
        return df
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

spx = fetch_spx_data()

# Check and select price column
required_cols = ['Adj Close', 'Close']
price_col = None
for col in required_cols:
    if col in spx.columns:
        price_col = col
        break

if spx.empty or price_col is None:
    st.error("❌ SPX adatok nem érhetők el vagy hiányzik az árfolyam oszlop.")
    st.write("SPX oszlopok:", spx.columns.tolist())
    st.stop()

# Ensure price_col is 1D Series
if isinstance(spx[price_col], pd.DataFrame):
    spx[price_col] = spx[price_col].squeeze()

# Drawdown calculation
spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1

# Indicators calculations with proper 1D input
spx['RSI'] = ta.momentum.RSIIndicator(spx[price_col], window=14).rsi()
spx['SMA50'] = spx[price_col].rolling(window=50).mean()
spx['SMA200'] = spx[price_col].rolling(window=200).mean()
macd = ta.trend.MACD(spx[price_col])
spx['MACD'] = macd.macd()
spx['MACD_signal'] = macd.macd_signal()
bb = ta.volatility.BollingerBands(spx[price_col])
spx['BB_high'] = bb.bollinger_hband()
spx['BB_low'] = bb.bollinger_lband()
stoch = ta.momentum.StochasticOscillator(spx['High'], spx['Low'], spx[price_col], window=14, smooth_window=3)
spx['Stoch'] = stoch.stoch()
spx['Stoch_signal'] = stoch.stoch_signal()

# Buy/Sell Score (0-6)
spx['Buy_Score'] = (
    (spx['RSI'] < 30).astype(int) +
    (spx['Drawdown'] < -0.10).astype(int) +  # >10% drawdown
    (spx['MACD'] > spx['MACD_signal']).astype(int) +
    (spx['Stoch'] < 20).astype(int) +
    (spx[price_col] < spx['SMA50']).astype(int) +
    (spx[price_col] < spx['BB_low']).astype(int)
)

spx['Sell_Score'] = (
    (spx['RSI'] > 70).astype(int) +
    (spx['Drawdown'] > -0.01).astype(int) +
    (spx['MACD'] < spx['MACD_signal']).astype(int) +
    (spx['Stoch'] > 80).astype(int) +
    (spx[price_col] > spx['SMA50']).astype(int) +
    (spx[price_col] > spx['BB_high']).astype(int)
)

# Plots
st.subheader("📉 SPX Price Chart")
st.line_chart(spx[price_col])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
st.line_chart(spx['RSI'])

st.subheader("📊 Moving Averages (SMA50 & SMA200)")
st.line_chart(spx[['SMA50', 'SMA200']])

st.subheader("📈 MACD & Signal Line")
st.line_chart(spx[['MACD', 'MACD_signal']])

st.subheader("📈 Bollinger Bands")
st.line_chart(spx[[price_col, 'BB_high', 'BB_low']])

st.subheader("📈 Stochastic Oscillator & Signal")
st.line_chart(spx[['Stoch', 'Stoch_signal']])

st.subheader("🟢 Buy & 🔴 Sell Scores (0-6)")
import altair as alt

# Prepare scores with colored dots for points >=4
df_scores = spx.reset_index()
base = alt.Chart(df_scores).encode(x='Date:T')

buy_points = base.mark_circle(color='green', size=60).encode(
    y='Buy_Score:Q',
    tooltip=['Date:T', 'Buy_Score:Q']
).transform_filter(alt.datum.Buy_Score >= 4)

sell_points = base.mark_circle(color='red', size=60).encode(
    y='Sell_Score:Q',
    tooltip=['Date:T', 'Sell_Score:Q']
).transform_filter(alt.datum.Sell_Score >= 4)

line = base.mark_line().encode(
    y='Buy_Score:Q',
    color=alt.value('green')
) + base.mark_line().encode(
    y='Sell_Score:Q',
    color=alt.value('red')
)

st.altair_chart(line + buy_points + sell_points, use_container_width=True)

st.subheader("📊 Raw Data Table (last 30 rows)")
st.dataframe(spx.tail(30))
