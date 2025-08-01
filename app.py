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
spx['Stoch'] = stoch.stoch()

# VIX price, és mozgóátlag 10 napos
if not vix.empty and 'Close' in vix.columns:
    vix['SMA10'] = vix['Close'].rolling(window=10).mean()
else:
    vix['Close'] = np.nan
    vix['SMA10'] = np.nan

# On-Balance Volume (OBV)
if 'Volume' in spx.columns:
    spx['OBV'] = ta.volume.OnBalanceVolumeIndicator(spx[price_col], spx['Volume']).on_balance_volume()
else:
    spx['OBV'] = np.nan

# Score számítás max 6 pont:
spx['Buy_Score'] = 0
spx['Sell_Score'] = 0

# Buy pontok hozzáadása
spx.loc[spx['Drawdown'] <= -0.1, 'Buy_Score'] += 1   # nagyobb 10% esés
spx.loc[spx['RSI'] < 30, 'Buy_Score'] += 1           # oversold RSI
spx.loc[(vix['Close'] > 25) & (vix['Close'] < vix['SMA10']), 'Buy_Score'] += 1  # VIX > 25 és csökkenő
spx.loc[(spx['MACD'] > spx['MACD_signal']), 'Buy_Score'] += 1  # MACD bullish crossover
spx.loc[(spx['SMA50'] > spx['SMA200']), 'Buy_Score'] += 1    # golden cross
spx.loc[spx['Stoch'] < 20, 'Buy_Score'] += 1                 # Stochastic oversold

# Sell pontok hozzáadása
spx.loc[spx['Drawdown'] >= -0.01, 'Sell_Score'] += 1        # minimális esés vagy emelkedés
spx.loc[spx['RSI'] > 70, 'Sell_Score'] += 1                 # overbought RSI
spx.loc[(vix['Close'] > 25) & (vix['Close'] > vix['SMA10']), 'Sell_Score'] += 1  # VIX > 25 és emelkedő
spx.loc[(spx['MACD'] < spx['MACD_signal']), 'Sell_Score'] += 1  # MACD bearish crossover
spx.loc[(spx['SMA50'] < spx['SMA200']), 'Sell_Score'] += 1    # death cross
spx.loc[spx['Stoch'] > 80, 'Sell_Score'] += 1                 # Stochastic overbought

# Grafikonok - ár + RSI
st.subheader("📉 SPX Price Chart")
st.line_chart(spx[price_col])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
st.line_chart(spx['RSI'])

st.subheader("🟢 Buy & 🔴 Sell Scores (0-6)")
buy_chart = alt.Chart(spx.reset_index()).mark_line(color='green').encode(
    x='Date:T',
    y='Buy_Score:Q',
    tooltip=['Date:T', 'Buy_Score']
)
sell_chart = alt.Chart(spx.reset_index()).mark_line(color='red').encode(
    x='Date:T',
    y='Sell_Score:Q',
    tooltip=['Date:T', 'Sell_Score']
)

combined = alt.layer(buy_chart, sell_chart).properties(width=800, height=300)
st.altair_chart(combined, use_container_width=True)

st.subheader("📊 Raw Data Table (last 30 rows)")
st.dataframe(spx.tail(30))
