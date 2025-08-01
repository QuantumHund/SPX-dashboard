import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.set_page_config(layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard")

@st.cache_data(ttl=300)
def fetch_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    # Ha multiindex az oszlop, akkor level-eljük
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    return df

spx = fetch_data("^GSPC")
vix = fetch_data("^VIX")

if spx.empty or vix.empty:
    st.error("Nem sikerült letölteni az adatokat.")
    st.stop()

price_col = "Adj Close" if "Adj Close" in spx.columns else "Close"

# Drawdown számítás
spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1

# RSI
spx['RSI'] = ta.momentum.RSIIndicator(spx[price_col], window=14).rsi()

# Moving Averages
spx['SMA50'] = spx[price_col].rolling(window=50).mean()
spx['SMA200'] = spx[price_col].rolling(window=200).mean()
spx['EMA50'] = spx[price_col].ewm(span=50, adjust=False).mean()

# MACD
macd = ta.trend.MACD(spx[price_col])
spx['MACD'] = macd.macd()
spx['MACD_signal'] = macd.macd_signal()
spx['MACD_diff'] = macd.macd_diff()

# Bollinger Bands
bollinger = ta.volatility.BollingerBands(spx[price_col])
spx['BB_high'] = bollinger.bollinger_hband()
spx['BB_low'] = bollinger.bollinger_lband()

# Stochastic Oscillator
stoch = ta.momentum.StochasticOscillator(spx['High'], spx['Low'], spx[price_col])
spx['Stoch'] = stoch.stoch()

# On-Balance Volume (OBV)
spx['OBV'] = ta.volume.OnBalanceVolumeIndicator(spx[price_col], spx['Volume']).on_balance_volume()

# VIX záróár és mozgása
vix_close = vix['Close']
vix_diff = vix_close.diff()

# Score alapjai (0-6 pont max)
score = pd.Series(0, index=spx.index)

# 1. Drawdown >10% figyelmeztetés
score += (spx['Drawdown'] < -0.10).astype(int)  # +1

# 2. RSI oversold (RSI < 30)
score += (spx['RSI'] < 30).astype(int)  # +1

# 3. VIX > 25 és csökken (volatilitás enyhül)
vix_flag = ((vix_close > 25) & (vix_diff < 0)).reindex(spx.index, method='ffill').fillna(False)
score += vix_flag.astype(int)  # +1

# 4. MACD bullish crossover
macd_bull_cross = (spx['MACD'] > spx['MACD_signal']) & (spx['MACD'].shift(1) <= spx['MACD_signal'].shift(1))
score += macd_bull_cross.astype(int)  # +1

# 5. Bollinger Bands squeeze (szűkülés) - alacsony volatilitás -> potenciális mozgás
bb_width = spx['BB_high'] - spx['BB_low']
bb_squeeze = bb_width < bb_width.rolling(window=20).mean()
score += bb_squeeze.astype(int)  # +1

# 6. SMA50 / SMA200 Golden Cross (bullish trend)
golden_cross = (spx['SMA50'] > spx['SMA200']) & (spx['SMA50'].shift(1) <= spx['SMA200'].shift(1))
score += golden_cross.astype(int)  # +1

spx['Market_Score'] = score

# --- Vizualizációk ---

import matplotlib.pyplot as plt

st.subheader("📉 SPX Price Chart")
st.line_chart(spx[price_col])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
st.line_chart(spx['RSI'])

st.subheader("🟢 Market Score (0-6)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(spx.index, spx['Market_Score'], label='Market Score', color='blue')

# Jelölések a magas pontszámokra (>=4)
high_score = spx['Market_Score'] >= 4
ax.scatter(spx.index[high_score], spx['Market_Score'][high_score], color='green', s=50, label='High Score (≥4)')

ax.set_ylim(0, 6.5)
ax.set_ylabel('Score')
ax.legend()
st.pyplot(fig)

st.subheader("📈 MACD and Signal")
st.line_chart(spx[['MACD', 'MACD_signal']])

st.subheader("📊 Raw Data (last 30 rows)")
st.dataframe(spx.tail(30))
