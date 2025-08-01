import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

st.set_page_config(page_title="📊 SPX Multi-Indicator Market Score Dashboard", layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard")

@st.cache_data(ttl=3600)
def download_data(ticker, period="3mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df

# Adatok letöltése
spx = download_data("^GSPC")
vix = download_data("^VIX")

# Ellenőrzés, hogy letöltődtek-e az adatok és van 'Close' oszlop
if spx.empty or 'Close' not in spx.columns:
    st.error("❌ SPX adatok nem érhetők el vagy hiányzik az árfolyam oszlop.")
    st.stop()

if vix.empty or 'Close' not in vix.columns:
    st.error("❌ VIX adatok nem érhetők el vagy hiányzik az árfolyam oszlop.")
    st.stop()

# Csak 1D Series az árfolyamokhoz (megakadályozzuk az ndarray (n,1) problémát)
spx_close = spx['Close']
vix_close = vix['Close']

# Indikátorok számítása
spx['RSI'] = ta.momentum.RSIIndicator(spx_close, window=14).rsi()
spx['SMA50'] = ta.trend.SMAIndicator(spx_close, window=50).sma_indicator()
spx['SMA200'] = ta.trend.SMAIndicator(spx_close, window=200).sma_indicator()

macd = ta.trend.MACD(spx_close)
spx['MACD'] = macd.macd()
spx['MACD_signal'] = macd.macd_signal()

bb = ta.volatility.BollingerBands(spx_close)
spx['BB_high'] = bb.bollinger_hband()
spx['BB_low'] = bb.bollinger_lband()

stoch = ta.momentum.StochasticOscillator(spx['High'], spx['Low'], spx_close, window=14, smooth_window=3)
spx['Stoch'] = stoch.stoch()
spx['Stoch_signal'] = stoch.stoch_signal()

obv = ta.volume.OnBalanceVolumeIndicator(spx_close, spx['Volume'])
spx['OBV'] = obv.on_balance_volume()

# SPX drawdown számítás (max 10%-os visszaesés jelzéshez)
spx['Drawdown'] = (spx_close / spx_close.cummax()) - 1

# RSI oversold (30 alatti), VIX > 25 és csökkenő, MACD bullish crossover
# MACD crossover megállapítása (MACD vonal keresztezi a jelzővonalat felfelé)
macd_bullish_cross = (spx['MACD'] > spx['MACD_signal']) & (spx['MACD'].shift(1) <= spx['MACD_signal'].shift(1))
macd_bearish_cross = (spx['MACD'] < spx['MACD_signal']) & (spx['MACD'].shift(1) >= spx['MACD_signal'].shift(1))

# Bearish sentiment spike - például VIX emelkedés nagyobb, mint egy küszöb
vix['VIX_change'] = vix_close.pct_change()
bearish_sentiment_spike = vix['VIX_change'] > 0.05  # 5% emelkedés VIX-ben napi szinten

# Score összesítés
def calc_score(row):
    score = 0
    # Drawdown > -10% (azaz 10%-os vagy nagyobb esés)
    if row['Drawdown'] <= -0.10:
        score += 1
    # RSI < 30 oversold = jó vételi jel, tehát -1 pont (negatív)
    if row['RSI'] < 30:
        score -= 1
    # VIX > 25 és csökken
    if vix_close.loc[row.name] > 25:
        # Nézzük, csökken-e a VIX
        idx = vix_close.index.get_loc(row.name)
        if idx > 0 and vix_close.iloc[idx] < vix_close.iloc[idx-1]:
            score += 1
    # MACD bullish crossover +1 pont
    if macd_bullish_cross.loc[row.name]:
        score += 1
    # Bearish sentiment spike +1 pont
    if row.name in bearish_sentiment_spike.index and bearish_sentiment_spike.loc[row.name]:
        score += 1
    return score

spx['Score'] = spx.apply(calc_score, axis=1)
max_score = 5

# Vizualizáció
st.subheader("SPX és indikátorok")

# Score alapján pontok színezése
buy_points = spx[spx['Score'] >= 3]
sell_points = spx[spx['Score'] <= 0]

import matplotlib.pyplot as plt

fig, ax1 = plt.subplots(figsize=(12,6))

ax1.plot(spx.index, spx_close, label="SPX Close", color="black")
ax1.set_ylabel("SPX árfolyam", color="black")

ax2 = ax1.twinx()
ax2.plot(spx.index, spx['Score'], label="Score", color="green")
ax2.scatter(buy_points.index, buy_points['Score'], color='green', s=50, label="Buy signal", marker='o')
ax2.scatter(sell_points.index, sell_points['Score'], color='red', s=50, label='Sell signal', marker='o')
ax2.set_ylabel("Market Score", color="green")

ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

st.pyplot(fig)

st.markdown("### SPX adatok (utolsó 5 nap)")
st.dataframe(spx.tail())

st.markdown("### VIX adatok (utolsó 5 nap)")
st.dataframe(vix.tail())
