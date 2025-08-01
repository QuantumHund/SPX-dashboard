import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolume

st.set_page_config(layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard")

@st.cache_data(ttl=300)
def fetch_data(ticker):
    df = yf.download(ticker, period="6mo", interval="1d", progress=False)
    return df

spx = fetch_data("^GSPC")
vix = fetch_data("^VIX")

# Ellenőrzés: megvannak-e az alap adatok
if spx.empty or vix.empty:
    st.error("❌ Nem sikerült letölteni az SPX vagy VIX adatokat.")
    st.stop()

# Válaszd ki az árfolyam oszlopot
price_col = "Adj Close" if "Adj Close" in spx.columns else "Close"
if price_col not in spx.columns:
    st.error(f"Nem található '{price_col}' oszlop az SPX adatok között.")
    st.stop()

# Alap mutatók számítása SPX-re
spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1

spx['RSI'] = RSIIndicator(close=spx[price_col], window=14).rsi()

macd = MACD(close=spx[price_col])
spx['MACD'] = macd.macd()
spx['MACD_Signal'] = macd.macd_signal()

bb = BollingerBands(close=spx[price_col])
spx['BB_High'] = bb.bollinger_hband()
spx['BB_Low'] = bb.bollinger_lband()

stoch = StochasticOscillator(high=spx['High'], low=spx['Low'], close=spx[price_col])
spx['Stoch'] = stoch.stoch()

spx['OBV'] = OnBalanceVolume(close=spx[price_col], volume=spx['Volume']).on_balance_volume()

# VIX indikátor (egyszerű jelzés: VIX > 25 és csökkenő)
vix['VIX_Falling'] = vix['Close'].diff() < 0

# Score számítása (max 6 pont)
spx['Score'] = 0

# 1 pont: Drawdown nagyobb mint -10%
spx.loc[spx['Drawdown'] < -0.10, 'Score'] += 1

# 1 pont: RSI oversold (RSI < 30)
spx.loc[spx['RSI'] < 30, 'Score'] += 1

# 1 pont: VIX > 25 és csökkenő
if not vix.empty:
    vix_latest = vix.iloc[-1]
    if vix_latest['Close'] > 25 and vix_latest['VIX_Falling']:
        spx['Score'] += 1  # VIX indikátor pontot az egész időszakra adom, lehet finomítani

# 1 pont: MACD bullish crossover (MACD vonal a jelzővonal felett)
spx.loc[spx['MACD'] > spx['MACD_Signal'], 'Score'] += 1

# 1 pont: Stochastic oversold jelzés (Stoch < 20)
spx.loc[spx['Stoch'] < 20, 'Score'] += 1

# 1 pont: Árfolyam a Bollinger alsó szalag közelében
spx.loc[spx[price_col] < spx['BB_Low'], 'Score'] += 1

# Végső értékelés max 6 pont

# Grafikonok megjelenítése
st.subheader("📉 SPX Price Chart")
st.line_chart(spx[price_col])

st.subheader("📈 RSI")
st.line_chart(spx['RSI'])

st.subheader("📈 MACD és Signal Line")
st.line_chart(spx[['MACD', 'MACD_Signal']])

st.subheader("📈 Stochastic Oscillator")
st.line_chart(spx['Stoch'])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 Bollinger Bands")
st.line_chart(pd.DataFrame({
    'Price': spx[price_col],
    'BB High': spx['BB_High'],
    'BB Low': spx['BB_Low']
}))

st.subheader("📊 Market Score (0-6)")
st.line_chart(spx['Score'])

st.subheader("📊 Raw Data Table (last 30 rows)")
st.dataframe(spx.tail(30))
