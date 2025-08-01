import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import talib

st.set_page_config(layout="wide")
st.title("📉 SPX Correction Opportunity Dashboard - Enhanced MVP")

@st.cache_data(ttl=300)
def fetch_data(ticker, period="1y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    return df

# --- Fetch data ---
spx = fetch_data("^GSPC", period="1y")
vix = fetch_data("^VIX", period="1y")

if spx.empty or vix.empty:
    st.error("Data fetch error.")
    st.stop()

price_col = 'Adj Close' if 'Adj Close' in spx.columns else 'Close'

# --- Calculate Indicators ---

# Drawdown
spx['Peak'] = spx[price_col].cummax()
spx['Drawdown'] = (spx[price_col] - spx['Peak']) / spx['Peak']

# RSI
spx['RSI'] = talib.RSI(spx[price_col], timeperiod=14)

# MACD
macd, macd_signal, _ = talib.MACD(spx[price_col], fastperiod=12, slowperiod=26, signalperiod=9)
spx['MACD'] = macd
spx['MACD_Signal'] = macd_signal
spx['MACD_Hist'] = macd - macd_signal

# 200 DMA
spx['200DMA'] = spx[price_col].rolling(window=200).mean()

# Rate of change (10 days)
spx['ROC_10'] = spx[price_col].pct_change(10) * 100

# VIX indicators
vix['VIX_Spike'] = vix['Close'] > 25
vix['VIX_Falling'] = vix['Close'].shift(1) > vix['Close']
vix['VIX_Spike_Then_Fall'] = vix['VIX_Spike'] & vix['VIX_Falling']

# Join VIX to SPX
spx = spx.join(vix['Close'].rename('VIX_Close'), how='left')
spx = spx.join(vix['VIX_Spike_Then_Fall'], how='left')

# --- Sentiment and Macro placeholders (Replace with API data in future) ---
# Put/Call ratio dummy (example > 1.2 = fear)
spx['PutCallRatio_Fear'] = False
# AAII bearish sentiment dummy > 50%
spx['AAII_Bearish'] = False
# Fear & Greed index dummy < 20 = fear
spx['FearGreed_Fear'] = False
# Macro catalyst dummy (can be based on news sentiment NLP)
spx['Macro_Catalyst'] = False

# --- Technical oversold signals ---
spx['RSI_Oversold'] = spx['RSI'] < 30
spx['MACD_BullishCross'] = (spx['MACD'] > spx['MACD_Signal']) & (spx['MACD'].shift(1) <= spx['MACD_Signal'].shift(1))

# Breadth proxy: % of days below 200DMA (simplified for demo)
spx['Below_200DMA'] = spx[price_col] < spx['200DMA']

# Rate of change < -7%
spx['ROC_10_Low'] = spx['ROC_10'] < -7

# Reversal candle: Higher low + falling volume
spx['Higher_Low'] = spx[price_col] > spx[price_col].shift(5)
spx['Volume_Falling'] = spx['Volume'] < spx['Volume'].shift(5)
spx['Reversal_Candle'] = spx['Higher_Low'] & spx['Volume_Falling']

# Drawdown trigger (7-12%)
spx['Drawdown_7_12'] = spx['Drawdown'].between(-0.12, -0.07)
# Drawdown trigger (>=10%)
spx['Drawdown_10'] = spx['Drawdown'] <= -0.10

# Composite sentiment bearish (any of the dummy fear flags)
spx['Sentiment_Bearish'] = spx[['PutCallRatio_Fear', 'AAII_Bearish', 'FearGreed_Fear']].any(axis=1)

# --- Buy Score Calculation ---
def calc_buy_score(row):
    score = 0
    if row['Drawdown_10']: score += 2
    if row['VIX_Spike_Then_Fall']: score += 1
    if row['RSI_Oversold']: score += 1
    if row['Sentiment_Bearish']: score += 1
    if row['Macro_Catalyst']: score += 1
    if row['Reversal_Candle']: score += 1
    return score

spx['Buy_Score'] = spx.apply(calc_buy_score, axis=1)

# --- Show Latest Data ---
latest = spx.iloc[-1]

st.subheader("Latest Market Status")
st.markdown(f"""
**Date:** {latest.name.date()}  
**SPX Price:** {latest[price_col]:.2f}  
**Drawdown:** {latest['Drawdown']*100:.2f}%  
**RSI:** {latest['RSI']:.1f}  
**MACD Histogram:** {latest['MACD_Hist']:.4f}  
**VIX:** {latest['VIX_Close']:.2f}  
**Buy Score:** {latest['Buy_Score']} / 6
""")

st.subheader("Buy Signal Conditions")
cols = st.columns(3)

conds = [
    ("Drawdown >10%", latest['Drawdown_10']),
    ("VIX Spike (>25) then Fall", latest['VIX_Spike_Then_Fall']),
    ("RSI < 30 (Oversold)", latest['RSI_Oversold']),
    ("Sentiment Bearish (Composite)", latest['Sentiment_Bearish']),
    ("Macro Catalyst Present", latest['Macro_Catalyst']),
    ("Reversal Candle (Higher Low + Falling Volume)", latest['Reversal_Candle']),
]

for i, (desc, val) in enumerate(conds):
    status = "✅" if val else "❌"
    cols[i % 3].metric(desc, status)

# --- Plots ---
import matplotlib.pyplot as plt

fig, axs = plt.subplots(5,1, figsize=(12,14), sharex=True)

axs[0].plot(spx.index, spx[price_col], label='SPX Price')
axs[0].plot(spx.index, spx['200DMA'], label='200DMA', linestyle='--')
axs[0].set_ylabel('Price')
axs[0].legend()

axs[1].plot(spx.index, spx['Drawdown']*100, label='Drawdown %', color='orange')
axs[1].axhline(-10, color='red', linestyle='--', label='-10% Threshold')
axs[1].set_ylabel('Drawdown %')
axs[1].legend()

axs[2].plot(spx.index, spx['RSI'], label='RSI', color='purple')
axs[2].axhline(30, color='red', linestyle='--')
axs[2].set_ylabel('RSI')
axs[2].legend()

axs[3].plot(spx.index, spx['MACD_Hist'], label='MACD Histogram', color='green')
axs[3].axhline(0, color='black', linestyle='--')
axs[3].set_ylabel('MACD Hist')
axs[3].legend()

axs[4].plot(spx.index, spx['Buy_Score'], label='Buy Score', color='blue')
axs[4].set_ylim(-0.5, 6.5)
axs[4].set_ylabel('Buy Score')
axs[4].legend()

plt.xticks(rotation=30)
st.pyplot(fig)

# --- Raw Data Preview ---
st.subheader("Raw Data (last 20 rows)")
st.dataframe(spx.tail(20))
