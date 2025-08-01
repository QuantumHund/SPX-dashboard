import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 SPX Dashboard")

# Fetch SPX data
spx = yf.download("^GSPC", period="6mo", interval="1d", group_by="column")

# Check if data is valid
if spx.empty or 'Adj Close' not in spx.columns:
    st.error("Failed to fetch SPX data or 'Adj Close' column is missing.")
    st.stop()

# Calculate drawdown
spx['Drawdown'] = (spx['Adj Close'] / spx['Adj Close'].cummax()) - 1

# Calculate simple indicators
rsi_window = 14
rsi_numerator = spx['Adj Close'].pct_change().rolling(rsi_window).mean()
rsi_denominator = spx['Adj Close'].pct_change().rolling(rsi_window).std()
spx['RSI'] = 100 - (100 / (1 + rsi_numerator / rsi_denominator))

# Buy/Sell signal logic
spx['Buy_Score'] = ((spx['RSI'] < 30) & (spx['Drawdown'] < -0.05)).astype(int)
spx['Sell_Score'] = ((spx['RSI'] > 70) & (spx['Drawdown'] > -0.01)).astype(int)

# Charts
st.subheader("📉 SPX Price Chart")
st.line_chart(spx[['Adj Close']])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("🟢 Buy & 🔴 Sell Signals")
st.line_chart(spx[['Buy_Score', 'Sell_Score']])

st.subheader("📊 Raw Data Table")
st.dataframe(spx.tail(30))
