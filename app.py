
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("📊 SPX Dashboard")

# Fetch SPX data
spx = yf.download("^GSPC", period="6mo", interval="1d")
spx['Drawdown'] = (spx['Adj Close'] / spx['Adj Close'].cummax()) - 1

# Calculate simple indicators for signals
spx['RSI'] = 100 - (100 / (1 + spx['Adj Close'].pct_change().rolling(14).mean() / spx['Adj Close'].pct_change().rolling(14).std()))
spx['Buy_Score'] = ((spx['RSI'] < 30) & (spx['Drawdown'] < -0.05)).astype(int)
spx['Sell_Score'] = ((spx['RSI'] > 70) & (spx['Drawdown'] > -0.01)).astype(int)

st.subheader("📉 SPX Price Chart")
st.line_chart(spx[['Adj Close']])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("🟢 Buy & 🔴 Sell Signals")
signal_data = spx[['Buy_Score', 'Sell_Score']]
st.line_chart(signal_data)

st.subheader("📊 Raw Data Table")
st.dataframe(spx.tail(30))
