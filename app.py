import streamlit as st
import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator

st.title("TA teszt: RSI SPX")

spx = yf.download("^GSPC", period="3mo", interval="1d")
if not spx.empty:
    spx['RSI'] = RSIIndicator(close=spx['Close'], window=14).rsi()
    st.line_chart(spx[['Close', 'RSI']])
else:
    st.error("❌ Nem sikerült SPX adatot letölteni.")
