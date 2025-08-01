import streamlit as st
import yfinance as yf
import pandas as pd

st.title("SPX teszt: Close árak")

spx = yf.download("^GSPC", period="3mo", interval="1d")
if not spx.empty:
    st.line_chart(spx['Close'])
else:
    st.error("❌ Nem sikerült SPX adatot letölteni.")
