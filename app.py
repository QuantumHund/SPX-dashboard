import streamlit as st
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")
st.title("📊 SPX Live Dashboard with RSI (ta)")

# Auto-refresh every 5 minutes
st_autorefresh(interval=300000, key="data_refresh")

@st.cache_data(ttl=300)
def fetch_spx_data():
    try:
        df = yf.download("^GSPC", period="6mo", interval="1d", progress=False)

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)

        df = df.dropna(how='all', axis=1)
        return df

    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

spx = fetch_spx_data()

# Determine price column
price_col = 'Adj Close' if 'Adj Close' in spx.columns else 'Close'

if spx.empty or price_col not in spx.columns:
    st.error(f"❌ Failed to fetch SPX data or '{price_col}' column is missing.")
    st.write("Debug - Columns received:", spx.columns)
    st.stop()

# --- Indicators ---

# Ensure 1D Series
close_series = spx[price_col]
if isinstance(close_series, pd.DataFrame):
    close_series = close_series.iloc[:, 0]

# RSI (ta)
spx['RSI'] = RSIIndicator(close=close_series, window=14).rsi()

# Drawdown
spx['Drawdown'] = (close_series / close_series.cummax()) - 1

# Scoring: Buy and Sell scores from 0–3
spx['Buy_Score'] = (
    (spx['RSI'] < 30).astype(int) +
    (spx['Drawdown'] < -0.05).astype(int) +
    (spx['RSI'] < 20).astype(int)
)

spx['Sell_Score'] = (
    (spx['RSI'] > 70).astype(int) +
    (spx['Drawdown'] > -0.01).astype(int) +
    (spx['RSI'] > 80).astype(int)
)

# --- Visuals ---

st.subheader("📉 SPX Price Chart")
st.line_chart(spx[[price_col]])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
st.line_chart(spx['RSI'])

st.subheader("🟢 Buy & 🔴 Sell Scores (0–3)")
st.line_chart(spx[['Buy_Score', 'Sell_Score']])

st.subheader("📊 Raw Data Table (last 30 rows)")
st.dataframe(spx.tail(30))
