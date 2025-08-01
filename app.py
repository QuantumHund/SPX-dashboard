import streamlit as st
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")
st.title("📊 SPX Live Dashboard (RSI + Drawdown)")

# Auto-refresh every 5 minutes
st_autorefresh(interval=300000, key="data_refresh")

@st.cache_data(ttl=300)
def fetch_spx_data():
    try:
        ticker = yf.Ticker("^GSPC")
        df = ticker.history(period="6mo", interval="1d")

        if df.empty:
            return pd.DataFrame()

        # A history() általában nem MultiIndex, de azért ellenőrzés
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(how='all', axis=1)
        return df

    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

# --- Load and prepare data
spx = fetch_spx_data()
price_col = 'Adj Close' if 'Adj Close' in spx.columns else 'Close'

if spx.empty or price_col not in spx.columns:
    st.error(f"❌ Failed to fetch SPX data or '{price_col}' column is missing.")
    st.write("Debug - Columns received:", list(spx.columns))
    st.stop()

# --- Indicators
spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1
spx['RSI'] = RSIIndicator(close=spx[price_col], window=14).rsi()

# --- Buy/Sell Scoring (0–3)
spx['Buy_Score'] = (
    (spx['RSI'] < 30).astype(int) +
    (spx['RSI'] < 20).astype(int) +
    (spx['Drawdown'] < -0.05).astype(int)
)

spx['Sell_Score'] = (
    (spx['RSI'] > 70).astype(int) +
    (spx['RSI'] > 80).astype(int) +
    (spx['Drawdown'] > -0.01).astype(int)
)

# --- Charts
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
