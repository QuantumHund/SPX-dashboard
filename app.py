import streamlit as st
import pandas as pd
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")
st.title("📊 SPX Live Dashboard")

# 🔁 Auto-refresh every 5 minutes (300,000 milliseconds)
st_autorefresh(interval=300000, key="data_refresh")

# ✅ Cached data fetcher
@st.cache_data(ttl=300)
def fetch_spx_data():
    try:
        df = yf.download("^GSPC", period="6mo", interval="1d", group_by="ticker", progress=False)

        if df.empty:
            return pd.DataFrame()

        # 🔧 Flatten MultiIndex columns if needed
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)  # ['Open', 'High', ..., 'Adj Close']

        df = df.dropna(how='all', axis=1)
        return df

    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

# 📥 Get data
spx = fetch_spx_data()

# ❌ Handle fetch failure
if spx.empty or 'Adj Close' not in spx.columns:
    st.error("❌ Failed to fetch SPX data or 'Adj Close' column is missing.")
    st.write("Debug - Columns received:", spx.columns)
    st.stop()

# 📉 Calculate Drawdown
spx['Drawdown'] = (spx['Adj Close'] / spx['Adj Close'].cummax()) - 1

# 📈 Calculate RSI
rsi_window = 14
delta = spx['Adj Close'].diff()
gain = delta.clip(lower=0).rolling(rsi_window).mean()
loss = -delta.clip(upper=0).rolling(rsi_window).mean()
rs = gain / loss
spx['RSI'] = 100 - (100 / (1 + rs))

# 🟢🔴 Buy/Sell scores
spx['Buy_Score'] = ((spx['RSI'] < 30) & (spx['Drawdown'] < -0.05)).astype(int)
spx['Sell_Score'] = ((spx['RSI'] > 70) & (spx['Drawdown'] > -0.01)).astype(int)

# 📊 Charts
st.subheader("📉 SPX Price Chart")
st.line_chart(spx[['Adj Close']])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("🟢 Buy & 🔴 Sell Signals")
st.line_chart(spx[['Buy_Score', 'Sell_Score']])

st.subheader("📊 Raw Data Table")
st.dataframe(spx.tail(30))
