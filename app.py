import streamlit as st
import pandas as pd
import yfinance as yf
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")
st.title("📊 SPX Live Dashboard")

st_autorefresh(interval=300000, key="data_refresh")  # Refresh every 5 min

@st.cache_data(ttl=300)
def fetch_spx_data():
    try:
        df = yf.download("^GSPC", period="6mo", interval="1d", group_by="ticker", progress=False)

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

# Use Close column if Adj Close is missing
price_col = 'Adj Close' if 'Adj Close' in spx.columns else 'Close'

if spx.empty or price_col not in spx.columns:
    st.error(f"❌ Failed to fetch SPX data or '{price_col}' column is missing.")
    st.write("Debug - Columns received:", spx.columns)
    st.stop()

# Calculate drawdown using chosen price column
spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1

# Calculate RSI
rsi_window = 14
delta = spx[price_col].diff()
gain = delta.clip(lower=0).rolling(rsi_window).mean()
loss = -delta.clip(upper=0).rolling(rsi_window).mean()
rs = gain / loss
spx['RSI'] = 100 - (100 / (1 + rs))

# Buy/Sell signals
spx['Buy_Score'] = ((spx['RSI'] < 30) & (spx['Drawdown'] < -0.05)).astype(int)
spx['Sell_Score'] = ((spx['RSI'] > 70) & (spx['Drawdown'] > -0.01)).astype(int)

# Charts
st.subheader("📉 SPX Price Chart")
st.line_chart(spx[[price_col]])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("🟢 Buy & 🔴 Sell Signals")
st.line_chart(spx[['Buy_Score', 'Sell_Score']])

st.subheader("📊 Raw Data Table")
st.dataframe(spx.tail(30))
