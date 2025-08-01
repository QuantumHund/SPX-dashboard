mport streamlit as st
import pandas as pd
import yfinance as yf
import ta
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide")
st.title("📊 SPX Live Dashboard with 'ta' RSI")

# Auto-refresh every 5 minutes
st_autorefresh(interval=300000, key="data_refresh")

@st.cache_data(ttl=300)
def fetch_spx_data():
    try:
        df = yf.download("^GSPC", period="6mo", interval="1d", progress=False)
        if df.empty:
            return pd.DataFrame()

        df = df.dropna(how='all', axis=1)
        return df

    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

spx = fetch_spx_data()

price_col = 'Adj Close' if 'Adj Close' in spx.columns else 'Close'

if spx.empty or price_col not in spx.columns:
    st.error(f"❌ Failed to fetch SPX data or '{price_col}' column is missing.")
    st.write("Debug - Columns received:", spx.columns)
    st.stop()

# Drawdown calculation
spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1

# RSI számítás 'ta' könyvtárral
spx['RSI'] = ta.momentum.RSIIndicator(spx[price_col], window=14).rsi()

# Buy/Sell score egyszerűbb cizelláltabb verzió (0-3 pont)
spx['Buy_Score'] = (
    ((spx['RSI'] < 30).astype(int)) +
    ((spx['Drawdown'] < -0.05).astype(int)) +
    ((spx['RSI'] < 20).astype(int))  # erősebb oversold jelzés
)

spx['Sell_Score'] = (
    ((spx['RSI'] > 70).astype(int)) +
    ((spx['Drawdown'] > -0.01).astype(int)) +
    ((spx['RSI'] > 80).astype(int))  # erősebb overbought jelzés
)

# Grafikonok
st.subheader("📉 SPX Price Chart")
st.line_chart(spx[[price_col]])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
st.line_chart(spx['RSI'])

st.subheader("🟢 Buy & 🔴 Sell Signals (Score 0-3)")
st.line_chart(spx[['Buy_Score', 'Sell_Score']])

st.subheader("📊 Raw Data Table (last 30 rows)")
st.dataframe(spx.tail(30))
