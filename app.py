import streamlit as st
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from streamlit_autorefresh import st_autorefresh
import altair as alt

st.set_page_config(layout="wide")
st.title("📊 SPX Live Dashboard (RSI + Drawdown)")

# Auto-refresh every 5 minutes (300000 ms)
st_autorefresh(interval=300000, key="data_refresh")

@st.cache_data(ttl=300)
def fetch_spx_data():
    try:
        ticker = yf.Ticker("^GSPC")
        df = ticker.history(period="6mo", interval="1d")

        if df.empty:
            return pd.DataFrame()

        # Ha multiindex az oszlop, pl. Adj Close, levesszük a szinteket
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.dropna(how='all', axis=1)
        return df

    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return pd.DataFrame()

spx = fetch_spx_data()
price_col = 'Adj Close' if 'Adj Close' in spx.columns else 'Close'

if spx.empty or price_col not in spx.columns:
    st.error(f"❌ Failed to fetch SPX data or '{price_col}' column is missing.")
    st.write("Debug - Columns received:", list(spx.columns))
    st.stop()

# Drawdown = jelenlegi ár / eddigi legmagasabb ár - 1
spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1

# RSI számítás
spx['RSI'] = RSIIndicator(close=spx[price_col], window=14).rsi()

# Buy Score (0-3 pont)
spx['Buy_Score'] = (
    (spx['RSI'] < 30).astype(int) +
    (spx['RSI'] < 20).astype(int) +
    (spx['Drawdown'] < -0.05).astype(int)
)

# Sell Score (0-3 pont)
spx['Sell_Score'] = (
    (spx['RSI'] > 70).astype(int) +
    (spx['RSI'] > 80).astype(int) +
    (spx['Drawdown'] > -0.01).astype(int)
)

# Altair chart készítése
chart_data = spx.reset_index()[['Date', 'Buy_Score', 'Sell_Score']]

buy_line = alt.Chart(chart_data).mark_line(color='green').encode(
    x='Date:T',
    y='Buy_Score:Q',
    tooltip=['Date:T', 'Buy_Score']
)

sell_line = alt.Chart(chart_data).mark_line(color='red').encode(
    x='Date:T',
    y='Sell_Score:Q',
    tooltip=['Date:T', 'Sell_Score']
)

buy_points = alt.Chart(chart_data[chart_data['Buy_Score'] >= 2]).mark_circle(color='green', size=100).encode(
    x='Date:T',
    y='Buy_Score:Q',
    tooltip=['Date:T', 'Buy_Score']
)

sell_points = alt.Chart(chart_data[chart_data['Sell_Score'] >= 2]).mark_circle(color='red', size=100).encode(
    x='Date:T',
    y='Sell_Score:Q',
    tooltip=['Date:T', 'Sell_Score']
)

final_chart = (buy_line + sell_line + buy_points + sell_points).properties(
    width=900,
    height=400,
    title="🟢 Buy Score és 🔴 Sell Score (kiemelt pontok)"
).interactive()

# Megjelenítés
st.subheader("📉 SPX Price Chart")
st.line_chart(spx[[price_col]])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
st.line_chart(spx['RSI'])

st.subheader("🟢 Buy & 🔴 Sell Scores (0-3) kiemelt pontokkal")
st.altair_chart(final_chart, use_container_width=True)

st.subheader("📊 Raw Data Table (last 30 rows)")
st.dataframe(spx.tail(30))
