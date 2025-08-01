import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import altair as alt

st.set_page_config(layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard")

@st.cache_data(ttl=300)
def fetch_data(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            st.warning(f"⚠️ Nem érkeztek adatok a {ticker} tickerhez.")
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.error(f"Adatlekérés hiba a {ticker} esetén: {e}")
        return pd.DataFrame()

spx = fetch_data("^GSPC")
vix = fetch_data("^VIX")

if spx.empty:
    st.error("❌ SPX adatok nem érhetők el.")
    st.stop()

price_col = None
for col in ['Adj Close', 'Close']:
    if col in spx.columns:
        price_col = col
        break

if price_col is None:
    st.error("❌ Nem található 'Close' vagy 'Adj Close' oszlop az SPX adatok között.")
    st.stop()

try:
    spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1
    spx['RSI'] = ta.momentum.RSIIndicator(spx[price_col], window=14).rsi()
    spx['SMA50'] = spx[price_col].rolling(window=50).mean()
    spx['SMA200'] = spx[price_col].rolling(window=200).mean()

    macd = ta.trend.MACD(spx[price_col])
    spx['MACD'] = macd.macd()
    spx['MACD_signal'] = macd.macd_signal()

    bollinger = ta.volatility.BollingerBands(spx[price_col])
    spx['BB_upper'] = bollinger.bollinger_hband()
    spx['BB_lower'] = bollinger.bollinger_lband()

    stoch = ta.momentum.StochasticOscillator(spx['High'], spx['Low'], spx[price_col])
    spx['Stoch'] = stoch.stoch()
except Exception as e:
    st.error(f"Hiba az indikátorok számításakor: {e}")
    st.stop()

spx['Buy_Score'] = (
    ((spx['RSI'] < 30).astype(int)) +
    ((spx['Drawdown'] < -0.10).astype(int)) +
    ((spx['MACD'] > spx['MACD_signal']).astype(int)) +
    ((spx['SMA50'] > spx['SMA200']).astype(int)) +
    ((spx[price_col] < spx['BB_lower']).astype(int)) +
    ((spx['Stoch'] < 20).astype(int))
)

spx['Sell_Score'] = (
    ((spx['RSI'] > 70).astype(int)) +
    ((spx['Drawdown'] > -0.01).astype(int)) +
    ((spx['MACD'] < spx['MACD_signal']).astype(int)) +
    ((spx['SMA50'] < spx['SMA200']).astype(int)) +
    ((spx[price_col] > spx['BB_upper']).astype(int)) +
    ((spx['Stoch'] > 80).astype(int))
)

st.subheader("📉 SPX Árfolyam")
st.line_chart(spx[price_col])

if not vix.empty:
    vix_chart = alt.Chart(vix.reset_index()).mark_line(color='orange').encode(
        x='Date:T',
        y=alt.Y('Close:Q', title='VIX')
    ).properties(title="📈 VIX Index")
    st.altair_chart(vix_chart, use_container_width=True)

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
rsi_df = spx.reset_index()[['Date', 'RSI']].dropna()
rsi_chart = alt.Chart(rsi_df).mark_line(color='blue').encode(
    x='Date:T',
    y=alt.Y('RSI:Q', scale=alt.Scale(domain=[0, 100]))
)

rsi_background = alt.Chart(rsi_df).mark_rect(opacity=0.1, color='green').encode(
    x='Date:T',
    y=alt.Y('RSI:Q', bin=alt.Bin(step=1))
).transform_filter('datum.RSI < 30')

rsi_background_red = alt.Chart(rsi_df).mark_rect(opacity=0.1, color='red').encode(
    x='Date:T',
    y=alt.Y('RSI:Q', bin=alt.Bin(step=1))
).transform_filter('datum.RSI > 70')

st.altair_chart(rsi_chart + rsi_background + rsi_background_red, use_container_width=True)

st.subheader("🟢 Buy & 🔴 Sell Score (0-6)")
df_scores = spx.reset_index()[['Date', 'Buy_Score', 'Sell_Score']]
df_scores = df_scores.melt(id_vars='Date', value_vars=['Buy_Score', 'Sell_Score'], var_name='Signal', value_name='Score')

color_scale = alt.Scale(domain=['Buy_Score', 'Sell_Score'], range=['green', 'red'])
chart = alt.Chart(df_scores).mark_line().encode(
    x='Date:T',
    y='Score:Q',
    color=alt.Color('Signal:N', scale=color_scale),
    tooltip=['Date:T', 'Signal:N', 'Score:Q']
).interactive()

st.altair_chart(chart, use_container_width=True)

st.subheader("📊 Részletes adatok (utolsó 30 sor)")
st.dataframe(spx.tail(30))
