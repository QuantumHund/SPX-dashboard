import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.set_page_config(layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard")

@st.cache_data(ttl=300)
def fetch_data(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            st.warning(f"⚠️ Nem érkeztek adatok a {ticker} tickerhez.")
            return pd.DataFrame()
        # ha multiindex az oszlop
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        st.error(f"Adatlekérés hiba a {ticker} esetén: {e}")
        return pd.DataFrame()

spx = fetch_data("^GSPC")

if spx.empty:
    st.error("❌ SPX adatok nem érhetők el.")
    st.stop()

vix = fetch_data("^VIX")

# Debug print oszlopok
st.write("SPX oszlopok:", spx.columns.tolist())
st.write("VIX oszlopok:", vix.columns.tolist() if not vix.empty else "Nincs VIX adat")

price_col = None
for col in ['Adj Close', 'Close']:
    if col in spx.columns:
        price_col = col
        break

if price_col is None:
    st.error("❌ Nem található 'Close' vagy 'Adj Close' oszlop az SPX adatok között.")
    st.stop()

# Biztosítsuk, hogy 1D legyen az árfolyam sorozat
spx_close = spx[price_col]
if isinstance(spx_close, pd.DataFrame):
    spx_close = spx_close.squeeze()

try:
    spx['Drawdown'] = (spx_close / spx_close.cummax()) - 1
    spx['RSI'] = ta.momentum.RSIIndicator(spx_close, window=14).rsi()
    spx['SMA50'] = spx_close.rolling(window=50).mean()
    spx['SMA200'] = spx_close.rolling(window=200).mean()

    macd = ta.trend.MACD(spx_close)
    spx['MACD'] = macd.macd()
    spx['MACD_signal'] = macd.macd_signal()

    bollinger = ta.volatility.BollingerBands(spx_close)
    spx['BB_upper'] = bollinger.bollinger_hband()
    spx['BB_lower'] = bollinger.bollinger_lband()

    # A High és Low oszlopoknál is érdemes ellenőrizni és squeeze-elni, ha szükséges
    spx_high = spx['High']
    spx_low = spx['Low']
    if isinstance(spx_high, pd.DataFrame):
        spx_high = spx_high.squeeze()
    if isinstance(spx_low, pd.DataFrame):
        spx_low = spx_low.squeeze()

    stoch = ta.momentum.StochasticOscillator(spx_high, spx_low, spx_close)
    spx['Stoch'] = stoch.stoch()
except Exception as e:
    st.error(f"Hiba az indikátorok számításakor: {e}")
    st.stop()

spx['Buy_Score'] = (
    ((spx['RSI'] < 30).astype(int)) +
    ((spx['Drawdown'] < -0.10).astype(int)) +
    ((spx['MACD'] > spx['MACD_signal']).astype(int)) +
    ((spx['SMA50'] > spx['SMA200']).astype(int)) +
    ((spx_close < spx['BB_lower']).astype(int)) +
    ((spx['Stoch'] < 20).astype(int))
)

spx['Sell_Score'] = (
    ((spx['RSI'] > 70).astype(int)) +
    ((spx['Drawdown'] > -0.01).astype(int)) +
    ((spx['MACD'] < spx['MACD_signal']).astype(int)) +
    ((spx['SMA50'] < spx['SMA200']).astype(int)) +
    ((spx_close > spx['BB_upper']).astype(int)) +
    ((spx['Stoch'] > 80).astype(int))
)

import altair as alt

st.subheader("📉 SPX Árfolyam")
st.line_chart(spx_close)

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
st.line_chart(spx['RSI'])

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
