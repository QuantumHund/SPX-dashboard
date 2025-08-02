import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import altair as alt

st.set_page_config(layout="wide")
st.title("📊 SPX 8-Factor Market Score Dashboard")

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

price_col = None
for col in ['Adj Close', 'Close']:
    if col in spx.columns:
        price_col = col
        break

if spx.empty or price_col is None:
    st.error("❌ SPX adatok nem érhetők el vagy hiányzik az árfolyam oszlop.")
    st.stop()

try:
    spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1
    spx['RSI'] = ta.momentum.RSIIndicator(spx[price_col], window=14).rsi()
    spx['SMA50'] = spx[price_col].rolling(window=50).mean()
    spx['SMA200'] = spx[price_col].rolling(window=200).mean()

    macd = ta.trend.MACD(spx[price_col])
    spx['MACD'] = macd.macd()
    spx['MACD_signal'] = macd.macd_signal()

    boll = ta.volatility.BollingerBands(spx[price_col])
    spx['BB_upper'] = boll.bollinger_hband()
    spx['BB_lower'] = boll.bollinger_lband()

    stoch = ta.momentum.StochasticOscillator(spx['High'], spx['Low'], spx[price_col])
    spx['Stoch'] = stoch.stoch()

    # OBV manuális számítása
    spx['OBV'] = (np.where(spx[price_col] > spx[price_col].shift(1),
                           spx['Volume'],
                           np.where(spx[price_col] < spx[price_col].shift(1),
                                    -spx['Volume'], 0))).cumsum()
    spx['OBV_trend'] = spx['OBV'].diff()

    # VIX feltételekhez merge
    if not vix.empty and price_col in vix.columns:
        spx['VIX'] = vix[price_col]
    else:
        spx['VIX'] = np.nan

except Exception as e:
    st.error(f"Hiba az indikátorok számításakor: {e}")
    st.stop()

# Scoring logika
spx['Buy_Score'] = (
    (spx['RSI'] < 30).astype(int) +
    (spx['Drawdown'] < -0.10).astype(int) +
    (spx['MACD'] > spx['MACD_signal']).astype(int) +
    (spx['SMA50'] > spx['SMA200']).astype(int) +
    (spx[price_col] < spx['BB_lower']).astype(int) +
    (spx['Stoch'] < 20).astype(int) +
    (spx['OBV_trend'] > 0).astype(int) +
    (spx['VIX'] < 15).astype(int)
)

spx['Sell_Score'] = (
    (spx['RSI'] > 70).astype(int) +
    (spx['Drawdown'] > -0.01).astype(int) +
    (spx['MACD'] < spx['MACD_signal']).astype(int) +
    (spx['SMA50'] < spx['SMA200']).astype(int) +
    (spx[price_col] > spx['BB_upper']).astype(int) +
    (spx['Stoch'] > 80).astype(int) +
    (spx['OBV_trend'] < 0).astype(int) +
    (spx['VIX'] > 20).astype(int)
)

# Chartok
st.subheader("📉 SPX Árfolyam")
st.line_chart(spx[price_col])

st.subheader("📈 RSI (Relative Strength Index)")
rsi_chart = alt.Chart(spx.reset_index()).mark_line().encode(
    x='Date:T', y='RSI:Q'
).properties(height=200).interactive()
st.altair_chart(rsi_chart, use_container_width=True)

st.subheader("📉 OBV")
st.line_chart(spx['OBV'])

st.subheader("🌪️ VIX")
st.line_chart(spx['VIX'])

# Score chart
st.subheader("🟢 Buy & 🔴 Sell Score (0-8)")
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
