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
        return df
    except Exception as e:
        st.error(f"Adatlekérés hiba a {ticker} esetén: {e}")
        return pd.DataFrame()

# SPX adat
spx = fetch_data("^GSPC")

# Debug: mutatjuk az oszlopokat, ha jön adat
if not spx.empty:
    st.write("SPX oszlopok:", spx.columns.tolist())
else:
    st.error("❌ SPX adatok nem érhetők el.")
    st.stop()

# VIX adat
vix = fetch_data("^VIX")

if not vix.empty:
    st.write("VIX oszlopok:", vix.columns.tolist())
else:
    st.warning("⚠️ VIX adatok nem érhetők el.")

# Megnézzük, van-e Close vagy Adj Close az SPX-ben
price_col = None
for col in ['Adj Close', 'Close']:
    if col in spx.columns:
        price_col = col
        break

if price_col is None:
    st.error("❌ Nem található 'Close' vagy 'Adj Close' oszlop az SPX adatok között.")
    st.stop()

# Indikátorok számítása (ha adatok megvannak)
try:
    # Drawdown
    spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1

    # RSI
    spx['RSI'] = ta.momentum.RSIIndicator(spx[price_col], window=14).rsi()

    # SMA 50, 200
    spx['SMA50'] = spx[price_col].rolling(window=50).mean()
    spx['SMA200'] = spx[price_col].rolling(window=200).mean()

    # MACD
    macd = ta.trend.MACD(spx[price_col])
    spx['MACD'] = macd.macd()
    spx['MACD_signal'] = macd.macd_signal()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(spx[price_col])
    spx['BB_upper'] = bollinger.bollinger_hband()
    spx['BB_lower'] = bollinger.bollinger_lband()

    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(spx['High'], spx['Low'], spx[price_col])
    spx['Stoch'] = stoch.stoch()
    
except Exception as e:
    st.error(f"Hiba az indikátorok számításakor: {e}")
    st.stop()

# Egyszerű pontozás az indikátorok alapján (max 6 pont)
spx['Buy_Score'] = (
    ((spx['RSI'] < 30).astype(int)) +
    ((spx['Drawdown'] < -0.10).astype(int)) +  # nagyobb drawdown = jobb vételi pont
    ((spx['MACD'] > spx['MACD_signal']).astype(int)) +  # bullish MACD crossover
    ((spx['SMA50'] > spx['SMA200']).astype(int)) +  # golden cross
    ((spx[price_col] < spx['BB_lower']).astype(int)) +  # ár a Bollinger alsó szalag alatt
    ((spx['Stoch'] < 20).astype(int))  # stochastic oversold
)

spx['Sell_Score'] = (
    ((spx['RSI'] > 70).astype(int)) +
    ((spx['Drawdown'] > -0.01).astype(int)) +
    ((spx['MACD'] < spx['MACD_signal']).astype(int)) +
    ((spx['SMA50'] < spx['SMA200']).astype(int)) +
    ((spx[price_col] > spx['BB_upper']).astype(int)) +
    ((spx['Stoch'] > 80).astype(int))
)

# Grafikonok
st.subheader("📉 SPX Árfolyam")
st.line_chart(spx[price_col])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
st.line_chart(spx['RSI'])

st.subheader("🟢 Buy & 🔴 Sell Score (0-6)")
import altair as alt

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

# Részletes adat táblázat
st.subheader("📊 Részletes adatok (utolsó 30 sor)")
st.dataframe(spx.tail(30))

