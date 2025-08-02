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

if spx.empty:
    st.error("❌ SPX adatok nem érhetők el.")
    st.stop()

vix = fetch_data("^VIX")

price_col = None
for col in ['Adj Close', 'Close']:
    if col in spx.columns:
        price_col = col
        break

if price_col is None:
    st.error("❌ Nem található 'Close' vagy 'Adj Close' oszlop az SPX adatok között.")
    st.stop()

# On-Balance Volume számítása
def calculate_obv(df, price_col):
    obv = [0]
    for i in range(1, len(df)):
        if df[price_col].iloc[i] > df[price_col].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df[price_col].iloc[i] < df[price_col].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

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

    spx['OBV'] = calculate_obv(spx, price_col)
except Exception as e:
    st.error(f"Hiba az indikátorok számításakor: {e}")
    st.stop()

# VIX alapján vételi/eladási jelzés (score)
vix_value = vix['Close'].iloc[-1] if not vix.empty else None

# OBV trend vizsgálata (egyszerű lineáris trend)
obv_trend = (spx['OBV'].iloc[-1] - spx['OBV'].iloc[-15]) > 0  # utolsó 15 nap emelkedett-e az OBV

spx['Buy_Score'] = (
    ((spx['RSI'] < 30).astype(int)) +
    ((spx['Drawdown'] < -0.10).astype(int)) +
    ((spx['MACD'] > spx['MACD_signal']).astype(int)) +
    ((spx['SMA50'] > spx['SMA200']).astype(int)) +
    ((spx[price_col] < spx['BB_lower']).astype(int)) +
    ((spx['Stoch'] < 20).astype(int)) +
    (obv_trend.astype(int)) +
    ((vix_value is not None and vix_value < 15).astype(int) if vix_value is not None else 0)
)

spx['Sell_Score'] = (
    ((spx['RSI'] > 70).astype(int)) +
    ((spx['Drawdown'] > -0.01).astype(int)) +
    ((spx['MACD'] < spx['MACD_signal']).astype(int)) +
    ((spx['SMA50'] < spx['SMA200']).astype(int)) +
    ((spx[price_col] > spx['BB_upper']).astype(int)) +
    ((spx['Stoch'] > 80).astype(int)) +
    ((~obv_trend).astype(int)) +
    ((vix_value is not None and vix_value > 20).astype(int) if vix_value is not None else 0)
)

# Indikátor magyarázatok tooltipként
indicators = {
    "RSI": "Relative Strength Index: jelzi a túlvett vagy túladott állapotot (RSI<30 vétel, RSI>70 eladás).",
    "Drawdown": "Az árfolyam visszaesése a legmagasabb értékhez képest, nagy visszaesés vételi lehetőség.",
    "MACD": "Két mozgóátlag konvergenciája és divergenciája, trendfordulók jelzésére.",
    "SMA Golden Cross": "50 napos mozgóátlag metszi a 200 napost – vételi vagy eladási jelzés.",
    "Bollinger Bands": "Az árfolyam volatilitásának mutatója, szélsőséges értékek vételi vagy eladási jelzések.",
    "Stochastic Oscillator": "Árfolyam helyzete a múltbeli árakhoz képest, túladottság vagy túlvétel jelzésére.",
    "OBV": "On-Balance Volume: árfolyam és forgalom összefüggése, trend megerősítésére.",
    "VIX": "Volatilitás index, a piaci félelem szintjét mutatja (VIX>20 eladás, VIX<15 vétel)."
}

st.subheader("Indikátorok magyarázata")
for name, desc in indicators.items():
    st.write(f"**{name}**", "ℹ️", help=desc)

# Grafikonok

st.subheader("📉 SPX Árfolyam")
st.line_chart(spx[price_col])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
rsi_chart = alt.Chart(spx.reset_index()).mark_line().encode(
    x='Date:T',
    y='RSI:Q',
    tooltip=['Date:T', 'RSI:Q']
).properties(height=150)
rsi_rule_30 = alt.Chart(pd.DataFrame({'y':[30]})).mark_rule(color='green', strokeDash=[4,4]).encode(y='y')
rsi_rule_70 = alt.Chart(pd.DataFrame({'y':[70]})).mark_rule(color='red', strokeDash=[4,4]).encode(y='y')

st.altair_chart(rsi_chart + rsi_rule_30 + rsi_rule_70, use_container_width=True)

st.subheader("📊 Buy & Sell Score (0-8)")

df_scores = spx.reset_index()[['Date', 'Buy_Score', 'Sell_Score']]
df_scores = df_scores.melt(id_vars='Date', value_vars=['Buy_Score', 'Sell_Score'], var_name='Signal', value_name='Score')

color_scale = alt.Scale(domain=['Buy_Score', 'Sell_Score'], range=['green', 'red'])

score_chart = alt.Chart(df_scores).mark_line().encode(
    x='Date:T',
    y='Score:Q',
    color=alt.Color('Signal:N', scale=color_scale),
    tooltip=['Date:T', 'Signal:N', 'Score:Q']
).interactive()

st.altair_chart(score_chart, use_container_width=True)

# OBV chart külön
st.subheader("📈 On-Balance Volume (OBV)")
st.line_chart(spx['OBV'])

# VIX chart külön (ha van adat)
if not vix.empty:
    st.subheader("📈 VIX (Volatilitás Index)")
    st.line_chart(vix['Close'])

st.subheader("📊 Részletes adatok (utolsó 30 sor)")
st.dataframe(spx.tail(30))
