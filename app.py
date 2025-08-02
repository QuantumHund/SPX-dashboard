import streamlit as st
import yfinance as yf
import pandas as pd
import ta
import altair as alt
import numpy as np

st.set_page_config(layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard - 8 Faktoros Verzió")

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

# Adatok letöltése
spx = fetch_data("^GSPC")
vix = fetch_data("^VIX")

if spx.empty:
    st.error("❌ SPX adatok nem érhetők el.")
    st.stop()

if vix.empty:
    st.warning("⚠️ VIX adatok nem érhetők el, a VIX mutató nem lesz elérhető.")

# Árfolyam oszlop keresése
price_col = None
for col in ['Adj Close', 'Close']:
    if col in spx.columns:
        price_col = col
        break
if price_col is None:
    st.error("❌ Nem található 'Close' vagy 'Adj Close' oszlop az SPX adatok között.")
    st.stop()

# OBV kézi számítása
def calc_obv(df):
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

    spx['OBV'] = calc_obv(spx)
except Exception as e:
    st.error(f"Hiba az indikátorok számításakor: {e}")
    st.stop()

# 8 faktoros scoring

def vix_score(vix_value):
    if vix_value > 20:
        return 1  # Eladási jelzés
    elif vix_value < 15:
        return 0  # Vételi jelzés
    else:
        return 0  # Semleges

# OBV trend (egyszerűen az OBV növekvő vagy csökkenő)
spx['OBV_diff'] = spx['OBV'].diff()
spx['OBV_Score'] = (spx['OBV_diff'] > 0).astype(int)

# VIX score illesztése SPX indexhez (összehangoljuk dátummal)
spx = spx.merge(vix[['Close']], left_index=True, right_index=True, how='left', suffixes=('', '_VIX'))
spx['VIX_Score'] = spx['Close_VIX'].apply(vix_score).fillna(0).astype(int)

spx['Buy_Score'] = (
    ((spx['RSI'] < 30).astype(int)) +
    ((spx['Drawdown'] < -0.10).astype(int)) +
    ((spx['MACD'] > spx['MACD_signal']).astype(int)) +
    ((spx['SMA50'] > spx['SMA200']).astype(int)) +
    ((spx[price_col] < spx['BB_lower']).astype(int)) +
    ((spx['Stoch'] < 20).astype(int)) +
    spx['OBV_Score'] +
    (spx['VIX_Score'] == 0).astype(int)  # VIX alacsony = vétel
)

spx['Sell_Score'] = (
    ((spx['RSI'] > 70).astype(int)) +
    ((spx['Drawdown'] > -0.01).astype(int)) +
    ((spx['MACD'] < spx['MACD_signal']).astype(int)) +
    ((spx['SMA50'] < spx['SMA200']).astype(int)) +
    ((spx[price_col] > spx['BB_upper']).astype(int)) +
    ((spx['Stoch'] > 80).astype(int)) +
    (spx['OBV_Score'] == 0).astype(int) +  # OBV csökken = eladás
    spx['VIX_Score']  # VIX magas = eladás
)

# Tooltip-es indikátor magyarázatok
indicators = {
    "RSI": "Relatív erősség index – túlvettség vagy túladottság jelző",
    "Drawdown": "Maximális visszaesés az árfolyamban",
    "VIX": "Volatilitási index – piaci félelem mutatója",
    "MACD": "Mozgóátlag konvergencia divergencia – trend és momentum indikátor",
    "SMA50/SMA200": "50 és 200 napos egyszerű mozgóátlag – aranykereszt és halálkereszt jelek",
    "Stochastic Oscillator": "Túlvett és túladott zónák az árfolyam sebességén alapulva",
    "Bollinger Bands": "Árfolyam volatilitás és szélsőségek jelzése",
    "OBV": "On-Balance Volume – volumenváltozások és trend összefüggései"
}

st.subheader("📊 Indikátorok magyarázata (hover az elnevezésen)")

for name, desc in indicators.items():
    st.markdown(f'''
    <p style="display:inline-block; border-bottom:1px dotted black; cursor: help; margin-right: 15px;" title="{desc}"><b>{name}</b></p>
    ''', unsafe_allow_html=True)

# RSI chart színezéssel
def plot_rsi(df):
    base = alt.Chart(df.reset_index()).encode(x='Date:T')

    rsi_line = base.mark_line(color='blue').encode(y='RSI:Q')

    # Háttér színezés túlvettség és túladottság zónákhoz
    overbought = alt.Chart(df.reset_index()).mark_rect(opacity=0.15, color='red').encode(
        y='RSI:Q',
        y2=alt.value(70)
    ).transform_filter(alt.datum.RSI > 70)

    oversold = alt.Chart(df.reset_index()).mark_rect(opacity=0.15, color='green').encode(
        y=alt.value(30),
        y2='RSI:Q'
    ).transform_filter(alt.datum.RSI < 30)

    # De mivel mark_rect nehéz így megoldani így másképp: rajzoljuk a backgroundot egy sávval

    rsi_chart = alt.Chart(df.reset_index()).mark_line(color='blue').encode(
        x='Date:T',
        y='RSI:Q',
    )

    band = alt.Chart(pd.DataFrame({
        'y0': [70], 'y1': [100], 'color': ['red']
    })).mark_rect(opacity=0.1).encode(
        y='y0:Q',
        y2='y1:Q',
        color=alt.value('red')
    )

    band2 = alt.Chart(pd.DataFrame({
        'y0': [0], 'y1': [30], 'color': ['green']
    })).mark_rect(opacity=0.1).encode(
        y='y0:Q',
        y2='y1:Q',
        color=alt.value('green')
    )

    rsi_chart = alt.layer(band, band2, rsi_line).encode(x='Date:T', y='RSI:Q').properties(height=200)

    return rsi_chart

st.subheader("📈 RSI Chart (túlvettség/túladottság háttérszínnel)")
st.altair_chart(plot_rsi(spx), use_container_width=True)

st.subheader("📉 SPX Árfolyam")
st.line_chart(spx[price_col])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📉 VIX Index")
if not vix.empty:
    st.line_chart(vix['Close'])
else:
    st.write("VIX adat nem elérhető.")

st.subheader("📊 Buy & Sell Score (0-8)")

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
