import streamlit as st
import yfinance as yf
import pandas as pd
import ta

st.set_page_config(layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard")

@st.cache_data(ttl=300)
def fetch_data(ticker, period="6mo", interval="1d"):
    return yf.download(ticker, period=period, interval=interval, progress=False)

# Adatok lekérése
spx = fetch_data("^GSPC")
vix = fetch_data("^VIX")

if spx.empty or vix.empty:
    st.error("Nem sikerült letölteni az adatokat.")
    st.stop()

price_col = "Adj Close" if "Adj Close" in spx.columns else "Close"

# 1) Drawdown számítása
spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1

# 2) RSI
spx['RSI'] = ta.momentum.RSIIndicator(spx[price_col], window=14).rsi()

# 3) MACD
macd = ta.trend.MACD(spx[price_col])
spx['MACD'] = macd.macd()
spx['MACD_Signal'] = macd.macd_signal()
spx['MACD_Bullish_Crossover'] = (spx['MACD'] > spx['MACD_Signal']) & (spx['MACD'].shift(1) <= spx['MACD_Signal'].shift(1))

# 4) SMA 50 és 200, Golden Cross
spx['SMA_50'] = spx[price_col].rolling(window=50).mean()
spx['SMA_200'] = spx[price_col].rolling(window=200).mean()
golden_cross = (spx['SMA_50'].iloc[-1] > spx['SMA_200'].iloc[-1])

# 5) Stochastic Oscillator
stoch = ta.momentum.StochasticOscillator(high=spx['High'], low=spx['Low'], close=spx[price_col], window=14, smooth_window=3)
spx['Stoch_K'] = stoch.stoch()
spx['Stoch_D'] = stoch.stoch_signal()
stoch_oversold = spx['Stoch_K'].iloc[-1] < 20

# 6) Bollinger Bands
bb = ta.volatility.BollingerBands(spx[price_col], window=20, window_dev=2)
spx['BB_High'] = bb.bollinger_hband()
spx['BB_Low'] = bb.bollinger_lband()
# Bollinger squeeze - ha a sávok szűkülnek, jelezhet egy mozgást
bb_width = spx['BB_High'] - spx['BB_Low']
bb_squeeze = bb_width.iloc[-1] < bb_width.rolling(window=20).mean().iloc[-1] * 0.7

# 7) OBV (On-Balance Volume)
spx['OBV'] = ta.volume.OnBalanceVolumeIndicator(spx[price_col], spx['Volume']).on_balance_volume()

# 8) VIX elemzés (árfolyam + csökkenő volatilitás)
vix_close = vix['Close']
vix_latest = vix_close.iloc[-1]
vix_prev = vix_close.iloc[-2]
vix_alert = (vix_latest > 25) and (vix_latest < vix_prev)

# Score számítása (max 8 pont)
score = 0
score += 1 if spx['RSI'].iloc[-1] < 30 else 0
score += 1 if spx['Drawdown'].iloc[-1] < -0.10 else 0
score += 1 if vix_alert else 0
score += 1 if spx['MACD_Bullish_Crossover'].iloc[-1] else 0
score += 1 if golden_cross else 0
score += 1 if stoch_oversold else 0
score += 1 if bb_squeeze else 0
# OBV trend vizsgálat: OBV növekszik az elmúlt 5 napban?
obv_trend = spx['OBV'].iloc[-5:].is_monotonic_increasing
score += 1 if obv_trend else 0

def score_color(sc):
    if sc >= 6:
        return "🟢 Erős vételi jel"
    elif sc >= 3:
        return "🟡 Figyelmeztetés"
    else:
        return "🔴 Gyenge jel vagy eladási hangulat"

st.header(f"📊 Összesített Market Score: {score}/8")
st.markdown(f"**Állapot:** {score_color(score)}")

# Grafikonok

st.subheader("SPX Close és Drawdown")
st.line_chart(spx[[price_col, 'Drawdown']])

st.subheader("RSI")
st.line_chart(spx['RSI'])

st.subheader("MACD és jelvonal")
st.line_chart(spx[['MACD', 'MACD_Signal']])

st.subheader("SMA 50 és SMA 200")
st.line_chart(spx[['SMA_50', 'SMA_200']])

st.subheader("Stochastic Oscillator K és D")
st.line_chart(spx[['Stoch_K', 'Stoch_D']])

st.subheader("Bollinger Bands és BB Width")
st.line_chart(spx[[price_col, 'BB_High', 'BB_Low']])
st.line_chart(bb_width)

st.subheader("On-Balance Volume (OBV)")
st.line_chart(spx['OBV'])

st.subheader("VIX Close")
st.line_chart(vix_close)
