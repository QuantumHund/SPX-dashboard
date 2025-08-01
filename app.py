import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta

st.set_page_config(layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard")

# Adatok letöltése SPX és VIX
@st.cache_data(ttl=300)
def fetch_data(ticker, period="6mo", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return pd.DataFrame()
        df = df.dropna(how='all', axis=1)
        return df
    except Exception as e:
        st.error(f"Data fetch error for {ticker}: {e}")
        return pd.DataFrame()

spx = fetch_data("^GSPC")
vix = fetch_data("^VIX")

price_col = 'Adj Close' if 'Adj Close' in spx.columns else 'Close'

if spx.empty or price_col not in spx.columns:
    st.error(f"❌ Failed to fetch SPX data or '{price_col}' column is missing.")
    st.write("Debug - Columns received:", spx.columns.tolist())
    st.stop()

if vix.empty or 'Close' not in vix.columns:
    st.error("❌ Failed to fetch VIX data or 'Close' column missing.")
    st.stop()

# Drawdown számítása
spx['Drawdown'] = (spx[price_col] / spx[price_col].cummax()) - 1

# RSI számítása
spx['RSI'] = ta.momentum.RSIIndicator(spx[price_col], window=14).rsi()

# Moving Averages (50, 200)
spx['SMA_50'] = ta.trend.SMAIndicator(spx[price_col], window=50).sma_indicator()
spx['SMA_200'] = ta.trend.SMAIndicator(spx[price_col], window=200).sma_indicator()

# Golden/Death Cross jelzés
spx['Golden_Cross'] = ((spx['SMA_50'] > spx['SMA_200']) & (spx['SMA_50'].shift(1) <= spx['SMA_200'].shift(1))).astype(int)
spx['Death_Cross'] = ((spx['SMA_50'] < spx['SMA_200']) & (spx['SMA_50'].shift(1) >= spx['SMA_200'].shift(1))).astype(int)

# MACD számítása
macd = ta.trend.MACD(spx[price_col])
spx['MACD'] = macd.macd()
spx['MACD_Signal'] = macd.macd_signal()
spx['MACD_Bullish_Crossover'] = ((spx['MACD'] > spx['MACD_Signal']) & (spx['MACD'].shift(1) <= spx['MACD_Signal'].shift(1))).astype(int)

# Bollinger Bands
bb = ta.volatility.BollingerBands(spx[price_col])
spx['BB_High'] = bb.bollinger_hband()
spx['BB_Low'] = bb.bollinger_lband()

# Stochastic Oscillator
stoch = ta.momentum.StochasticOscillator(spx['High'], spx['Low'], spx[price_col], window=14, smooth_window=3)
spx['Stoch'] = stoch.stoch()
spx['Stoch_Signal'] = stoch.stoch_signal()

# Manuális OBV számítás
def calculate_obv(df):
    obv = [0]
    for i in range(1, len(df)):
        if df[price_col].iloc[i] > df[price_col].iloc[i-1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df[price_col].iloc[i] < df[price_col].iloc[i-1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)

spx['OBV'] = calculate_obv(spx)

# VIX 25 felett és csökkenő?
vix['VIX_Falling'] = (vix['Close'] < vix['Close'].shift(1)).astype(int)
vix_condition = (vix['Close'] > 25) & (vix['VIX_Falling'] == 1)

# Score kalkuláció (max 6 pont)
spx['Score'] = (
    (spx['Drawdown'] < -0.10).astype(int) +         # nagyobb mint 10% drawdown -> 1 pont
    (spx['RSI'] < 30).astype(int) +                 # RSI oversold -> 1 pont
    spx['MACD_Bullish_Crossover'] +                  # MACD bullish crossover -> 1 pont
    spx['Golden_Cross'] +                             # Golden cross -> 1 pont
    (spx['Stoch'] < 20).astype(int) +                # Stoch oversold -> 1 pont
    # OBV növekedés (aktuális vs 5 nappal korábbi) - ha OBV nőtt, akkor +1 pont
    ((spx['OBV'] > spx['OBV'].shift(5)).astype(int))
)

# Ha VIX adat elérhető, a score-hoz hozzáadhatjuk a VIX jelet
# De mivel külön adat, nem teljesen egy sorban, így ezt jelezzük külön
latest_vix = vix.iloc[-1] if not vix.empty else None
vix_signal = "✅ VIX > 25 és csökkenő" if latest_vix is not None and latest_vix['Close'] > 25 and latest_vix['VIX_Falling'] == 1 else "❌ VIX jelzés nem teljesül"

# Megjelenítés
st.subheader("📉 SPX Close Price")
st.line_chart(spx[price_col])

st.subheader("📉 Drawdown")
st.area_chart(spx['Drawdown'])

st.subheader("📈 RSI")
st.line_chart(spx['RSI'])

st.subheader("⚡ MACD Bullish Crossover")
st.line_chart(spx['MACD_Bullish_Crossover'])

st.subheader("🟢 Golden / 🔴 Death Cross")
st.line_chart(spx[['Golden_Cross', 'Death_Cross']])

st.subheader("📊 Stochastic Oscillator")
st.line_chart(spx[['Stoch', 'Stoch_Signal']])

st.subheader("📈 OBV (On-Balance Volume)")
st.line_chart(spx['OBV'])

st.subheader("📊 Market Score (max 6)")
st.line_chart(spx['Score'])

st.markdown(f"### VIX Jelzés: {vix_signal}")

st.subheader("📊 Raw Data (last 30 rows)")
st.dataframe(spx.tail(30))
