import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolume

st.set_page_config(layout="wide")
st.title("📊 SPX Multi-Indicator Market Score Dashboard")

# --- Adatok letöltése és előkészítése ---

@st.cache_data(ttl=300)
def fetch_data(ticker, period="6mo", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    # Ha multiindex van az oszlopokban, akkor bontsuk ki (pl. SPX adatoknál előfordulhat)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    df.dropna(how='all', inplace=True)
    return df

spx = fetch_data("^GSPC")
vix = fetch_data("^VIX")

# Árfolyam oszlop kiválasztása
price_col = "Adj Close" if "Adj Close" in spx.columns else "Close"
vix_price_col = "Adj Close" if "Adj Close" in vix.columns else "Close"

if spx.empty or price_col not in spx.columns:
    st.error("❌ SPX adatok nem érhetők el vagy hiányzik az árfolyam oszlop.")
    st.stop()

if vix.empty or vix_price_col not in vix.columns:
    st.warning("⚠️ VIX adat nem érhető el vagy hiányzik az árfolyam oszlop.")
    # Ez nem állítja meg az appot, csak nem számol VIX-et

# --- RSI számítás (biztosítva 1D sorozat) ---
price_series = spx[price_col].dropna()
try:
    rsi = RSIIndicator(close=price_series, window=14).rsi()
    spx.loc[rsi.index, "RSI"] = rsi
except Exception as e:
    st.error(f"RSI számítás hiba: {e}")
    spx["RSI"] = np.nan

# --- Egyéb indikátorok ---
try:
    macd_indicator = MACD(close=price_series)
    spx.loc[price_series.index, "MACD"] = macd_indicator.macd()
    spx.loc[price_series.index, "MACD_signal"] = macd_indicator.macd_signal()

    bollinger = BollingerBands(close=price_series)
    spx.loc[price_series.index, "BB_high"] = bollinger.bollinger_hband()
    spx.loc[price_series.index, "BB_low"] = bollinger.bollinger_lband()

    stoch = StochasticOscillator(high=spx["High"], low=spx["Low"], close=price_series)
    spx.loc[price_series.index, "Stoch"] = stoch.stoch()

    obv = OnBalanceVolume(close=price_series, volume=spx["Volume"])
    spx.loc[price_series.index, "OBV"] = obv.on_balance_volume()

except Exception as e:
    st.error(f"Technikai indikátor számítás hiba: {e}")

# --- Drawdown számítás ---
spx["Drawdown"] = spx[price_col] / spx[price_col].cummax() - 1

# --- VIX adatok, ha elérhető ---
vix_current = None
vix_trend_falling = False
if not vix.empty and vix_price_col in vix.columns:
    vix_current = vix[vix_price_col].iloc[-1]
    vix_diff = vix[vix_price_col].diff().dropna()
    vix_trend_falling = vix_diff.iloc[-5:].mean() < 0  # utolsó 5 nap átlagos trendje csökkenő

# --- Buy/Sell Score számítása max 6 pont ---
spx["Buy_Score"] = 0
spx["Sell_Score"] = 0

# Buy pontok
spx.loc[price_series.index, "Buy_Score"] += (spx["RSI"] < 30).astype(int)
spx.loc[price_series.index, "Buy_Score"] += (spx["Drawdown"] < -0.10).astype(int)
spx.loc[price_series.index, "Buy_Score"] += (vix_current is not None and (vix_current > 25 and vix_trend_falling)).astype(int)
spx.loc[price_series.index, "Buy_Score"] += ((spx["MACD"] > spx["MACD_signal"]).astype(int))
spx.loc[price_series.index, "Buy_Score"] += ((spx["Stoch"] < 20).astype(int))
spx.loc[price_series.index, "Buy_Score"] += ((spx["Close"] < spx["BB_low"]).astype(int))

# Sell pontok
spx.loc[price_series.index, "Sell_Score"] += (spx["RSI"] > 70).astype(int)
spx.loc[price_series.index, "Sell_Score"] += (spx["Drawdown"] > -0.01).astype(int)
spx.loc[price_series.index, "Sell_Score"] += ((spx["MACD"] < spx["MACD_signal"]).astype(int))
spx.loc[price_series.index, "Sell_Score"] += ((spx["Stoch"] > 80).astype(int))
spx.loc[price_series.index, "Sell_Score"] += ((spx["Close"] > spx["BB_high"]).astype(int))

# --- Grafikonok és vizualizáció ---

st.subheader("📉 SPX Price Chart")
st.line_chart(spx[price_col])

st.subheader("📉 Drawdown")
st.area_chart(spx["Drawdown"])

st.subheader("📈 RSI")
st.line_chart(spx["RSI"])

st.subheader("🟢 Buy & 🔴 Sell Scores (max 6)")
import altair as alt

score_df = spx.loc[price_series.index, ["Buy_Score", "Sell_Score"]].reset_index()

base = alt.Chart(score_df).encode(
    x='Date:T'
)

buy_circles = base.mark_circle(color="green", size=60).encode(
    y='Buy_Score:Q',
    tooltip=['Date:T', 'Buy_Score:Q']
).transform_filter(
    'datum.Buy_Score > 0'
)

sell_circles = base.mark_circle(color="red", size=60).encode(
    y='Sell_Score:Q',
    tooltip=['Date:T', 'Sell_Score:Q']
).transform_filter(
    'datum.Sell_Score > 0'
)

buy_line = base.mark_line(color="green").encode(y='Buy_Score:Q')
sell_line = base.mark_line(color="red").encode(y='Sell_Score:Q')

chart = (buy_line + sell_line + buy_circles + sell_circles).properties(height=300)
st.altair_chart(chart, use_container_width=True)

st.subheader("📊 Raw Data (last 30 rows)")
st.dataframe(spx.tail(30))
