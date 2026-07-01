import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import date

# Page setup
st.set_page_config(page_title="Stock Market Predictor", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background-color: #000000; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 style="font-family:Arial Bold; color:White;">Stock Market Predictor</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="font-family:Arial; color:White;">LSTM-based next-day price predictor, '
    'trained across a diverse basket of stocks - works with most Yahoo Finance tickers.</p>',
    unsafe_allow_html=True,
)

st.warning(
    "⚠️This model is trained purely on historical closing prices with no knowledge "
    "of news, earnings, or market sentiment. Treat predictions as a talking point, "
    "not a signal to trade on.",
    icon="⚠️",
)

WINDOW = 100  # how many past days the model looks at to predict the next day
MODEL_FILE = "Stock_Predictions_Model_v2.keras"


# Cached resources
@st.cache_resource
def get_model():
    return load_model(MODEL_FILE)


@st.cache_data(ttl=3600)  # re-download at most once an hour
def get_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.download(ticker, start=start, end=end)


model = get_model()

# User input 
stock = st.text_input("Enter Stock Symbol (any Yahoo Finance ticker)", "GOOG").strip().upper()
start = "2012-01-01"
end = date.today().strftime("%Y-%m-%d")

data = get_stock_data(stock, start, end)

if data.empty or len(data) < WINDOW * 2:
    st.error(
        f"Couldn't find enough data for '{stock}'. "
        "Double-check the ticker symbol (e.g. AAPL, NVDA, KO) and try again."
    )
    st.stop()

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

st.subheader("Stock Data")
st.write(data)


# Chart styling helper
def style_dark_axes(ax):
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="white", which="both")
    ax.figure.set_facecolor("none")
    ax.set_facecolor("none")
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price", color="white")
    ax.legend(facecolor="black", labelcolor="white")


# Moving average charts
ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

st.subheader("Price vs MA50")
fig1, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(data.index, ma_50, "r", label="MA50")
ax1.plot(data.index, data.Close, "g", label="Close")
style_dark_axes(ax1)
st.pyplot(fig1)
plt.close(fig1)

st.subheader("Price vs MA50 vs MA100")
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.plot(data.index, ma_50, "r", label="MA50")
ax2.plot(data.index, ma_100, "b", label="MA100")
ax2.plot(data.index, data.Close, "g", label="Close")
style_dark_axes(ax2)
st.pyplot(fig2)
plt.close(fig2)

st.subheader("Price vs MA100 vs MA200")
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.plot(data.index, ma_100, "r", label="MA100")
ax3.plot(data.index, ma_200, "b", label="MA200")
ax3.plot(data.index, data.Close, "g", label="Close")
style_dark_axes(ax3)
st.pyplot(fig3)
plt.close(fig3)


# Per-window normalization helpers
# This replaces the old global MinMaxScaler. Each window is scaled by its 
# OWN min/max, so the model reasons about relative shape, not absolute price level. 
def build_test_windows(closes: np.ndarray, split_idx: int):
    """Build every (window, target) pair in the test region, each scaled
    by its own min/max. Returns model-ready X plus the min/span needed to
    invert each prediction back to real price."""
    X, mins, spans, targets, target_positions = [], [], [], [], []
    test_start = max(split_idx - WINDOW, 0)
    for i in range(test_start + WINDOW, len(closes)):
        window = closes[i - WINDOW : i]
        w_min, w_max = window.min(), window.max()
        span = w_max - w_min if (w_max - w_min) > 1e-6 else 1e-6
        X.append((window - w_min) / span)
        mins.append(w_min)
        spans.append(span)
        targets.append(closes[i])
        target_positions.append(i)
    return (
        np.array(X).reshape(-1, WINDOW, 1),
        np.array(mins),
        np.array(spans),
        np.array(targets),
        target_positions,
    )


closes = data.Close.values.astype(float)
split_idx = int(len(closes) * 0.80)

X_test, mins, spans, y_actual, target_positions = build_test_windows(closes, split_idx)

if len(X_test) == 0:
    st.warning("Not enough history after the 80/20 split to run a backtest for this ticker.")
else:
    pred_scaled = model.predict(X_test, verbose=0).flatten()
    predicted = pred_scaled * spans + mins

    test_dates = data.index[target_positions]

    st.subheader("Original Price vs Predicted Price (test set)")
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.plot(test_dates, y_actual, "g", label="Original Price")
    ax4.plot(test_dates, predicted, "r", label="Predicted Price")
    style_dark_axes(ax4)
    st.pyplot(fig4)
    plt.close(fig4)

    rmse = float(np.sqrt(np.mean((predicted - y_actual) ** 2)))
    mape = float(np.mean(np.abs((predicted - y_actual) / y_actual)) * 100)
    st.markdown(
        f'<p style="font-family:Arial; color:White;">Test RMSE: {rmse:.2f} &nbsp;|&nbsp; '
        f'Test MAPE: {mape:.2f}%</p>',
        unsafe_allow_html=True,
    )

# Forward-looking prediction for the next trading day
st.subheader("Next-Day Prediction")

last_window = closes[-WINDOW:]
w_min, w_max = last_window.min(), last_window.max()
span = w_max - w_min if (w_max - w_min) > 1e-6 else 1e-6
x_next = ((last_window - w_min) / span).reshape(1, WINDOW, 1)
next_scaled = model.predict(x_next, verbose=0)[0, 0]
next_price = float(next_scaled * span + w_min)

last_close = float(closes[-1])
delta = next_price - last_close
delta_pct = (delta / last_close) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Last Close", f"${last_close:,.2f}")
col2.metric("Predicted Next Close", f"${next_price:,.2f}", f"{delta:+.2f} ({delta_pct:+.2f}%)")
if len(X_test) > 0:
    col3.metric("Test RMSE (this ticker)", f"{rmse:.2f}")

st.caption(
    "⚠️ This prediction is generated by an LSTM trained across a diverse basket of "
    "stocks, using only historical closing prices. It does not account for news, "
    "earnings, macro events, or market sentiment, and should not be used as "
    "financial advice."
)
