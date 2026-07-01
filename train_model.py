"""
train_model.py

Trains a generalizable LSTM stock predictor across MANY tickers instead of
just GOOG, using per-window (not global) normalization.

Output: Stock_Predictions_Model_v2.keras
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from keras.callbacks import EarlyStopping

WINDOW = 100
START = "2010-01-01"
END = date.today().strftime("%Y-%m-%d")

# A deliberately diverse basket: different sectors, price ranges, and volatility profiles, so the model doesn't just re-learn "one type" of stock.
TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA",     # big tech
    "JPM", "BAC", "V", "MA",                                    # financials
    "JNJ", "PFE", "UNH", "MRK",                                 # healthcare
    "XOM", "CVX",                                               # energy
    "KO", "PEP", "WMT", "PG", "MCD", "NKE",                     # consumer staples
    "DIS", "NFLX",                                              # media
    "INTC", "CSCO", "ORCL", "IBM", "AMD",                       # legacy/enterprise tech
    "BA", "CAT", "HD", "T",                                     # industrials/telecom
]


def build_windows(prices: np.ndarray):
    """Turn a 1D price series into (X, y) pairs, each window scaled by its
    OWN min/max — not the ticker's or dataset's global range."""
    X, y = [], []
    for i in range(WINDOW, len(prices)):
        window = prices[i - WINDOW : i]
        target = prices[i]
        w_min, w_max = window.min(), window.max()
        span = w_max - w_min
        if span < 1e-6:
            continue  # skip degenerate/flat windows
        X.append((window - w_min) / span)
        y.append((target - w_min) / span)
    return X, y


def main():
    all_X, all_y = [], []

    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=START, end=END, progress=False)
            if df.empty or len(df) < WINDOW + 50:
                print(f"Skipping {ticker}: not enough data")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes = df["Close"].values.astype(float).flatten()
            X, y = build_windows(closes)
            all_X.extend(X)
            all_y.extend(y)
            print(f"{ticker}: {len(X)} windows (total so far: {len(all_X)})")
        except Exception as e:
            print(f"Failed {ticker}: {e}")

    X = np.array(all_X).reshape(-1, WINDOW, 1)
    y = np.array(all_y)
    print("\nTotal training windows:", X.shape)

    # Safe to shuffle: each window is independently normalized, so unlike a
    # single-stock model there's no global scaling relationship across time
    # that shuffling would break.
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(len(X) * 0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = Sequential([
        LSTM(50, activation="relu", return_sequences=True, input_shape=(WINDOW, 1)),
        Dropout(0.2),
        LSTM(60, activation="relu", return_sequences=True),
        Dropout(0.3),
        LSTM(80, activation="relu", return_sequences=True),
        Dropout(0.4),
        LSTM(120, activation="relu"),
        Dropout(0.5),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1,
    )

    val_loss = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nFinal validation MSE (normalized scale): {val_loss:.6f}")

    model.save("Stock_Predictions_Model_v2.keras")
    print("Saved: Stock_Predictions_Model_v2.keras")


if __name__ == "__main__":
    main()
