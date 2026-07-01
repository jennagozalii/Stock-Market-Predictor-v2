# Stock Market Predictor v2 📈

An LSTM-based Streamlit app that predicts next-day stock closing prices for
**any Yahoo Finance ticker**, trained across a diverse basket of stocks
(tech, financials, healthcare, energy, consumer staples, and more) rather
than a single stock.

> **Disclaimer:** Educational project only - not financial advice. The model
> is trained purely on historical closing prices, with no knowledge of news,
> earnings, or market sentiment. Treat predictions as a talking point, not a
> trading signal. See the in-app warning for details.

## Live demo

🔗 https://stock-market-predictor-v2.streamlit.app/ 

## What's in this repo

| File | Purpose |
|---|---|
| `app_v2.py` | The Streamlit web app |
| `train_model.py` | Script that trains the multi-ticker LSTM model |
| `Stock_Predictions_Model_v2.keras` | Pre-trained model |
| `requirements.txt` | Python dependencies needed to run/deploy the app |

## How it works

1. **Data** - pulls daily historical closing prices for a chosen ticker
   from Yahoo Finance (`yfinance`).
2. **Normalization** - instead of scaling by one global price range (which
   breaks down across tickers with very different price levels), each
   100-day input window is scaled by its **own** min/max. This lets the
   model learn the *shape* of price movement rather than an absolute price
   level, which is what makes it usable across many different stocks.
3. **Model** - a stacked LSTM (4 layers, with dropout) trained across ~30
   tickers spanning different sectors and price ranges, using the
   per-window scaling above.
4. **Prediction** - the app shows a historical backtest (predicted vs.
   actual on a held-out test split) plus a live next-day price prediction
   for whatever ticker you enter.

## Improvements from v1 → v2

### Bug fixes
- **Fixed a data-leakage bug** - the original scaler was re-fit (`fit_transform`) on the test set instead of only the training set, letting test-set statistics leak into scaling and quietly inflating apparent accuracy. Now fit once on train, `.transform()` only elsewhere.
- **Fixed a memory leak** - matplotlib figures weren't being closed after rendering, which builds up over time on a long-running deployed app. Added `plt.close(fig)` after every chart.
- **Fixed stale data** - the end date was hardcoded to a fixed past date instead of pulling up to the current day.
- **Fixed missing error handling** - an invalid ticker symbol used to crash the app; now shows a clear message instead.

### Performance
- **Added caching** (`@st.cache_data`, `@st.cache_resource`) so the model and Yahoo Finance data aren't re-downloaded/re-loaded on every single widget interaction.

### Generalization - the core v2 upgrade
- **Replaced global min-max scaling with per-window normalization.** The v1 model was scaled using one stock's absolute price range, so it only ever "understood" prices in that narrow band - meaningless for a stock at a very different price level. Each 100-day window is now scaled by its own min/max, so the model learns the *shape* of price movement rather than an absolute price level.
- **Added RMSE and MAPE metrics** on the backtest, so accuracy is quantified rather than just visual.
- **Added a real next-day forecast** (last close → predicted close → % change) instead of only a historical backtest chart.
- **Added `train_model.py`** as a reproducible, documented retraining script.


## Business value

Use to quickly gauge trend direction and get a data-driven reference point - not as a standalone basis for investment decisions.

- **Trend at a glance (moving averages).** Daily prices are noisy; the MA50/MA100/MA200 lines smooth that out so a stakeholder can immediately see "this stock has been climbing for months" or "this one's been sliding," without reading raw data.
- **A directional forecast, clearly labeled as such.** The next-day prediction gives a concrete number to react to - but the app is explicit that it's based only on price history, not news, earnings, or fundamentals, so it's a talking point and sanity check, not a trading signal.
- **A visible accuracy check (RMSE/MAPE).** The backtest shows exactly how close past predictions were to actual prices - so anyone using it can calibrate how much weight to give it.


## Retraining the model

`train_model.py` downloads ~14 years of daily data across ~30 tickers and
retrains the model from scratch using per-window normalization.


## Known limitations

- Trained on a single feature (closing price only) - no volume, technical
  indicators, or macro/news signals.
- Not a reliable trading signal on its own - markets are influenced by
  information (news, earnings, sentiment) the model never sees.
- Very new tickers (recent IPOs) or highly illiquid/volatile stocks may fall
  outside the price behavior the training basket covered, so predictions
  there should be trusted even less than usual.
- No walk-forward validation or hyperparameter tuning was performed - a
  single chronological 80/20 split per ticker was used.

## Tech stack

Python · TensorFlow / Keras · Streamlit · yfinance · pandas · NumPy · Matplotlib
