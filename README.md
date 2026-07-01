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

🔗 _[Streamlit link]_

## What's in this repo

| File | Purpose |
|---|---|
| `app_v2.py` | The Streamlit web app |
| `train_model.py` | Script that trains the multi-ticker LSTM model |
| `Stock_Predictions_Model_v2.keras` | Pre-trained model — do not rename without updating `app_v2.py` |
| `requirements.txt` | Python dependencies needed to run/deploy the app |

## How it works

1. **Data** — pulls daily historical closing prices for a chosen ticker
   from Yahoo Finance (`yfinance`).
2. **Normalization** — instead of scaling by one global price range (which
   breaks down across tickers with very different price levels), each
   100-day input window is scaled by its **own** min/max. This lets the
   model learn the *shape* of price movement rather than an absolute price
   level, which is what makes it usable across many different stocks.
3. **Model** — a stacked LSTM (4 layers, with dropout) trained across ~30
   tickers spanning different sectors and price ranges, using the
   per-window scaling above.
4. **Prediction** — the app shows a historical backtest (predicted vs.
   actual on a held-out test split) plus a live next-day price prediction
   for whatever ticker you enter.

## Run it locally

```bash
git clone https://github.com/jennagozalii/Stock-Market-Predictor-v2.git
cd Stock-Market-Predictor-v2
pip install -r requirements.txt
streamlit run app_v2.py
```

Then open the URL Streamlit prints (usually `http://localhost:8501`).

## Retraining the model

`train_model.py` downloads ~14 years of daily data across ~30 tickers and
retrains the model from scratch using per-window normalization. This needs
real compute time and internet access to Yahoo Finance, so it's best run in
**Google Colab**

```bash
pip install yfinance tensorflow scikit-learn pandas numpy
python train_model.py
```

This produces a new `Stock_Predictions_Model_v2.keras` — swap it into the
repo to update the deployed app.

## Known limitations

- Trained on a single feature (closing price only) - no volume, technical
  indicators, or macro/news signals.
- Not a reliable trading signal on its own — markets are influenced by
  information (news, earnings, sentiment) the model never sees.
- Very new tickers (recent IPOs) or highly illiquid/volatile stocks may fall
  outside the price behavior the training basket covered, so predictions
  there should be trusted even less than usual.
- No walk-forward validation or hyperparameter tuning was performed — a
  single chronological 80/20 split per ticker was used.

## Tech stack

Python · TensorFlow / Keras · Streamlit · yfinance · pandas · NumPy · Matplotlib
