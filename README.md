# 🧠 IV Delta Forecasting with XGBoost and LSTM

This notebook predicts the **daily change in implied volatility (ΔIV)** for SPY (S\&P 500 ETF) options using two distinct ML approaches:

* 🔺 **Quantile Regression with XGBoost**
* 🔁 **Sequential Forecasting with LSTM (Neural Network)**

## 📦 How It Works

### 1. **Data Collection**

* SPY (price, volume) and VIX (volatility index) are downloaded using `yfinance`.
* IV is approximated using 5-day rolling standard deviation of SPY returns.
* Target: **ΔIV = IV(t+1) - IV(t)**

### 2. **Feature Engineering**

* Market features: SPY returns, VIX, volume changes
* Option-style metadata: random DTE, call/put indicator
* Lagged IV/delta\_IV values
* Prophet-derived seasonal features (`trend`, `weekly`)

### 3. **Quantile XGBoost**

* Trains separate models for 10th, 50th, 90th percentiles
* Visualizes median forecast + confidence band (10–90%)
* Metrics: MAE, MSE, R², Directional Accuracy

### 4. **LSTM Model**

* Uses 60-day sequences of engineered features
* Predicts ΔIV using a simple 1-layer LSTM
* Evaluates performance and plots predictions + residuals

---

## 📈 Example Output

| Model   | MAE    | R²      | Directional Accuracy |
| ------- | ------ | ------- | -------------------- |
| XGBoost | 0.0073 | -6.5655 | 42.4%                |
| LSTM    | 0.0029 | -4.4223 | 59.1%                |

---

## 🛠 Requirements

Install dependencies using:

```bash
pip install yfinance prophet xgboost scikit-learn keras tensorflow
```

---

## ✅ Future Improvements

* Replace rolling IV with real implied volatility (from options chain)
* Use earnings calendar and macroeconomic events
* Test more advanced LSTM/Transformer architectures
