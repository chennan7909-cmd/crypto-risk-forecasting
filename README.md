# AI-Driven Risk Detection and Price Forecasting in Cryptocurrency Markets

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classification-00897B?style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Baseline_Models-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Yahoo Finance](https://img.shields.io/badge/yfinance-Market_Data-6001D2?style=for-the-badge)

**Ontario Tech University · Winter 2026**  
*Nan Chen · Student ID: 101021912*

</div>

---

## Executive Summary

> A dual-task machine learning pipeline targeting Solana (SOL) using Bitcoin (BTC) as a leading cross-asset signal. The system simultaneously addresses **next-day price forecasting** (LSTM regression) and **tail-risk event detection** — identifying days on which SOL declines more than 5% (XGBoost binary classification). Four models are trained and benchmarked head-to-head, with walk-forward temporal validation throughout.

---

## Business Problem

Cryptocurrency markets exhibit extreme volatility, thin liquidity, and strong cross-asset contagion. BTC has been empirically observed to lead altcoin price movements by 1 to 3 days. This project exploits that market microstructure insight to answer two practical questions:

- Can tomorrow's SOL closing price be forecast reliably using BTC's lagged signals?
- Can days with SOL drawdowns exceeding 5% be identified in advance for risk management?

---

## Key Results at a Glance

### Regression (Price Forecasting)

| Asset | Model | RMSE (USD) | MAPE (%) |
|-------|-------|-----------|---------|
| BTC | **LSTM** (primary) | ~2,100 | ~2.8% |
| BTC | Linear Regression (baseline) | ~2,450 | ~3.3% |
| SOL | **LSTM** (primary) | ~4.2 | ~2.5% |
| SOL | Linear Regression (baseline) | ~5.8 | ~3.6% |

### Classification (Tail-Risk Detection)

| Asset | Model | F1-Score | ROC-AUC | Threshold |
|-------|-------|---------|--------|---------|
| BTC | **XGBoost** (primary) | ~0.41 | ~0.77 | 0.30 |
| BTC | Logistic Regression (baseline) | ~0.29 | ~0.65 | 0.30 |
| SOL | **XGBoost** (primary) | ~0.38 | ~0.74 | 0.30 |
| SOL | Logistic Regression (baseline) | ~0.26 | ~0.62 | 0.30 |

LSTM outperforms the linear baseline on both assets. XGBoost's AUC advantage over Logistic Regression holds across all decision thresholds, confirming that non-linear interactions among technical indicators and cross-asset signals provide genuine discriminating power.

---

## Architecture Overview

```
Yahoo Finance (yfinance)
    ↓  BTC-USD + SOL-USD daily OHLCV  (Jan 2021 – Mar 2026, ~1,900 days)
    ↓  load_and_preprocess()  — auto-detects yfinance metadata headers
    ↓  Standard technical indicators (MA7, MA21, RSI, BB, MACD, Volatility)
    ↓  engineer_cross_asset_features()
         BTC signals lagged [1, 3, 7 days] → appended to SOL feature matrix
         15 additional cross-asset predictors
    ↓
    ├── REGRESSION PIPELINE
    │     MinMaxScaler (features) + separate y_scaler (target)
    │     LSTM: 64 → Dropout(0.2) → 32 → Dropout(0.2) → Dense(16) → Dense(1)
    │     EarlyStopping (patience=5) | TimeSeriesSplit (5-fold)
    │     Inverse-transform → RMSE & MAPE in USD
    │
    └── CLASSIFICATION PIPELINE
          XGBoost (scale_pos_weight for 12-15% minority class)
          Decision threshold = 0.30  (prioritises recall over precision)
          TimeSeriesSplit (5-fold) | Metrics: F1-Score, ROC-AUC
```

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Data Acquisition | **yfinance** | Daily OHLCV bars, BTC + SOL, 2021–2026 |
| Feature Engineering | **NumPy / Pandas** | Technical indicators, cross-asset lags |
| Regression (primary) | **LSTM (TensorFlow/Keras)** | Non-linear temporal price forecasting |
| Regression (baseline) | **Linear Regression (sklearn)** | Naive benchmark |
| Classification (primary) | **XGBoost** | Tail-risk event detection |
| Classification (baseline) | **Logistic Regression (sklearn)** | Linear benchmark |
| Validation | **TimeSeriesSplit** | Walk-forward, no data leakage |
| Scaling | **MinMaxScaler** (separate for X and y) | Prevents LSTM convergence failure |

---

## Repository Structure

```
crypto-risk-forecasting/
├── Final_project_Nan_Chen.ipynb       # Full pipeline notebook
├── Nan_Chen_Final_Report.pdf          # Technical report
└── README.md
```

> **Dataset**: Fetched live from Yahoo Finance via `yfinance`. No static data file required — the notebook downloads BTC-USD and SOL-USD automatically on first run.

---

## Original Contributions

### 1. Cross-Asset Leading Indicator Framework
`engineer_cross_asset_features()` lags five BTC signals (Returns, Volatility, RSI, MACD\_Hist, MA\_Ratio) by 1, 3, and 7 days and appends them to SOL's feature matrix — yielding 15 additional predictors. The BTC-leads-SOL hypothesis is validated empirically via Pearson cross-correlation across ±14-day lags, confirming positive correlation peaking at BTC lag +1 to +3. This framework is not available in any open-source library.

### 2. Dual-Task Architecture
A single pipeline simultaneously solves price forecasting (regression) and crash detection (classification) across two assets, producing four head-to-head model comparisons within one coherent system.

### 3. y_scaler Bug-Fix (LSTM Convergence)
Identified and corrected an LSTM convergence failure caused by unscaled USD targets (range \$15,000–\$124,000). Implemented an independent `MinMaxScaler` for the regression target with a full inverse-transform pipeline to restore USD-denominated predictions. Documented as an original engineering contribution.

### 4. Robust CSV Parser
`load_and_preprocess()` auto-detects yfinance metadata header rows using the `'20xx'` date prefix, eliminating manual inspection across different yfinance API versions.

---

## Model Design Decisions

| Decision | Rationale |
|---------|-----------|
| **Walk-forward validation (TimeSeriesSplit)** | Random K-fold violates temporal ordering and causes data leakage in financial time-series |
| **Decision threshold = 0.30** | Applied uniformly to both classifiers; a missed crash (false negative) is costlier than a false alarm in risk management |
| **Separate y_scaler for LSTM** | Raw USD price range prevents gradient convergence; inverse-transform restores interpretable units |
| **EarlyStopping on training loss** | No separate validation split — maximises use of limited financial time-series data |
| **scale_pos_weight in XGBoost** | Risk events occur in only 12–15% of days; class imbalance addressed at the model level |
| **F1-Score over Accuracy** | Predicting 0 always yields ~87% accuracy with zero recall — accuracy is misleading on imbalanced data |

---

## Feature Set

**BTC (8 features):** Close, Volume, Volatility, MA7, RSI, BB\_Position, MACD\_Hist, MA\_Ratio

**SOL (23 features):** Same 8 base features + 15 cross-asset lagged BTC signals (3 lags × 5 BTC indicators)

---

## How to Run

1. **Clone the repo**
```bash
git clone https://github.com/chennan7909-cmd/crypto-risk-forecasting.git
```

2. **Install dependencies**
```bash
pip install yfinance tensorflow xgboost scikit-learn pandas numpy matplotlib
```

3. **Run the notebook**

Open `Final_project_Nan_Chen.ipynb` in Jupyter or Google Colab and run all cells. The notebook fetches market data automatically — no manual data download required.

> Note: LSTM training time varies by hardware. GPU runtime (Google Colab) is recommended for faster execution.

---

## Ethical Considerations

This project is strictly academic. Model outputs should not be interpreted as financial advice. Key limitations include:

- Training window (2021–2026) covers specific market regimes; the model may underperform in structurally different conditions such as a post-regulation environment or BTC ETF-driven decoupling from altcoins.
- The 5% drawdown threshold for risk events is domain-specific and affects class balance and calibration.
- Algorithmic trading tools disproportionately benefit technically literate, high-income participants — a recognised equity concern in fintech.

---

## Future Work

- [ ] Extend cross-asset framework to include Ethereum (ETH) as an additional leading signal
- [ ] Implement Temporal Fusion Transformer (TFT) for multi-horizon forecasting
- [ ] Add SHAP explainability to XGBoost risk detection for feature attribution
- [ ] Incorporate on-chain metrics (active addresses, transaction volume) as alternative data
- [ ] Backtest trading strategy based on combined regression + classification signals

---

## References

- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. *KDD 2016.*
- Hochreiter & Schmidhuber (1997). Long Short-Term Memory. *Neural Computation, 9(8).*
- Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR, 12.*
- Arık & Pfister (2021). TabNet: Attentive Interpretable Tabular Learning. *AAAI 2021.*

---

<div align="center">

*Built with TensorFlow · XGBoost · yfinance · Scikit-Learn · NumPy*  
*Ontario Tech University · Winter 2026*

</div>
