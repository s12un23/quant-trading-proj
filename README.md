# Quantitative Trading Pipeline — S&P 500 Tech Stocks

(A simple project to get started.)

A complete end-to-end quantitative trading system built on S&P 500 technology
sector stocks (2015–2025), implementing alpha factor generation, evaluation,
ML modeling, and portfolio optimization.

## Project Structure

```
quant-trading/
├── data/              # Raw and processed data (not tracked by git)
├── notebooks/
│   ├── data_collection.ipynb   # Part 1: Data download & cleaning
│   ├── project_1.ipynb         # Part 2-5: Full pipeline
│   └── project_1_simple.ipynb  # A Simple version without Alpha101
├── src/
│   └── alphas.py               # Alpha101 factor implementations
└── results/                    # Output charts
```

## Pipeline Overview

### Part 1 — Data Collection

- Downloaded OHLCV data for ~500 S&P 500 stocks via yfinance (2015–2025)
- Stock universe filtered to Technology sector, liquidity Top 50 based on
  2015–2016 average dollar volume (to avoid look-ahead bias)
- Final universe: 50 stocks

### Part 2 — Alpha Factor Generation & Evaluation

- Implemented 53 alpha factors based on WorldQuant Alpha101 framework,
  plus custom momentum and volume factors
- Preprocessing: MAD winsorization + time-series z-score + cross-sectional z-score
- IC evaluation using 5-day forward returns (Spearman rank correlation)
- Factor selection: removed NaN factors, degenerate distributions,
  high-correlation duplicates, and weak factors (ICIR < 0.015)
- MVO factor combination: used rolling 250-day IC history to compute
  optimal factor weights via Mean-Variance Optimization

### Part 3 — ML Modeling

- Implemented Ridge Regression and Logistic Regression with expanding-window
  time-series cross-validation (20 folds, starting from 50% train split)
- Both models showed near-zero out-of-sample IC, confirming that simple linear
  models struggle to capture stable factor-return relationships across different
  market regimes (2020 COVID, 2022 rate hikes, 2023–2024 AI rally)

### Part 4 — Portfolio Optimization

- Two-layer strategy: equal-weight factor score for stock selection (Top 15)
  - MVO for position sizing
- Rebalancing every 10 trading days

### Part 5 — Performance Evaluation (2018–2025)

| Strategy                        | Ann. Return | Sharpe | Max DD  |
| ------------------------------- | ----------- | ------ | ------- |
| MVO Serial Strategy             | +24.46%     | 0.714  | -42.01% |
| Equal Weight Benchmark (Top 15) | +26.69%     | 0.863  | -37.56% |
| Market Equal Weight (45 stocks) | +22.20%     | 0.750  | -34.41% |

Both the MVO strategy and equal-weight benchmark outperformed the market
equal-weight baseline.

## Known Limitations

- Sector classification uses current GICS labels (not historical),
  which may differ from classifications at the start of the backtest period
- No transaction costs or slippage modeled
- Survivorship bias: universe constructed from current S&P 500 constituents

## Tech Stack

Python · pandas · numpy · yfinance · scikit-learn · scipy · matplotlib · seaborn

## References

- Kakushadze, Z. (2016). _101 Formulaic Alphas_
- Lopez de Prado, M. (2018). _Advances in Financial Machine Learning_
