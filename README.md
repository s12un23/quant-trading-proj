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

**Key finding:** MVO-weighted factor combination (ICIR = 0.1213)
underperformed equal-weight combination (ICIR = 0.1770), consistent
with the "estimation error dominates optimization" effect in high-dimensional
settings with limited samples.

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

---

# 量化交易完整流程 — 标普500科技股

基于标普500科技板块股票（2015–2025年）构建的端到端量化交易系统，涵盖alpha因子生成、评估、机器学习建模与组合优化全流程。

## 项目结构

```
quant-trading/
├── data/              # 原始及处理后的数据（不纳入git追踪）
├── notebooks/
│   ├── data_collection.ipynb   # Part 1：数据下载与清洗
│   ├── project_1.ipynb         # Part 2-5：完整流程
│   └── project_1_simple.ipynb  # 一个简易版本（若干经典因子，不含Alpha101）
├── src/
│   └── alphas.py               # Alpha101因子实现
└── results/                    # 输出图表

```

## 流程说明

### Part 1 — 数据收集

- 通过yfinance下载约500只标普500成分股的OHLCV数据（2015–2025年）
- 股票池筛选：科技行业 + 以2015–2016年平均日成交额为标准取流动性Top50（使用历史截止日期避免前视偏差）
- 最终股票池：45只股票

### Part 2 — Alpha因子生成与评估

- 基于WorldQuant Alpha101框架实现53个alpha因子，另加自定义动量与量价因子
- 预处理：MAD去极值 + 时序z-score + 截面z-score标准化
- 使用5日远期收益率进行IC评估（Spearman秩相关系数）
- 因子筛选：剔除NaN因子、退化分布因子、高相关重复因子及弱因子（ICIR < 0.015）
- MVO因子组合：用滚动250日IC历史，通过均值-方差优化求最优因子权重

**核心发现：** MVO加权因子组合（ICIR = 0.1213）的表现弱于等权组合（ICIR = 0.1770），
符合高维小样本场景下"估计误差主导优化结果"的理论预期。

### Part 3 — 机器学习建模

- 使用扩展窗口时间序列交叉验证（20个fold，最小训练集为前50%数据）训练Ridge回归与逻辑回归
- 两个模型的样本外IC均接近于零，说明简单线性模型难以在不同市场环境下（2020年疫情、2022年加息、2023–2024年AI行情）捕捉稳定的因子-收益关系

### Part 4 — 组合优化

- 两层策略：等权因子得分选股（Top 15）+ MVO仓位优化
- 每10个交易日再平衡一次

### Part 5 — 绩效评估（2018–2025年）

| 策略               | 年化收益 | Sharpe | 最大回撤 |
| ------------------ | -------- | ------ | -------- |
| MVO串联策略        | +24.46%  | 0.714  | -42.01%  |
| 等权基准（Top 15） | +26.69%  | 0.863  | -37.56%  |
| 市场等权（45只）   | +22.20%  | 0.750  | -34.41%  |

MVO策略与等权基准均跑赢了全仓持有45只股票的市场基准。

## 已知局限性

- 行业分类使用当前GICS标签，而非回测起点时的历史分类，可能存在偏差
- 未计入交易成本与滑点
- 存在幸存者偏差：股票池基于当前标普500成分股构建

## 技术栈

Python · pandas · numpy · yfinance · scikit-learn · scipy · matplotlib · seaborn

## 参考文献

- Kakushadze, Z. (2016). _101 Formulaic Alphas_
- Lopez de Prado, M. (2018). _Advances in Financial Machine Learning_
