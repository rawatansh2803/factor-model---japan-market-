# Japanese ML Factor Model for Benchmark-Relative Long-Only Investing

This repository implements a thesis-oriented empirical asset pricing pipeline for **Japanese equities** with the exact design principles in your proposal:

- **Prediction target:** one-period-ahead **stock return relative to a benchmark**
- **Market:** intended for **liquid Japanese cash equities**
- **Features:** deliberately constrained to **Value + Momentum**
- **Models:** classical linear baseline plus **non-linear ML models**
- **Validation:** strict **walk-forward** training / validation / test splits
- **Portfolio implementation:** **long-only**, benchmark-aware, turnover-constrained
- **Evaluation:** predictive accuracy, **Information Ratio**, drawdown, turnover, and transaction-cost-adjusted returns

## What is implemented

### 1) Benchmark-relative target
For each stock `i` at rebalance date `t`, the pipeline estimates:

`target_active_return_{i,t+1} = stock_forward_return_{i,t+1} - benchmark_forward_return_{t+1}`

This matches your thesis objective more closely than traditional excess returns over the risk-free rate.

### 2) Factor space restricted to Value + Momentum
The model only uses a core factor set:

**Value**
- book-to-market
- earnings-to-price
- sales-to-price
- cash-flow-to-price

**Momentum / trend**
- 1-month momentum
- 3-month momentum
- 6-month momentum
- 12-1 momentum
- 12-month volatility

This avoids a broad “factor zoo” and keeps the design aligned with your stated research gap.

### 3) Baselines and ML models
Implemented models:
- `factor_sort`: classical factor-composite baseline
- `linear`: OLS baseline
- `elastic_net`: penalized linear model
- `random_forest`: non-linear tree ensemble
- `gradient_boosting`: non-linear boosted tree model
- `mlp`: feed-forward neural network baseline

### 4) Walk-forward validation
The pipeline **never uses random cross-validation**. It uses:
- training window
- chronological validation window
- forward out-of-sample test window

This is appropriate for financial time series and avoids leakage.

### 5) Implementable long-only portfolio construction
The backtest does **not** assume a frictionless long-short portfolio.

Instead it builds a **benchmark-relative long-only tilt portfolio**:
- start from benchmark weights if available, otherwise market-cap weights
- overweight top-ranked names
- underweight bottom-ranked names
- enforce **long-only** and **max position size**
- constrain **turnover**
- subtract **transaction costs**

This makes the research output much closer to institutional implementation.

---

## Suggested Japanese market use

For your actual thesis dataset, I recommend:

- **Universe:** TOPIX constituents, TOPIX 500, or TSE Prime with liquidity screens
- **Benchmark:** TOPIX or TOPIX 500
- **Frequency:** monthly
- **Liquidity filter:** top liquid names by traded value, plus market-cap minimum
- **Fundamental lag:** at least 3 months after reported accounting dates
- **Transaction costs:** calibrate using a realistic bps assumption for Japanese cash equities

The current code already supports these assumptions through the config.

---

## Data contract

Provide CSV files with the following structure.

### `prices.csv`
Required columns:
- `date`
- `ticker`

Recommended columns:
- `adj_close`
- `return`  ← if omitted, computed from `adj_close`
- `volume`
- `market_cap`
- `shares_outstanding` ← optional fallback for market cap calculation

### `fundamentals.csv`
Required columns:
- `effective_date`
- `ticker`

Recommended columns:
- `book_equity`
- `net_income`
- `sales`
- `operating_cash_flow`

### `benchmark.csv`
Required columns:
- `date`

Need one of:
- `benchmark_return`, or
- `benchmark_level`

### `benchmark_weights.csv` (optional but preferred)
- `date`
- `ticker`
- `benchmark_weight`

If benchmark weights are missing, the pipeline falls back to market-cap weights.

---

## Repository structure

```text
japan_factor_model/
├── README.md
├── THESIS_IMPLEMENTATION_NOTES.md
├── requirements.txt
├── generate_sample_data.py
├── sample_config.json
├── sample_quick_config.json
└── src/
    └── japan_factor_model/
        ├── __init__.py
        ├── cli.py
        ├── config.py
        ├── data.py
        ├── evaluation.py
        ├── features.py
        ├── models.py
        ├── pipeline.py
        └── portfolio.py
```

---

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Generate demo data

```bash
python generate_sample_data.py
```

### 3) Run the full pipeline

```bash
PYTHONPATH=src python -m japan_factor_model.cli --config sample_config.json
```

### 4) Or run a faster demo configuration

```bash
PYTHONPATH=src python -m japan_factor_model.cli --config sample_quick_config.json
```

The run will create a timestamped folder in `results/` containing:
- `config.json`
- `feature_panel_snapshot.csv`
- `predictions.csv`
- `prediction_summary.csv`
- `backtest_timeseries.csv`
- `backtest_summary.csv`
- `walk_forward_splits.csv`

---

## Example configuration

A sample config file is included in `sample_config.json`.

You can run with:

```bash
PYTHONPATH=src python -m japan_factor_model.cli --config sample_config.json
```

Key parameters:
- `horizon_months`: forecast horizon
- `train_months`, `validation_months`, `test_months`
- `active_budget`
- `max_weight`
- `max_turnover`
- `transaction_cost_bps`
- `liquidity_quantile`
- `use_benchmark_weights`

---

## Important thesis notes

### A) Japanese data quality matters more than model choice
The code is ready, but empirical credibility will depend on:
- point-in-time benchmark constituents
- point-in-time benchmark weights
- accounting availability dates
- delisting returns / dead firms
- split/dividend adjusted prices
- survivorship-bias-free universe construction

### B) For quarterly or annual horizons
The code supports `horizon_months > 1`, but for clean non-overlapping evaluation, it is best to set:

- `test_months = horizon_months`

That keeps the realized target horizon aligned with the test window.

### C) This implementation is intentionally practical
Your cited literature discusses end-to-end cost-aware learning. This code implements a pragmatic thesis-ready version:
- benchmark-relative stock-level prediction
- realistic long-only portfolio construction
- explicit turnover and transaction-cost penalties

If you want, the next extension would be a **cost-aware optimizer** or **end-to-end learning objective**.

---

## Outputs and thesis interpretation

### Predictive outputs
`prediction_summary.csv` reports:
- mean rank IC
- pooled out-of-sample R²
- number of observations
- number of dates

### Economic outputs
`backtest_summary.csv` reports:
- annualized active return (gross and net)
- tracking error (gross and net)
- **Information Ratio** (gross and net)
- maximum drawdown
- average turnover
- annualized turnover

This directly supports the thesis claim:

> Do non-linear ML models outperform traditional factor models after imposing realistic, long-only implementation constraints in Japanese equities?

---

## Recommended next extensions for a PhD-grade version

1. **Industry-neutral standardization**
2. **Delisting return handling**
3. **TOPIX constituent history ingestion**
4. **Sector constraints**
5. **Tracking error optimization using a covariance model**
6. **SHAP / feature importance diagnostics**
7. **Reality-check / SPA tests for model comparison**
8. **Subsample analysis**
   - Abenomics
   - COVID period
   - inflation / BoJ regime changes
9. **Capacity and market impact stress tests**
10. **End-to-end implementable efficient frontier extension**

---

## Thesis mapping

This code maps cleanly to thesis chapters:

- **Problem definition:** benchmark-relative cross-sectional return prediction
- **Literature review:** linear vs non-linear asset pricing models
- **Research gap:** practical, long-only ML factor investing in a liquid market
- **Methodology:** factor baseline + ML + walk-forward validation
- **Evaluation:** IR, drawdown, turnover-adjusted net returns

---

## If you want the next step

I can also help you with any of these immediately:

1. adapt the code to **actual Japanese vendor data**
2. add a **TOPIX/TOPIX 500 research universe builder**
3. convert this into a **full empirical thesis chapter**
4. add **plots and report generation**
5. add **XGBoost / LightGBM / CatBoost** versions
6. add a **cvxpy benchmark-relative optimizer**
