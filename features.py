from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig


VALUE_FEATURES = [
    "book_to_market",
    "earnings_to_price",
    "sales_to_price",
    "cashflow_to_price",
]

MOMENTUM_FEATURES = [
    "mom_1m",
    "mom_3m",
    "mom_6m",
    "mom_12_1m",
    "vol_12m",
]


def _forward_compound(series: pd.Series, horizon: int) -> pd.Series:
    shifted = series.shift(-1)
    reversed_compound = shifted.iloc[::-1].rolling(horizon).apply(
        lambda x: np.prod(1 + x) - 1,
        raw=True,
    )
    return reversed_compound.iloc[::-1]


def _backward_compound(series: pd.Series, window: int, shift: int = 1) -> pd.Series:
    lagged = series.shift(shift)
    return lagged.rolling(window).apply(lambda x: np.prod(1 + x) - 1, raw=True)


def _winsorize(s: pd.Series, lower: float, upper: float) -> pd.Series:
    if s.notna().sum() == 0:
        return s
    lo = s.quantile(lower)
    hi = s.quantile(upper)
    return s.clip(lo, hi)


def _cross_sectional_zscore(s: pd.Series) -> pd.Series:
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return s * 0.0
    return (s - s.mean()) / std


def build_feature_panel(panel: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    panel = panel.sort_values([cfg.ticker_col, cfg.date_col]).copy()

    # Value features.
    market_cap = panel[cfg.market_cap_col].replace(0, np.nan)
    panel["book_to_market"] = panel.get(cfg.book_equity_col, np.nan) / market_cap
    panel["earnings_to_price"] = panel.get(cfg.net_income_col, np.nan) / market_cap
    panel["sales_to_price"] = panel.get(cfg.sales_col, np.nan) / market_cap
    panel["cashflow_to_price"] = panel.get(cfg.cashflow_col, np.nan) / market_cap

    # Group-wise momentum / risk features from past returns only.
    def by_ticker_transform(func):
        return panel.groupby(cfg.ticker_col, group_keys=False)[cfg.return_col].apply(func)

    panel["mom_1m"] = panel.groupby(cfg.ticker_col)[cfg.return_col].shift(1)
    panel["mom_3m"] = by_ticker_transform(lambda s: _backward_compound(s, 3, shift=1))
    panel["mom_6m"] = by_ticker_transform(lambda s: _backward_compound(s, 6, shift=1))
    panel["mom_12_1m"] = by_ticker_transform(lambda s: _backward_compound(s, 11, shift=2))
    panel["vol_12m"] = panel.groupby(cfg.ticker_col)[cfg.return_col].transform(
        lambda s: s.shift(1).rolling(12).std()
    )

    # Forward returns and benchmark-relative prediction target.
    panel["forward_return"] = panel.groupby(cfg.ticker_col)[cfg.return_col].transform(
        lambda s: _forward_compound(s, cfg.horizon_months)
    )
    benchmark = panel[[cfg.date_col, cfg.benchmark_return_col]].drop_duplicates().sort_values(cfg.date_col)
    benchmark["benchmark_forward_return"] = _forward_compound(
        benchmark[cfg.benchmark_return_col], cfg.horizon_months
    )
    panel = panel.merge(
        benchmark[[cfg.date_col, "benchmark_forward_return"]],
        on=cfg.date_col,
        how="left",
    )
    panel["target_active_return"] = (
        panel["forward_return"] - panel["benchmark_forward_return"]
    )

    feature_cols = VALUE_FEATURES + MOMENTUM_FEATURES
    for col in feature_cols:
        panel[col] = panel.groupby(cfg.date_col)[col].transform(
            lambda s: _winsorize(s, cfg.winsorize_lower, cfg.winsorize_upper)
        )
        if cfg.standardize_cross_sectionally:
            panel[col] = panel.groupby(cfg.date_col)[col].transform(_cross_sectional_zscore)

    panel["factor_sort_score"] = panel[VALUE_FEATURES + MOMENTUM_FEATURES].mean(axis=1)
    if cfg.standardize_cross_sectionally:
        panel["factor_sort_score"] = panel.groupby(cfg.date_col)["factor_sort_score"].transform(
            _cross_sectional_zscore
        )

    panel = panel.sort_values([cfg.date_col, cfg.ticker_col]).reset_index(drop=True)
    return panel


def get_feature_columns() -> list[str]:
    return VALUE_FEATURES + MOMENTUM_FEATURES
