from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import ResearchConfig


REQUIRED_PRICE_COLUMNS = {"date", "ticker"}
REQUIRED_FUNDAMENTAL_COLUMNS = {"effective_date", "ticker"}
REQUIRED_BENCHMARK_COLUMNS = {"date"}


def _month_end(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series) + pd.offsets.MonthEnd(0)


def _ensure_return(prices: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    prices = prices.sort_values([cfg.ticker_col, cfg.date_col]).copy()
    if cfg.return_col not in prices.columns:
        if cfg.price_col not in prices.columns:
            raise ValueError(
                f"prices file must contain either '{cfg.return_col}' or '{cfg.price_col}'."
            )
        prices[cfg.return_col] = (
            prices.groupby(cfg.ticker_col)[cfg.price_col].pct_change()
        )
    return prices


def _ensure_market_cap(prices: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    prices = prices.copy()
    if cfg.market_cap_col in prices.columns:
        return prices
    if {"shares_outstanding", cfg.price_col}.issubset(prices.columns):
        prices[cfg.market_cap_col] = prices["shares_outstanding"] * prices[cfg.price_col]
        return prices
    prices[cfg.market_cap_col] = np.nan
    return prices


def _ensure_liquidity_metric(prices: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    prices = prices.copy()
    if "dollar_volume" not in prices.columns:
        if cfg.volume_col in prices.columns and cfg.price_col in prices.columns:
            prices["dollar_volume"] = prices[cfg.volume_col] * prices[cfg.price_col]
        else:
            prices["dollar_volume"] = np.nan
    return prices


def _ensure_benchmark_return(benchmark: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    benchmark = benchmark.sort_values(cfg.date_col).copy()
    if cfg.benchmark_return_col not in benchmark.columns:
        if cfg.benchmark_level_col not in benchmark.columns:
            raise ValueError(
                f"benchmark file must contain either '{cfg.benchmark_return_col}' or '{cfg.benchmark_level_col}'."
            )
        benchmark[cfg.benchmark_return_col] = benchmark[cfg.benchmark_level_col].pct_change()
    return benchmark


def _asof_merge_fundamentals(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    cfg: ResearchConfig,
) -> pd.DataFrame:
    fundamentals = fundamentals.copy()
    fundamentals[cfg.fundamental_date_col] = _month_end(fundamentals[cfg.fundamental_date_col])
    fundamentals["available_date"] = fundamentals[cfg.fundamental_date_col] + pd.offsets.MonthEnd(
        cfg.fundamental_lag_months
    )

    prices = prices.sort_values([cfg.ticker_col, cfg.date_col]).copy()
    fundamentals = fundamentals.sort_values([cfg.ticker_col, "available_date"]).copy()

    merged_parts = []
    all_tickers = sorted(set(prices[cfg.ticker_col]).intersection(set(fundamentals[cfg.ticker_col])))
    for ticker in all_tickers:
        p = prices.loc[prices[cfg.ticker_col] == ticker].sort_values(cfg.date_col).copy()
        f = fundamentals.loc[fundamentals[cfg.ticker_col] == ticker].sort_values("available_date").copy()
        merged = pd.merge_asof(
            p,
            f,
            left_on=cfg.date_col,
            right_on="available_date",
            direction="backward",
            suffixes=("", "_fund"),
        )
        merged_parts.append(merged)

    # Preserve tickers with no available fundamentals.
    missing_fund_tickers = set(prices[cfg.ticker_col]) - set(all_tickers)
    if missing_fund_tickers:
        merged_parts.append(prices.loc[prices[cfg.ticker_col].isin(missing_fund_tickers)].copy())

    merged_all = pd.concat(merged_parts, ignore_index=True, sort=False)
    return merged_all


def _merge_benchmark_weights(
    panel: pd.DataFrame,
    benchmark_weights: Optional[pd.DataFrame],
    cfg: ResearchConfig,
) -> pd.DataFrame:
    panel = panel.copy()
    if benchmark_weights is None or not cfg.use_benchmark_weights:
        panel[cfg.benchmark_weight_col] = np.nan
        return panel

    benchmark_weights = benchmark_weights.copy()
    benchmark_weights[cfg.date_col] = _month_end(benchmark_weights[cfg.date_col])
    panel = panel.merge(
        benchmark_weights[[cfg.date_col, cfg.ticker_col, cfg.benchmark_weight_col]],
        on=[cfg.date_col, cfg.ticker_col],
        how="left",
    )
    return panel


def load_research_panel(cfg: ResearchConfig) -> pd.DataFrame:
    prices = pd.read_csv(cfg.prices_path)
    fundamentals = pd.read_csv(cfg.fundamentals_path)
    benchmark = pd.read_csv(cfg.benchmark_path)
    benchmark_weights = None
    if cfg.benchmark_weights_path and Path(cfg.benchmark_weights_path).exists():
        benchmark_weights = pd.read_csv(cfg.benchmark_weights_path)

    missing_price_cols = REQUIRED_PRICE_COLUMNS.difference(prices.columns)
    missing_fund_cols = REQUIRED_FUNDAMENTAL_COLUMNS.difference(fundamentals.columns)
    missing_bm_cols = REQUIRED_BENCHMARK_COLUMNS.difference(benchmark.columns)
    if missing_price_cols:
        raise ValueError(f"prices is missing required columns: {sorted(missing_price_cols)}")
    if missing_fund_cols:
        raise ValueError(f"fundamentals is missing required columns: {sorted(missing_fund_cols)}")
    if missing_bm_cols:
        raise ValueError(f"benchmark is missing required columns: {sorted(missing_bm_cols)}")

    prices[cfg.date_col] = _month_end(prices[cfg.date_col])
    fundamentals[cfg.fundamental_date_col] = pd.to_datetime(fundamentals[cfg.fundamental_date_col])
    benchmark[cfg.date_col] = _month_end(benchmark[cfg.date_col])

    prices = _ensure_return(prices, cfg)
    prices = _ensure_market_cap(prices, cfg)
    prices = _ensure_liquidity_metric(prices, cfg)
    benchmark = _ensure_benchmark_return(benchmark, cfg)

    panel = _asof_merge_fundamentals(prices, fundamentals, cfg)
    panel = panel.merge(
        benchmark[[cfg.date_col, cfg.benchmark_return_col]],
        on=cfg.date_col,
        how="left",
    )
    panel = _merge_benchmark_weights(panel, benchmark_weights, cfg)

    if cfg.min_market_cap > 0 and cfg.market_cap_col in panel.columns:
        panel = panel[panel[cfg.market_cap_col].fillna(0) >= cfg.min_market_cap].copy()

    if "dollar_volume" in panel.columns and not panel["dollar_volume"].isna().all():
        q = panel.groupby(cfg.date_col)["dollar_volume"].transform(
            lambda s: s.quantile(cfg.liquidity_quantile)
        )
        panel = panel[panel["dollar_volume"].fillna(-np.inf) >= q].copy()

    counts = panel.groupby(cfg.date_col)[cfg.ticker_col].transform("count")
    panel = panel[counts >= cfg.min_obs_per_date].copy()

    panel = panel.sort_values([cfg.date_col, cfg.ticker_col]).reset_index(drop=True)
    return panel


def save_panel_snapshot(panel: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
