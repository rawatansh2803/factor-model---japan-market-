from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig


EPS = 1e-12


def _normalize(weights: pd.Series) -> pd.Series:
    weights = weights.fillna(0.0).clip(lower=0.0)
    total = weights.sum()
    if total <= 0:
        return pd.Series(1.0 / len(weights), index=weights.index)
    return weights / total


def _project_max_weight(weights: pd.Series, max_weight: float) -> pd.Series:
    weights = _normalize(weights)
    if max_weight >= 1.0:
        return weights

    weights = weights.copy()
    for _ in range(20):
        over = weights > max_weight + EPS
        if not over.any():
            break
        excess = (weights[over] - max_weight).sum()
        weights[over] = max_weight
        under = weights < max_weight - EPS
        if not under.any() or excess <= EPS:
            break
        weights.loc[under] += excess * weights.loc[under] / weights.loc[under].sum()
        weights = _normalize(weights)
    return _normalize(weights.clip(upper=max_weight))


def _allocate_with_caps(raw: pd.Series, cap: pd.Series, budget: float) -> pd.Series:
    raw = raw.fillna(0.0).clip(lower=0.0)
    cap = cap.fillna(0.0).clip(lower=0.0)
    alloc = pd.Series(0.0, index=raw.index)
    remaining = min(float(budget), float(cap.sum()))
    active = cap > EPS

    while remaining > EPS and active.any() and raw[active].sum() > EPS:
        proposal = remaining * raw[active] / raw[active].sum()
        room = cap[active] - alloc[active]
        step = np.minimum(proposal, room)
        alloc.loc[active] += step
        used = float(step.sum())
        remaining -= used
        active = (cap - alloc) > EPS
        if used <= EPS:
            break
    return alloc


def baseline_weights(date_df: pd.DataFrame, cfg: ResearchConfig) -> pd.Series:
    idx = date_df[cfg.ticker_col]
    if cfg.use_benchmark_weights and cfg.benchmark_weight_col in date_df.columns:
        bw = pd.Series(date_df[cfg.benchmark_weight_col].values, index=idx)
        if bw.notna().sum() > 0 and bw.fillna(0).sum() > 0:
            return _normalize(bw.fillna(0.0))
    if cfg.market_cap_col in date_df.columns:
        cap = pd.Series(date_df[cfg.market_cap_col].values, index=idx)
        if cap.notna().sum() > 0 and cap.fillna(0).sum() > 0:
            return _normalize(cap.fillna(0.0))
    return pd.Series(1.0 / len(date_df), index=idx)


def build_target_weights(date_df: pd.DataFrame, score_col: str, cfg: ResearchConfig) -> pd.Series:
    tickers = pd.Index(date_df[cfg.ticker_col])
    base = baseline_weights(date_df, cfg).reindex(tickers).fillna(0.0)
    scores = pd.Series(date_df[score_col].values, index=tickers).astype(float)

    if cfg.portfolio_mode == "top_quantile":
        ranks = scores.rank(pct=True, method="average")
        selected = ranks >= (1.0 - cfg.top_quantile)
        if selected.sum() == 0:
            return _project_max_weight(base, cfg.max_weight)
        w = pd.Series(0.0, index=tickers)
        raw = scores[selected] - scores[selected].min() + EPS
        w.loc[selected] = raw / raw.sum()
        return _project_max_weight(w, cfg.max_weight)

    ranks = scores.rank(pct=True, method="average")
    top = ranks >= (1.0 - cfg.top_quantile)
    bottom = ranks <= cfg.bottom_quantile

    if top.sum() == 0 or bottom.sum() == 0:
        return _project_max_weight(base, cfg.max_weight)

    up_raw = (scores[top] - scores[top].min() + EPS)
    down_raw = (scores[bottom].max() - scores[bottom] + EPS)

    up_cap = (cfg.max_weight - base[top]).clip(lower=0.0)
    down_cap = base[bottom].clip(lower=0.0)
    budget = min(cfg.active_budget, float(up_cap.sum()), float(down_cap.sum()))

    up_alloc = _allocate_with_caps(up_raw, up_cap, budget)
    down_alloc = _allocate_with_caps(down_raw, down_cap, budget)
    actual_budget = min(float(up_alloc.sum()), float(down_alloc.sum()))

    if actual_budget <= EPS:
        return _project_max_weight(base, cfg.max_weight)

    if up_alloc.sum() > actual_budget + EPS:
        up_alloc *= actual_budget / up_alloc.sum()
    if down_alloc.sum() > actual_budget + EPS:
        down_alloc *= actual_budget / down_alloc.sum()

    target = base.copy()
    target.loc[top] = target.loc[top] + up_alloc
    target.loc[bottom] = target.loc[bottom] - down_alloc
    target = target.clip(lower=0.0)
    target = _project_max_weight(target, cfg.max_weight)
    return target


def apply_turnover_constraint(
    target: pd.Series,
    previous: pd.Series | None,
    cfg: ResearchConfig,
) -> tuple[pd.Series, float]:
    target = target.fillna(0.0)
    if previous is None or previous.empty:
        target = _normalize(target)
        target = _project_max_weight(target, cfg.max_weight)
        turnover = 0.5 * float(target.abs().sum())
        return target, turnover

    all_tickers = target.index.union(previous.index)
    tgt = target.reindex(all_tickers).fillna(0.0)
    prev = previous.reindex(all_tickers).fillna(0.0)

    raw_turnover = 0.5 * float((tgt - prev).abs().sum())
    if raw_turnover > cfg.max_turnover and raw_turnover > EPS:
        blend = cfg.max_turnover / raw_turnover
        tgt = prev + blend * (tgt - prev)
    tgt = _normalize(tgt)
    tgt = _project_max_weight(tgt, cfg.max_weight)
    turnover = 0.5 * float((tgt - prev).abs().sum())
    return tgt, turnover


def realized_portfolio_return(date_df: pd.DataFrame, weights: pd.Series, cfg: ResearchConfig) -> dict:
    tickers = weights.index
    df = date_df.set_index(cfg.ticker_col).reindex(tickers)
    port_return = float((weights * df["forward_return"].fillna(0.0)).sum())
    benchmark_forward = float(df["benchmark_forward_return"].dropna().iloc[0])
    active_return = port_return - benchmark_forward
    return {
        "portfolio_return": port_return,
        "benchmark_return": benchmark_forward,
        "active_return": active_return,
    }
