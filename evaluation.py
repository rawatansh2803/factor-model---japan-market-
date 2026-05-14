from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ResearchConfig
from .models import mean_rank_ic, pooled_r2


def max_drawdown(return_series: pd.Series) -> float:
    wealth = (1.0 + return_series.fillna(0.0)).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1.0
    return float(drawdown.min())


def summarize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for model_name, grp in predictions.groupby("model"):
        y_true = grp["target_active_return"].to_numpy()
        y_pred = grp["prediction"].to_numpy()
        summaries.append(
            {
                "model": model_name,
                "mean_rank_ic": mean_rank_ic(y_true, y_pred, grp["date"]),
                "pooled_r2": pooled_r2(y_true, y_pred),
                "n_obs": int(len(grp)),
                "n_dates": int(grp["date"].nunique()),
            }
        )
    return pd.DataFrame(summaries).sort_values("mean_rank_ic", ascending=False)


def summarize_backtest(backtest: pd.DataFrame, cfg: ResearchConfig) -> pd.DataFrame:
    ann = cfg.resolved_annualisation()
    summaries = []
    for model_name, grp in backtest.groupby("model"):
        grp = grp.sort_values("date")
        gross = grp["active_return"]
        net = grp["net_active_return"]
        te_gross = gross.std(ddof=0)
        te_net = net.std(ddof=0)
        ann_active_gross = gross.mean() * ann
        ann_active_net = net.mean() * ann
        ann_te_gross = te_gross * np.sqrt(ann)
        ann_te_net = te_net * np.sqrt(ann)
        summaries.append(
            {
                "model": model_name,
                "annual_active_return_gross": ann_active_gross,
                "annual_active_return_net": ann_active_net,
                "tracking_error_gross": ann_te_gross,
                "tracking_error_net": ann_te_net,
                "information_ratio_gross": ann_active_gross / ann_te_gross if ann_te_gross > 0 else np.nan,
                "information_ratio_net": ann_active_net / ann_te_net if ann_te_net > 0 else np.nan,
                "max_drawdown_net": max_drawdown(net),
                "avg_turnover": grp["turnover"].mean(),
                "annual_turnover": grp["turnover"].mean() * ann,
                "n_periods": int(len(grp)),
            }
        )
    return pd.DataFrame(summaries).sort_values("information_ratio_net", ascending=False)
