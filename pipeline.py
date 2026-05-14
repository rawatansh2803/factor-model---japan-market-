from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from .config import ResearchConfig
from .data import load_research_panel, save_panel_snapshot
from .evaluation import summarize_backtest, summarize_predictions
from .features import get_feature_columns, build_feature_panel
from .models import tune_and_fit_model
from .portfolio import (
    apply_turnover_constraint,
    build_target_weights,
    realized_portfolio_return,
)


def walk_forward_splits(dates: list[pd.Timestamp], cfg: ResearchConfig):
    start = cfg.train_months
    while start + cfg.validation_months + cfg.test_months <= len(dates):
        train_start = max(0, start - cfg.train_months) if cfg.rolling_training_window else 0
        train_dates = dates[train_start:start]
        valid_dates = dates[start : start + cfg.validation_months]
        test_dates = dates[
            start + cfg.validation_months : start + cfg.validation_months + cfg.test_months
        ]
        yield train_dates, valid_dates, test_dates
        start += cfg.test_months


def _prepare_model_frame(
    panel: pd.DataFrame,
    feature_cols: list[str],
    cfg: ResearchConfig,
) -> pd.DataFrame:
    cols = [
        cfg.date_col,
        cfg.ticker_col,
        "target_active_return",
        "forward_return",
        "benchmark_forward_return",
        "factor_sort_score",
        cfg.market_cap_col,
        cfg.benchmark_weight_col,
    ] + feature_cols
    available = [c for c in cols if c in panel.columns]
    frame = panel[available].copy()
    frame = frame.dropna(subset=["target_active_return"])
    return frame


def _fit_predict_factor_sort(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> np.ndarray:
    cal = LinearRegression()
    calibration_df = pd.concat([train_df, valid_df], axis=0)
    x = calibration_df[["factor_sort_score"]].fillna(0.0)
    y = calibration_df["target_active_return"]
    cal.fit(x, y)
    return cal.predict(test_df[["factor_sort_score"]].fillna(0.0))


def run_pipeline(cfg: ResearchConfig) -> dict[str, pd.DataFrame]:
    panel = load_research_panel(cfg)
    feature_panel = build_feature_panel(panel, cfg)
    feature_cols = get_feature_columns()
    model_frame = _prepare_model_frame(feature_panel, feature_cols, cfg)
    dates = sorted(pd.to_datetime(model_frame[cfg.date_col].unique()))

    predictions = []
    portfolio_rows = []
    previous_weights: dict[str, pd.Series] = {m: pd.Series(dtype=float) for m in cfg.models}
    split_summaries = []

    for split_id, (train_dates, valid_dates, test_dates) in enumerate(walk_forward_splits(dates, cfg), start=1):
        train_df = model_frame[model_frame[cfg.date_col].isin(train_dates)].copy()
        valid_df = model_frame[model_frame[cfg.date_col].isin(valid_dates)].copy()
        test_df = model_frame[model_frame[cfg.date_col].isin(test_dates)].copy()

        if train_df.empty or valid_df.empty or test_df.empty:
            continue

        x_train = train_df[feature_cols]
        y_train = train_df["target_active_return"]
        x_valid = valid_df[feature_cols]
        y_valid = valid_df["target_active_return"]

        split_summaries.append(
            {
                "split_id": split_id,
                "train_start": str(min(train_dates)),
                "train_end": str(max(train_dates)),
                "valid_start": str(min(valid_dates)),
                "valid_end": str(max(valid_dates)),
                "test_start": str(min(test_dates)),
                "test_end": str(max(test_dates)),
                "n_train": int(len(train_df)),
                "n_valid": int(len(valid_df)),
                "n_test": int(len(test_df)),
            }
        )

        fitted_models = {}
        for model_name in cfg.models:
            if model_name == "factor_sort":
                fitted_models[model_name] = None
                continue
            fitted_models[model_name] = tune_and_fit_model(
                model_name=model_name,
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
                valid_dates=valid_df[cfg.date_col],
                cfg=cfg,
            )

        for model_name in cfg.models:
            test_copy = test_df.copy()
            if model_name == "factor_sort":
                test_copy["prediction"] = _fit_predict_factor_sort(train_df, valid_df, test_copy)
                validation_score = np.nan
                params = {"calibration": "univariate_linear_on_factor_sort_score"}
            else:
                result = fitted_models[model_name]
                test_copy["prediction"] = result.estimator.predict(test_copy[feature_cols])
                validation_score = result.validation_score
                params = result.best_params

            test_copy["model"] = model_name
            test_copy["split_id"] = split_id
            test_copy["validation_score"] = validation_score
            test_copy["best_params"] = json.dumps(params)
            predictions.append(test_copy[[
                cfg.date_col,
                cfg.ticker_col,
                "model",
                "split_id",
                "target_active_return",
                "forward_return",
                "benchmark_forward_return",
                "prediction",
                "validation_score",
                "best_params",
            ]])

            for date, date_df in test_copy.groupby(cfg.date_col):
                target_weights = build_target_weights(date_df, "prediction", cfg)
                final_weights, turnover = apply_turnover_constraint(
                    target_weights,
                    previous_weights.get(model_name),
                    cfg,
                )
                previous_weights[model_name] = final_weights
                realized = realized_portfolio_return(date_df, final_weights, cfg)
                cost = turnover * cfg.transaction_cost_bps / 10000.0
                portfolio_rows.append(
                    {
                        "date": pd.to_datetime(date),
                        "model": model_name,
                        "split_id": split_id,
                        "turnover": turnover,
                        "transaction_cost": cost,
                        **realized,
                        "net_active_return": realized["active_return"] - cost,
                    }
                )

    if not predictions:
        raise RuntimeError(
            "No walk-forward splits were generated. Reduce train/validation/test windows or add more data."
        )

    predictions_df = pd.concat(predictions, ignore_index=True).sort_values(["model", cfg.date_col, cfg.ticker_col])
    backtest_df = pd.DataFrame(portfolio_rows).sort_values(["model", "date"])
    prediction_summary = summarize_predictions(predictions_df.rename(columns={cfg.date_col: "date"}))
    backtest_summary = summarize_backtest(backtest_df, cfg)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(cfg.results_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg.save(output_dir / "config.json")
    save_panel_snapshot(feature_panel, output_dir / "feature_panel_snapshot.csv")
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    backtest_df.to_csv(output_dir / "backtest_timeseries.csv", index=False)
    prediction_summary.to_csv(output_dir / "prediction_summary.csv", index=False)
    backtest_summary.to_csv(output_dir / "backtest_summary.csv", index=False)
    pd.DataFrame(split_summaries).to_csv(output_dir / "walk_forward_splits.csv", index=False)

    return {
        "feature_panel": feature_panel,
        "predictions": predictions_df,
        "backtest": backtest_df,
        "prediction_summary": prediction_summary,
        "backtest_summary": backtest_summary,
        "output_dir": pd.DataFrame({"path": [str(output_dir)]}),
    }
