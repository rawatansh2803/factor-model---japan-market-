from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ResearchConfig


@dataclass
class TunedModelResult:
    model_name: str
    estimator: Any
    best_params: dict[str, Any]
    validation_score: float


MODEL_GRIDS: dict[str, dict[str, list[Any]]] = {
    "linear": {},
    "elastic_net": {
        "model__alpha": [0.001, 0.01, 0.10],
        "model__l1_ratio": [0.2, 0.5, 0.8],
    },
    "random_forest": {
        "model__n_estimators": [200],
        "model__max_depth": [4, 8],
        "model__min_samples_leaf": [20, 50],
    },
    "gradient_boosting": {
        "model__learning_rate": [0.03, 0.07],
        "model__max_depth": [3, 6],
        "model__min_samples_leaf": [20, 50],
    },
    "mlp": {
        "model__hidden_layer_sizes": [(32, 16), (64, 32)],
        "model__alpha": [0.0001, 0.001],
    },
}


def build_model(model_name: str, cfg: ResearchConfig):
    if model_name == "linear":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )
    if model_name == "elastic_net":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    ElasticNet(max_iter=10000, random_state=cfg.random_state),
                ),
            ]
        )
    if model_name == "random_forest":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_leaf=20,
                        random_state=cfg.random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    if model_name == "gradient_boosting":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_depth=4,
                        min_samples_leaf=20,
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )
    if model_name == "mlp":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        alpha=0.0001,
                        max_iter=600,
                        early_stopping=True,
                        random_state=cfg.random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model: {model_name}")


def parameter_grid(model_name: str) -> list[dict[str, Any]]:
    grid = MODEL_GRIDS.get(model_name, {})
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def mean_rank_ic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: pd.Series,
) -> float:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates).values,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )

    ics = []
    for _, grp in frame.groupby("date"):
        grp = grp.dropna()
        if len(grp) < 3:
            continue
        corr = grp[["y_true", "y_pred"]].corr(method="spearman").iloc[0, 1]
        if pd.notna(corr):
            ics.append(corr)
    if not ics:
        return float("-inf")
    return float(np.mean(ics))


def pooled_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return np.nan
    yt = y_true[mask]
    yp = y_pred[mask]
    denom = np.sum((yt - yt.mean()) ** 2)
    if denom == 0:
        return np.nan
    return 1.0 - np.sum((yt - yp) ** 2) / denom


def tune_and_fit_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    valid_dates: pd.Series,
    cfg: ResearchConfig,
) -> TunedModelResult:
    best_score = float("-inf")
    best_params: dict[str, Any] = {}
    best_estimator = None

    base = build_model(model_name, cfg)
    for params in parameter_grid(model_name):
        estimator = clone(base)
        if params:
            estimator.set_params(**params)
        estimator.fit(x_train, y_train)
        pred = estimator.predict(x_valid)
        score = mean_rank_ic(y_valid.to_numpy(), pred, valid_dates)
        if score > best_score:
            best_score = score
            best_params = params
            best_estimator = estimator

    if best_estimator is None:
        best_estimator = build_model(model_name, cfg)
        best_estimator.fit(x_train, y_train)

    # Refit on train + validation once the best hyperparameters are selected.
    final_estimator = clone(base)
    if best_params:
        final_estimator.set_params(**best_params)
    x_full = pd.concat([x_train, x_valid], axis=0)
    y_full = pd.concat([y_train, y_valid], axis=0)
    final_estimator.fit(x_full, y_full)

    return TunedModelResult(
        model_name=model_name,
        estimator=final_estimator,
        best_params=best_params,
        validation_score=float(best_score),
    )
