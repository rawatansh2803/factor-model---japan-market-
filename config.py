from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json


@dataclass
class ResearchConfig:
    # Data paths
    prices_path: str = "data/sample/prices.csv"
    fundamentals_path: str = "data/sample/fundamentals.csv"
    benchmark_path: str = "data/sample/benchmark.csv"
    benchmark_weights_path: str | None = "data/sample/benchmark_weights.csv"
    results_dir: str = "results"

    # Data columns
    date_col: str = "date"
    ticker_col: str = "ticker"
    price_col: str = "adj_close"
    return_col: str = "return"
    market_cap_col: str = "market_cap"
    volume_col: str = "volume"
    benchmark_return_col: str = "benchmark_return"
    benchmark_level_col: str = "benchmark_level"
    benchmark_weight_col: str = "benchmark_weight"
    fundamental_date_col: str = "effective_date"

    # Fundamental inputs used for value features
    book_equity_col: str = "book_equity"
    net_income_col: str = "net_income"
    sales_col: str = "sales"
    cashflow_col: str = "operating_cash_flow"

    # Universe / timing
    benchmark_name: str = "TOPIX"
    frequency: str = "M"
    horizon_months: int = 1
    fundamental_lag_months: int = 3
    liquidity_quantile: float = 0.30
    min_market_cap: float = 0.0
    min_obs_per_date: int = 20

    # Walk-forward procedure
    rolling_training_window: bool = True
    train_months: int = 36
    validation_months: int = 12
    test_months: int = 1

    # Feature handling
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    standardize_cross_sectionally: bool = True

    # Portfolio construction
    portfolio_mode: str = "benchmark_tilt"  # benchmark_tilt or top_quantile
    active_budget: float = 0.20
    max_weight: float = 0.05
    top_quantile: float = 0.20
    bottom_quantile: float = 0.20
    max_turnover: float = 0.30
    transaction_cost_bps: float = 15.0
    use_benchmark_weights: bool = True

    # Model set
    models: list[str] = field(
        default_factory=lambda: [
            "factor_sort",
            "linear",
            "elastic_net",
            "random_forest",
            "gradient_boosting",
            "mlp",
        ]
    )
    random_state: int = 42

    # Output / annualisation
    annualisation: float | None = None

    def resolved_annualisation(self) -> float:
        if self.annualisation is not None:
            return self.annualisation
        return 12.0 / max(self.horizon_months, 1)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "ResearchConfig":
        data = json.loads(Path(path).read_text())
        return cls(**data)
