from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def generate_sample_data(output_dir: str = "data/sample", seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range("2014-01-31", "2024-12-31", freq="ME")
    n_tickers = 120
    tickers = [f"JP{1000 + i}" for i in range(n_tickers)]

    rows = []
    fund_rows = []
    weight_rows = []

    style_value = rng.normal(0, 1, n_tickers)
    style_quality = rng.normal(0, 1, n_tickers)
    market_beta = rng.normal(1.0, 0.2, n_tickers)
    base_caps = np.exp(rng.normal(24, 1.0, n_tickers))
    prices = rng.uniform(400, 3500, n_tickers)

    benchmark_rets = []
    benchmark_level = 1000.0

    prev_ret = np.zeros(n_tickers)
    for t, date in enumerate(dates):
        market = rng.normal(0.006, 0.04)
        benchmark_level *= 1 + market
        benchmark_rets.append({"date": date, "benchmark_return": market, "benchmark_level": benchmark_level})

        momentum_state = 0.30 * prev_ret + rng.normal(0, 0.02, n_tickers)
        size_noise = rng.normal(0, 0.1, n_tickers)
        market_caps = base_caps * np.exp(0.01 * t + size_noise)
        monthly_rets = (
            market_beta * market
            + 0.012 * style_value
            + 0.018 * momentum_state
            + 0.004 * style_quality
            + rng.normal(0, 0.07, n_tickers)
        )
        monthly_rets = np.clip(monthly_rets, -0.35, 0.35)
        prev_ret = monthly_rets
        prices = prices * (1 + monthly_rets)

        benchmark_weights = market_caps / market_caps.sum()

        for i, ticker in enumerate(tickers):
            volume = int(rng.integers(5_000_000, 50_000_000))
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adj_close": prices[i],
                    "return": monthly_rets[i],
                    "volume": volume,
                    "market_cap": market_caps[i],
                }
            )
            weight_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "benchmark_weight": benchmark_weights[i],
                }
            )

        # Quarterly fundamentals with a realistic reporting delay handled in pipeline.
        if date.month in {3, 6, 9, 12}:
            for i, ticker in enumerate(tickers):
                market_cap = market_caps[i]
                book_equity = market_cap * np.exp(rng.normal(0.0 + 0.35 * style_value[i], 0.25))
                net_income = market_cap * (0.015 + 0.008 * style_quality[i] + rng.normal(0, 0.01))
                sales = market_cap * (0.45 + 0.20 * style_quality[i] + rng.normal(0, 0.05))
                operating_cash_flow = market_cap * (0.02 + 0.01 * style_quality[i] + rng.normal(0, 0.01))
                fund_rows.append(
                    {
                        "effective_date": date,
                        "ticker": ticker,
                        "book_equity": max(book_equity, 1.0),
                        "net_income": net_income,
                        "sales": max(sales, 1.0),
                        "operating_cash_flow": operating_cash_flow,
                    }
                )

    pd.DataFrame(rows).to_csv(out / "prices.csv", index=False)
    pd.DataFrame(fund_rows).to_csv(out / "fundamentals.csv", index=False)
    pd.DataFrame(benchmark_rets).to_csv(out / "benchmark.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(out / "benchmark_weights.csv", index=False)
    print(f"Sample data written to {out.resolve()}")


if __name__ == "__main__":
    generate_sample_data()
