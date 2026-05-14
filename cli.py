from __future__ import annotations

import argparse
from pathlib import Path

from .config import ResearchConfig
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Japanese ML factor model pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional path to a JSON config file.",
    )
    args = parser.parse_args()

    if args.config:
        cfg = ResearchConfig.load(args.config)
    else:
        cfg = ResearchConfig()

    outputs = run_pipeline(cfg)
    out_dir = Path(outputs["output_dir"].iloc[0, 0])
    print(f"Research run completed. Results saved to: {out_dir}")
    print("\nPrediction summary:\n")
    print(outputs["prediction_summary"].to_string(index=False))
    print("\nBacktest summary:\n")
    print(outputs["backtest_summary"].to_string(index=False))


if __name__ == "__main__":
    main()
