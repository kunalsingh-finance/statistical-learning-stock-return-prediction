from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import RunConfig, run_project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FA590 stock return prediction workflow.")
    parser.add_argument("--data-path", default=None, help="Path to the return prediction CSV.")
    parser.add_argument("--output-dir", default="outputs/latest_run", help="Directory for CSV outputs and charts.")
    parser.add_argument("--rf-estimators", type=int, default=50, help="Number of trees for Random Forest.")
    parser.add_argument("--gb-estimators", type=int, default=50, help="Number of trees for Gradient Boosting.")
    parser.add_argument("--nn-epochs", type=int, default=25, help="Maximum epochs for the neural network.")
    parser.add_argument("--nn-batch-size", type=int, default=256, help="Batch size for the neural network.")
    parser.add_argument("--demo-months", type=int, default=36, help="Months to generate for demo mode.")
    parser.add_argument("--demo-stocks", type=int, default=180, help="Stocks per month for demo mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        data_path=args.data_path,
        output_dir=Path(args.output_dir).resolve(),
        rf_estimators=args.rf_estimators,
        gb_estimators=args.gb_estimators,
        nn_epochs=args.nn_epochs,
        nn_batch_size=args.nn_batch_size,
        demo_months=args.demo_months,
        demo_stocks=args.demo_stocks,
    )
    summary = run_project(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
