"""Collect tuned combined/subset experiment results into one comparison table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import tune_lstm


EXPERIMENT_SLUG = tune_lstm.EXPERIMENT_SLUG
DEFAULT_REPORT_DIR = tune_lstm.FINAL_REPORT_DIR
DEFAULT_CSV_PATH = DEFAULT_REPORT_DIR / f"feature_subset_tuning_comparison_{EXPERIMENT_SLUG}.csv"
DEFAULT_JSON_PATH = DEFAULT_REPORT_DIR / f"feature_subset_tuning_comparison_{EXPERIMENT_SLUG}.json"

EXPERIMENTS = [
    ("LSTM", "combined", "lstm_combined"),
    ("LSTM", "technical_only", "lstm_technical_only"),
    ("LSTM", "sentiment_only", "lstm_sentiment_only"),
    ("GRU", "combined", "gru_combined"),
    ("GRU", "technical_only", "gru_technical_only"),
    ("GRU", "sentiment_only", "gru_sentiment_only"),
]

FIELDNAMES = [
    "model",
    "features",
    "val_mae",
    "val_rmse",
    "val_directional_accuracy",
    "val_directional_baseline_accuracy",
    "val_directional_lift",
    "val_correlation",
    "test_mae",
    "test_rmse",
    "test_directional_accuracy",
    "test_directional_baseline_accuracy",
    "test_directional_lift",
    "test_correlation",
    "prediction_mean",
    "prediction_std",
    "best_trial",
    "epochs_ran",
    "source",
]


def best_path(report_dir: Path, stem: str) -> Path:
    return report_dir / f"{stem}_tuned_{EXPERIMENT_SLUG}_best.json"


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing tuned result file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_summary(model: str, features: str, source_path: Path) -> dict[str, Any]:
    summary = load_summary(source_path)
    validation = summary["best_validation"]
    test = summary["test_metrics"]
    return {
        "model": model,
        "features": features,
        "val_mae": validation["val_mae"],
        "val_rmse": validation["val_rmse"],
        "val_directional_accuracy": validation["val_directional_accuracy"],
        "val_directional_baseline_accuracy": validation["val_directional_baseline_accuracy"],
        "val_directional_lift": validation["val_directional_lift"],
        "val_correlation": validation["val_correlation"],
        "test_mae": test["mae"],
        "test_rmse": test["rmse"],
        "test_directional_accuracy": test["directional_accuracy"],
        "test_directional_baseline_accuracy": test["directional_baseline_accuracy"],
        "test_directional_lift": test["directional_lift"],
        "test_correlation": test["correlation"],
        "prediction_mean": test["prediction_mean"],
        "prediction_std": test["prediction_std"],
        "best_trial": validation["trial"],
        "epochs_ran": validation["epochs_ran"],
        "source": str(source_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect tuned experiment results.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        row_from_summary(model, features, best_path(args.report_dir, stem))
        for model, features, stem in EXPERIMENTS
    ]

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    args.json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {args.csv_path}")
    print(f"Wrote {args.json_path}")


if __name__ == "__main__":
    main()
