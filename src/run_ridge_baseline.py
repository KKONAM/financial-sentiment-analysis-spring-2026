from __future__ import annotations

import argparse
import csv
import json
import warnings
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import tune_lstm


TECHNICAL_FEATURES = [
    "returns",
    "rsi",
    "macd",
    "bbp",
    "momentum",
    "volume",
]
SENTIMENT_FEATURES = [
    "positive_score",
    "negative_score",
    "neutral_score",
    "sentiment_score",
    "sentiment_confidence",
    "article_count",
    "message_count",
]
FEATURE_GROUPS = {
    "combined": TECHNICAL_FEATURES + SENTIMENT_FEATURES,
    "technical_only": TECHNICAL_FEATURES,
    "sentiment_only": SENTIMENT_FEATURES,
}
FEATURE_LABELS = {
    "combined": "Combined",
    "technical_only": "Technical only",
    "sentiment_only": "Sentiment only",
}
DEFAULT_SEQUENCE_LENGTHS = [10, 20, 30, 45]
DEFAULT_ALPHA_GRID = np.logspace(-6, 3, 10).tolist()
DEFAULT_OUTPUT_STEM = tune_lstm.FINAL_REPORT_DIR / f"ridge_baseline_{tune_lstm.EXPERIMENT_SLUG}"


def main() -> None:
    args = parse_args()
    started_at = perf_counter()
    train_cutoff = pd.to_datetime(args.train_end_date).normalize()
    val_cutoff = pd.to_datetime(args.val_end_date).normalize()
    frame = tune_lstm.load_frame(args.data_path)

    rows = []
    predictions_by_group = {}
    for feature_group, feature_columns in FEATURE_GROUPS.items():
        result, predictions = run_feature_group(
            frame=frame,
            feature_group=feature_group,
            feature_columns=feature_columns,
            sequence_lengths=args.sequence_lengths,
            alpha_grid=args.alpha_grid,
            target_column=args.target_column,
            target_horizon_days=args.target_horizon_days,
            train_cutoff=train_cutoff,
            val_cutoff=val_cutoff,
        )
        rows.append(result)
        predictions_by_group[feature_group] = predictions
        test_metrics = result["test_metrics"]
        print(
            "[ridge] "
            f"features={FEATURE_LABELS[feature_group]} "
            f"seq={result['sequence_length']} "
            f"alpha={result['alpha']} "
            f"test_mae={test_metrics['mae']:.4f} "
            f"test_rmse={test_metrics['rmse']:.4f} "
            f"test_dir={test_metrics['directional_accuracy']:.4f} "
            f"test_corr={test_metrics['correlation']:.4f}",
            flush=True,
        )

    summary = {
        "model": "Ridge",
        "experiment": tune_lstm.EXPERIMENT_SLUG,
        "data_path": str(args.data_path),
        "target_column": args.target_column,
        "target_horizon_days": args.target_horizon_days,
        "train_end_date": args.train_end_date,
        "val_end_date": args.val_end_date,
        "sequence_lengths": args.sequence_lengths,
        "alpha_grid": args.alpha_grid,
        "selection_metric": "validation_rmse",
        "results": rows,
        "runtime_seconds": round(perf_counter() - started_at, 3),
    }

    json_path = args.output_stem.with_suffix(".json")
    csv_path = args.output_stem.with_suffix(".csv")
    predictions_path = args.output_stem.with_name(f"{args.output_stem.name}_predictions.csv")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_results_csv(csv_path, rows)
    write_predictions_csv(predictions_path, predictions_by_group)

    print(f"[ridge] wrote {json_path}", flush=True)
    print(f"[ridge] wrote {csv_path}", flush=True)
    print(f"[ridge] wrote {predictions_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ridge baselines on flattened rolling return sequences.")
    parser.add_argument("--data-path", type=Path, default=tune_lstm.DATA_PATH)
    parser.add_argument("--output-stem", type=Path, default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--target-column", default=tune_lstm.TARGET_COLUMN)
    parser.add_argument("--target-horizon-days", type=int, default=tune_lstm.TARGET_HORIZON_DAYS)
    parser.add_argument("--train-end-date", default=tune_lstm.TRAIN_END_DATE)
    parser.add_argument("--val-end-date", default=tune_lstm.VAL_END_DATE)
    parser.add_argument("--sequence-lengths", type=int, nargs="+", default=DEFAULT_SEQUENCE_LENGTHS)
    parser.add_argument("--alpha-grid", type=float, nargs="+", default=DEFAULT_ALPHA_GRID)
    return parser.parse_args()


def run_feature_group(
    frame: pd.DataFrame,
    feature_group: str,
    feature_columns: list[str],
    sequence_lengths: list[int],
    alpha_grid: list[float],
    target_column: str,
    target_horizon_days: int,
    train_cutoff: pd.Timestamp,
    val_cutoff: pd.Timestamp,
) -> tuple[dict[str, Any], pd.DataFrame]:
    scaled_frame, scaler = scale_features(
        frame=frame,
        feature_columns=feature_columns,
        train_cutoff=train_cutoff,
        target_column=target_column,
    )
    best_result = None

    for sequence_length in sequence_lengths:
        X, y, meta = build_flattened_sequences(
            frame=scaled_frame,
            feature_columns=feature_columns,
            target_column=target_column,
            sequence_length=sequence_length,
            target_horizon_days=target_horizon_days,
        )
        split_indices = split_sequence_indices(meta, train_cutoff, val_cutoff)
        for alpha in alpha_grid:
            model = Ridge(alpha=float(alpha))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=LinAlgWarning)
                model.fit(X[split_indices["train"]], y[split_indices["train"]])
            val_predictions = model.predict(X[split_indices["val"]])
            validation_metrics = tune_lstm.regression_metrics(y[split_indices["val"]], val_predictions)
            candidate = {
                "validation_rmse": validation_metrics["rmse"],
                "sequence_length": sequence_length,
                "alpha": float(alpha),
                "model": model,
                "X": X,
                "y": y,
                "meta": meta,
                "split_indices": split_indices,
                "validation_metrics": validation_metrics,
            }
            if best_result is None or candidate["validation_rmse"] < best_result["validation_rmse"]:
                best_result = candidate

    if best_result is None:
        raise RuntimeError(f"No ridge trials completed for feature group: {feature_group}")

    test_idx = best_result["split_indices"]["test"]
    test_predictions = best_result["model"].predict(best_result["X"][test_idx])
    test_targets = best_result["y"][test_idx]
    test_metrics = tune_lstm.regression_metrics(test_targets, test_predictions)
    prediction_frame = best_result["meta"].iloc[test_idx].reset_index(drop=True).copy()
    prediction_frame["feature_group"] = feature_group
    prediction_frame["actual_return"] = test_targets
    prediction_frame["predicted_return"] = test_predictions

    result = {
        "model": "Ridge",
        "feature_group": feature_group,
        "features": FEATURE_LABELS[feature_group],
        "feature_columns": feature_columns,
        "sequence_length": best_result["sequence_length"],
        "alpha": best_result["alpha"],
        "split_counts": {
            "train": int(len(best_result["split_indices"]["train"])),
            "val": int(len(best_result["split_indices"]["val"])),
            "test": int(len(test_idx)),
        },
        "validation_metrics": best_result["validation_metrics"],
        "test_metrics": test_metrics,
        "feature_scaler_mean": scaler.mean_.tolist(),
        "feature_scaler_scale": scaler.scale_.tolist(),
    }
    return result, prediction_frame


def scale_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_cutoff: pd.Timestamp,
    target_column: str = tune_lstm.TARGET_COLUMN,
) -> tuple[pd.DataFrame, StandardScaler]:
    missing_columns = [
        column
        for column in [*feature_columns, target_column, "ticker", "Date"]
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    scaled = frame.copy()
    train_rows = scaled["Date"] <= train_cutoff
    scaler = StandardScaler()
    scaler.fit(scaled.loc[train_rows, feature_columns])
    scaled[feature_columns] = scaler.transform(scaled[feature_columns])
    return scaled, scaler


def build_flattened_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
    target_horizon_days: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features = []
    targets = []
    metadata = []
    for ticker, ticker_frame in frame.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("Date").reset_index(drop=True)
        values = ticker_frame[feature_columns].to_numpy(dtype=np.float32)
        target_values = ticker_frame[target_column].to_numpy(dtype=np.float32)
        for end_idx in range(sequence_length - 1, len(ticker_frame)):
            start_idx = end_idx - sequence_length + 1
            target_end_idx = end_idx + target_horizon_days
            target_end_date = (
                ticker_frame.loc[target_end_idx, "Date"]
                if target_end_idx < len(ticker_frame)
                else pd.NaT
            )
            features.append(values[start_idx : end_idx + 1].reshape(-1))
            targets.append(target_values[end_idx])
            metadata.append(
                {
                    "ticker": ticker,
                    "Date": ticker_frame.loc[end_idx, "Date"],
                    "target_end_date": target_end_date,
                }
            )

    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), pd.DataFrame(metadata)


def split_sequence_indices(
    meta: pd.DataFrame,
    train_cutoff: pd.Timestamp,
    val_cutoff: pd.Timestamp,
) -> dict[str, np.ndarray]:
    dates = pd.to_datetime(meta["Date"])
    target_end_dates = pd.to_datetime(meta["target_end_date"])
    indices = {
        "train": np.flatnonzero((dates <= train_cutoff) & (target_end_dates <= train_cutoff)),
        "val": np.flatnonzero((dates > train_cutoff) & (dates <= val_cutoff) & (target_end_dates <= val_cutoff)),
        "test": np.flatnonzero(dates > val_cutoff),
    }
    if any(len(split_indices) == 0 for split_indices in indices.values()):
        raise ValueError(
            "Purged split produced an empty split: "
            f"train={len(indices['train'])} val={len(indices['val'])} test={len(indices['test'])}"
        )
    return indices


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "model",
        "features",
        "feature_group",
        "sequence_length",
        "alpha",
        "val_mae",
        "val_rmse",
        "val_directional_accuracy",
        "val_correlation",
        "val_prediction_std",
        "test_mae",
        "test_rmse",
        "test_directional_accuracy",
        "test_correlation",
        "test_prediction_std",
        "train_rows",
        "val_rows",
        "test_rows",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            validation = row["validation_metrics"]
            test = row["test_metrics"]
            split_counts = row["split_counts"]
            writer.writerow(
                {
                    "model": row["model"],
                    "features": row["features"],
                    "feature_group": row["feature_group"],
                    "sequence_length": row["sequence_length"],
                    "alpha": row["alpha"],
                    "val_mae": validation["mae"],
                    "val_rmse": validation["rmse"],
                    "val_directional_accuracy": validation["directional_accuracy"],
                    "val_correlation": validation["correlation"],
                    "val_prediction_std": validation["prediction_std"],
                    "test_mae": test["mae"],
                    "test_rmse": test["rmse"],
                    "test_directional_accuracy": test["directional_accuracy"],
                    "test_correlation": test["correlation"],
                    "test_prediction_std": test["prediction_std"],
                    "train_rows": split_counts["train"],
                    "val_rows": split_counts["val"],
                    "test_rows": split_counts["test"],
                }
            )


def write_predictions_csv(path: Path, predictions_by_group: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(predictions_by_group.values(), ignore_index=True)
    predictions.to_csv(path, index=False)


if __name__ == "__main__":
    main()
