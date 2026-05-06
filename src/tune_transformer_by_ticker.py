from __future__ import annotations

import argparse
import json
from dataclasses import asdict, fields
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from tune_lstm import (
    DATA_PATH,
    DEFAULT_TRAINING_SEED,
    EXPERIMENT_SLUG,
    FEATURE_COLUMNS,
    FINAL_CHECKPOINT_DIR,
    FINAL_REPORT_DIR,
    TARGET_COLUMN,
    TRAIN_END_DATE,
    VAL_END_DATE,
    load_frame,
    prepare_data,
    scale_features,
)
from tune_transformer import (
    DEFAULT_TRIALS,
    TrialConfig,
    build_search_options,
    evaluate_test_best,
    run_trial,
    sample_configs,
)


SYMBOLS = ["AAPL", "AMZN", "META", "NVDA", "TSLA"]
SUMMARY_CSV_PATH = FINAL_REPORT_DIR / f"transformer_per_ticker_tuning_comparison_{EXPERIMENT_SLUG}.csv"
SUMMARY_JSON_PATH = FINAL_REPORT_DIR / f"transformer_per_ticker_tuning_comparison_{EXPERIMENT_SLUG}.json"


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(args.torch_threads, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    started_at = perf_counter()
    frame = load_frame(args.data_path)
    tickers = normalize_tickers(args.tickers)
    trial_configs = sample_configs(
        build_search_options(),
        trials=args.trials,
        seed=args.search_seed,
        training_seed=args.training_seed,
    )

    print(
        f"[tune_transformer_by_ticker] device={device} tickers={','.join(tickers)} "
        f"trials={len(trial_configs)} max_epochs={args.max_epochs} patience={args.patience}",
        flush=True,
    )

    summaries = []
    for ticker in tickers:
        ticker_summary = tune_one_ticker(
            ticker=ticker,
            frame=frame,
            trial_configs=trial_configs,
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
        )
        summaries.append(ticker_summary)

    summary_frame = pd.DataFrame(summaries).sort_values(["test_mae", "test_rmse"])
    args.summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_frame.to_csv(args.summary_csv_path, index=False)
    args.summary_json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("[tune_transformer_by_ticker] final per-ticker summary:")
    print(summary_frame.to_string(index=False))
    print(f"[tune_transformer_by_ticker] wrote {args.summary_csv_path}")
    print(f"[tune_transformer_by_ticker] wrote {args.summary_json_path}")
    print(f"[tune_transformer_by_ticker] total runtime seconds={perf_counter() - started_at:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune separate Transformer models for each ticker.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--tickers", nargs="+", default=SYMBOLS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--summary-csv-path", type=Path, default=SUMMARY_CSV_PATH)
    parser.add_argument("--summary-json-path", type=Path, default=SUMMARY_JSON_PATH)
    return parser.parse_args()


def normalize_tickers(tickers: list[str]) -> list[str]:
    normalized = [ticker.upper() for ticker in tickers]
    unknown = sorted(set(normalized) - set(SYMBOLS))
    if unknown:
        raise ValueError(f"Unsupported ticker(s): {', '.join(unknown)}")
    return normalized


def tune_one_ticker(
    ticker: str,
    frame: pd.DataFrame,
    trial_configs: list[TrialConfig],
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float,
) -> dict[str, object]:
    ticker_started_at = perf_counter()
    train_cutoff = pd.to_datetime(TRAIN_END_DATE).normalize()
    val_cutoff = pd.to_datetime(VAL_END_DATE).normalize()
    ticker_frame = frame[frame["ticker"] == ticker].copy()
    if ticker_frame.empty:
        raise ValueError(f"No rows found for ticker {ticker}.")

    scaled_frame, feature_scaler = scale_features(ticker_frame, train_cutoff)
    print(
        f"[tune_transformer_by_ticker] ticker={ticker} rows={len(ticker_frame)}",
        flush=True,
    )

    rows = []
    best_row = None
    best_state = None
    best_artifacts = None
    for trial_number, config in enumerate(trial_configs, start=1):
        trial_started_at = perf_counter()
        artifacts = prepare_data(
            frame=scaled_frame,
            sequence_length=config.sequence_length,
            include_current_row=config.include_current_row,
            train_cutoff=train_cutoff,
            val_cutoff=val_cutoff,
            scale_target=config.scale_target,
            device=device,
        )
        row, state_dict = run_trial(
            config=config,
            artifacts=artifacts,
            device=device,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
        )
        row["ticker"] = ticker
        row["trial"] = trial_number
        row["seconds"] = round(perf_counter() - trial_started_at, 3)
        rows.append(row)

        if best_row is None or row["selection_score"] < best_row["selection_score"]:
            best_row = row
            best_state = state_dict
            best_artifacts = artifacts

        print(
            "[tune_transformer_by_ticker] "
            f"ticker={ticker} "
            f"trial={trial_number:03d}/{len(trial_configs):03d} "
            f"score={row['selection_score']:.6f} "
            f"val_mae={row['val_mae']:.6f} "
            f"val_rmse={row['val_rmse']:.6f} "
            f"val_dir={row['val_directional_accuracy']:.4f} "
            f"epochs={row['epochs_ran']:02d} "
            f"best={best_row['selection_score']:.6f}",
            flush=True,
        )

    if best_row is None or best_state is None or best_artifacts is None:
        raise RuntimeError(f"No tuning trials completed for {ticker}.")

    best_config = TrialConfig(**{field.name: best_row[field.name] for field in fields(TrialConfig)})
    test_metrics, predictions = evaluate_test_best(
        config=best_config,
        state_dict=best_state,
        artifacts=best_artifacts,
        device=device,
    )
    paths = output_paths(ticker)

    results = pd.DataFrame(rows).sort_values(["selection_score", "val_mae", "val_rmse"])
    paths["search_results"].parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(paths["search_results"], index=False)
    predictions.to_csv(paths["predictions"], index=False)
    torch.save(
        {
            "ticker": ticker,
            "model_state_dict": best_state,
            "config": asdict(best_config),
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "feature_scaler_mean": feature_scaler.mean_.tolist(),
            "feature_scaler_scale": feature_scaler.scale_.tolist(),
            "target_mean": best_artifacts["target_mean"],
            "target_scale": best_artifacts["target_scale"],
            "train_cutoff": train_cutoff.isoformat(),
            "val_cutoff": val_cutoff.isoformat(),
            "validation_metrics": best_row,
            "test_metrics": test_metrics,
        },
        paths["checkpoint"],
    )

    summary = {
        "model": "transformer",
        "ticker": ticker,
        "best_trial": int(best_row["trial"]),
        "epochs_ran": int(best_row["epochs_ran"]),
        "val_score": float(best_row["selection_score"]),
        "val_mae": float(best_row["val_mae"]),
        "val_rmse": float(best_row["val_rmse"]),
        "val_directional_accuracy": float(best_row["val_directional_accuracy"]),
        "val_correlation": float(best_row["val_correlation"]),
        "test_mae": float(test_metrics["mae"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_directional_accuracy": float(test_metrics["directional_accuracy"]),
        "test_directional_baseline_accuracy": float(test_metrics["directional_baseline_accuracy"]),
        "test_directional_lift": float(test_metrics["directional_lift"]),
        "test_correlation": float(test_metrics["correlation"]),
        "test_prediction_mean": float(test_metrics["prediction_mean"]),
        "test_prediction_std": float(test_metrics["prediction_std"]),
        "paths": {key: str(value) for key, value in paths.items()},
        "runtime_seconds": round(perf_counter() - ticker_started_at, 3),
    }
    paths["best_config"].write_text(
        json.dumps(
            {
                "ticker": ticker,
                "best_validation": best_row,
                "test_metrics": test_metrics,
                "paths": {key: str(value) for key, value in paths.items()},
                "runtime_seconds": summary["runtime_seconds"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[tune_transformer_by_ticker] final test metrics:")
    print(json.dumps(summary, indent=2))
    return summary


def output_paths(ticker: str) -> dict[str, Path]:
    ticker_slug = ticker.lower()
    stem = f"transformer_{ticker_slug}_tuned_{EXPERIMENT_SLUG}"
    return {
        "search_results": FINAL_REPORT_DIR / f"{stem}_search.csv",
        "best_config": FINAL_REPORT_DIR / f"{stem}_best.json",
        "predictions": FINAL_REPORT_DIR / f"{stem}_predictions.csv",
        "checkpoint": FINAL_CHECKPOINT_DIR / f"{stem}_best.pt",
    }


if __name__ == "__main__":
    main()
