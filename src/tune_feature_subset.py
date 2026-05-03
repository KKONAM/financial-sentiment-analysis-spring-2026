from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import tune_gru
import tune_lstm
from features.build_features import MARKET_FEATURES, SENTIMENT_FEATURES, combine_indicators_and_sentiment


SYMBOLS = ["AAPL", "AMZN", "META", "NVDA", "TSLA"]
START_DATE = "2020-01-01"
END_DATE = "2022-02-28"
TARGET_HORIZON_DAYS = 5
TARGET_COLUMN = "future_return_5d"
TRAIN_END_DATE = "2021-04-30"
VAL_END_DATE = "2021-08-31"
EXPERIMENT_SLUG = tune_lstm.EXPERIMENT_SLUG
DATA_PATH = tune_lstm.DATA_PATH
FINAL_REPORT_DIR = tune_lstm.FINAL_REPORT_DIR
FINAL_CHECKPOINT_DIR = tune_lstm.FINAL_CHECKPOINT_DIR
FEATURE_GROUPS = {
    "technical_only": MARKET_FEATURES,
    "sentiment_only": SENTIMENT_FEATURES,
}


def main() -> None:
    args = parse_args()
    model_module = model_module_for(args.model)
    feature_columns = FEATURE_GROUPS[args.feature_group]
    torch.set_num_threads(max(1, min(args.torch_threads, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    started_at = perf_counter()
    frame = load_frame(args.data_path)
    train_cutoff = pd.to_datetime(TRAIN_END_DATE).normalize()
    val_cutoff = pd.to_datetime(VAL_END_DATE).normalize()
    scaled_frame, feature_scaler = scale_features(frame, feature_columns, train_cutoff)
    trial_configs = model_module.sample_configs(
        model_module.build_search_options(),
        trials=args.trials,
        seed=args.search_seed,
        training_seed=args.training_seed,
    )

    print(
        f"[subset_tune] model={args.model} features={args.feature_group} "
        f"device={device} trials={len(trial_configs)}",
        flush=True,
    )

    rows = []
    best_row = None
    best_state = None
    best_artifacts = None

    for trial_number, config in enumerate(trial_configs, start=1):
        trial_started_at = perf_counter()
        set_seed(config.seed)
        artifacts = prepare_data(
            frame=scaled_frame,
            feature_columns=feature_columns,
            sequence_length=config.sequence_length,
            include_current_row=config.include_current_row,
            train_cutoff=train_cutoff,
            val_cutoff=val_cutoff,
            scale_target=config.scale_target,
            device=device,
        )
        row, state_dict = run_trial(
            model_module=model_module,
            config=config,
            artifacts=artifacts,
            feature_count=len(feature_columns),
            device=device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            min_delta=args.min_delta,
        )
        row["trial"] = trial_number
        row["seconds"] = round(perf_counter() - trial_started_at, 3)
        rows.append(row)

        if best_row is None or row["selection_score"] < best_row["selection_score"]:
            best_row = row
            best_state = state_dict
            best_artifacts = artifacts

        print(
            "[subset_tune] "
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
        raise RuntimeError("No subset tuning trials completed.")

    results = pd.DataFrame(rows).sort_values(["selection_score", "val_mae", "val_rmse"])
    search_results_path, best_config_path, predictions_path, checkpoint_path = output_paths(
        model=args.model,
        feature_group=args.feature_group,
        output_suffix=args.output_suffix,
    )
    search_results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(search_results_path, index=False)

    best_config = config_from_row(model_module, best_row)
    test_metrics, predictions = evaluate_test_best(
        model_module=model_module,
        config=best_config,
        state_dict=best_state,
        artifacts=best_artifacts,
        feature_count=len(feature_columns),
        device=device,
    )

    best_summary = {
        "model": args.model,
        "feature_group": args.feature_group,
        "feature_columns": feature_columns,
        "best_validation": best_row,
        "test_metrics": test_metrics,
        "paths": {
            "search_results": str(search_results_path),
            "best_config": str(best_config_path),
            "predictions": str(predictions_path),
            "checkpoint": str(checkpoint_path),
        },
        "runtime_seconds": round(perf_counter() - started_at, 3),
    }

    best_config_path.parent.mkdir(parents=True, exist_ok=True)
    best_config_path.write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_path, index=False)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(best_config),
            "model": args.model,
            "feature_group": args.feature_group,
            "feature_columns": feature_columns,
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
        checkpoint_path,
    )

    print("[subset_tune] best validation config:")
    print(json.dumps(best_row, indent=2))
    print("[subset_tune] final test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"[subset_tune] wrote {search_results_path}")
    print(f"[subset_tune] wrote {best_config_path}")
    print(f"[subset_tune] wrote {predictions_path}")
    print(f"[subset_tune] wrote {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune LSTM/GRU on a single feature subset.")
    parser.add_argument("--model", choices=["lstm", "gru"], required=True)
    parser.add_argument("--feature-group", choices=sorted(FEATURE_GROUPS), required=True)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--trials", type=int, default=tune_lstm.DEFAULT_TRIALS)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--training-seed", type=int, default=tune_lstm.DEFAULT_TRAINING_SEED)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-suffix", default=EXPERIMENT_SLUG)
    args = parser.parse_args()
    if args.max_epochs is None:
        args.max_epochs = 60 if args.model == "lstm" else 80
    if args.patience is None:
        args.patience = 8 if args.model == "lstm" else 10
    return args


def model_module_for(model_name: str):
    if model_name == "lstm":
        return tune_lstm
    if model_name == "gru":
        return tune_gru
    raise ValueError(f"Unsupported model: {model_name}")


def output_paths(model: str, feature_group: str, output_suffix: str) -> tuple[Path, Path, Path, Path]:
    stem = f"{model}_{feature_group}_tuned_{output_suffix}"
    return (
        FINAL_REPORT_DIR / f"{stem}_search.csv",
        FINAL_REPORT_DIR / f"{stem}_best.json",
        FINAL_REPORT_DIR / f"{stem}_predictions.csv",
        FINAL_CHECKPOINT_DIR / f"{stem}_best.pt",
    )


def load_frame(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["Date"])

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = combine_indicators_and_sentiment(
        symbols=SYMBOLS,
        start_date=START_DATE,
        end_date=END_DATE,
        train_end=TRAIN_END_DATE,
        forecast_horizon=TARGET_HORIZON_DAYS,
        text_column="title",
        news_date_column="date",
        news_ticker_column="ticker",
    )
    frame.to_csv(path, index=False)
    return frame


def scale_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    train_cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, StandardScaler]:
    missing_columns = [column for column in [*feature_columns, TARGET_COLUMN, "ticker", "Date"] if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    scaled = frame.copy()
    train_rows = scaled["Date"] <= train_cutoff
    scaler = StandardScaler()
    scaler.fit(scaled.loc[train_rows, feature_columns])
    scaled[feature_columns] = scaler.transform(scaled[feature_columns])
    return scaled, scaler


def prepare_data(
    frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    include_current_row: bool,
    train_cutoff: pd.Timestamp,
    val_cutoff: pd.Timestamp,
    scale_target: bool,
    device: torch.device,
) -> dict[str, object]:
    X, y, meta = build_sequences(
        frame=frame,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        sequence_length=sequence_length,
        include_current_row=include_current_row,
    )
    dates = pd.to_datetime(meta["Date"])
    target_end_dates = pd.to_datetime(meta["target_end_date"])
    train_idx = np.flatnonzero((dates <= train_cutoff) & (target_end_dates <= train_cutoff))
    val_idx = np.flatnonzero((dates > train_cutoff) & (dates <= val_cutoff) & (target_end_dates <= val_cutoff))
    test_idx = np.flatnonzero(dates > val_cutoff)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            "Purged split produced an empty split: "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )

    target_mean = float(np.mean(y[train_idx])) if scale_target else 0.0
    target_scale = float(np.std(y[train_idx])) if scale_target else 1.0
    if np.isclose(target_scale, 0.0):
        target_scale = 1.0
    model_y = (y - target_mean) / target_scale

    return {
        "X": X,
        "raw_y": y,
        "model_y": model_y.astype(np.float32),
        "meta": meta,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "x_train": torch.tensor(X[train_idx], dtype=torch.float32, device=device),
        "y_train": torch.tensor(model_y[train_idx], dtype=torch.float32, device=device),
        "x_val": torch.tensor(X[val_idx], dtype=torch.float32, device=device),
        "y_val": torch.tensor(model_y[val_idx], dtype=torch.float32, device=device),
        "x_test": torch.tensor(X[test_idx], dtype=torch.float32, device=device),
        "y_test": torch.tensor(model_y[test_idx], dtype=torch.float32, device=device),
    }


def build_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    sequence_length: int,
    include_current_row: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features = []
    targets = []
    metadata = []
    for ticker, ticker_frame in frame.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("Date").reset_index(drop=True)
        values = ticker_frame[feature_columns].to_numpy(dtype=np.float32)
        target_values = ticker_frame[target_column].to_numpy(dtype=np.float32)
        first_end = sequence_length - 1 if include_current_row else sequence_length
        for end_idx in range(first_end, len(ticker_frame)):
            start_idx = end_idx - sequence_length + 1 if include_current_row else end_idx - sequence_length
            end_slice = end_idx + 1 if include_current_row else end_idx
            target_end_idx = end_idx + TARGET_HORIZON_DAYS
            target_end_date = (
                ticker_frame.loc[target_end_idx, "Date"]
                if target_end_idx < len(ticker_frame)
                else pd.NaT
            )
            features.append(values[start_idx:end_slice])
            targets.append(target_values[end_idx])
            metadata.append(
                {
                    "ticker": ticker,
                    "Date": ticker_frame.loc[end_idx, "Date"],
                    "target_end_date": target_end_date,
                }
            )

    if not features:
        raise ValueError("Not enough rows to create sequences.")
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), pd.DataFrame(metadata)


def run_trial(
    model_module,
    config,
    artifacts: dict[str, object],
    feature_count: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float,
) -> tuple[dict[str, float | int | str | bool | None], dict[str, torch.Tensor]]:
    model = model_module.build_model(config, input_size=feature_count).to(device)
    optimizer = model_module.build_optimizer(config, model)
    loss_fn = model_module.build_loss(config)
    train_loader = DataLoader(
        TensorDataset(artifacts["x_train"], artifacts["y_train"]),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(artifacts["x_val"], artifacts["y_val"]),
        batch_size=config.batch_size,
        shuffle=False,
    )

    best_score = float("inf")
    best_metrics = None
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            if config.gradient_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()

        metrics = evaluate_split(
            model=model,
            loader=val_loader,
            raw_targets=np.asarray(artifacts["raw_y"])[artifacts["val_idx"]],
            target_mean=float(artifacts["target_mean"]),
            target_scale=float(artifacts["target_scale"]),
        )
        score = model_module.selection_score(metrics)

        if score < best_score - min_delta:
            best_score = score
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_metrics is None:
        raise RuntimeError("Trial failed to evaluate validation metrics.")

    row = {
        **asdict(config),
        **{f"val_{key}": value for key, value in best_metrics.items()},
        "selection_score": best_score,
        "epochs_ran": epoch,
    }
    return row, best_state


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    raw_targets: np.ndarray,
    target_mean: float,
    target_scale: float,
) -> dict[str, float]:
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_x, _ in loader:
            predictions.append(model(batch_x).detach().cpu().numpy())
    predicted = np.concatenate(predictions) * target_scale + target_mean
    return regression_metrics(raw_targets, predicted)


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    actual_direction = return_direction(targets)
    predicted_direction = return_direction(predictions)
    direction_counts = np.bincount(actual_direction + 1, minlength=3)
    directional_baseline_accuracy = float(direction_counts.max() / len(actual_direction))
    directional_accuracy_value = float(np.mean(actual_direction == predicted_direction))
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "directional_accuracy": directional_accuracy_value,
        "directional_baseline_accuracy": directional_baseline_accuracy,
        "directional_lift": float(directional_accuracy_value - directional_baseline_accuracy),
        "correlation": safe_correlation(targets, predictions),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions)),
    }


def evaluate_test_best(
    model_module,
    config,
    state_dict: dict[str, torch.Tensor],
    artifacts: dict[str, object],
    feature_count: int,
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model = model_module.build_model(config, input_size=feature_count).to(device)
    model.load_state_dict(state_dict)
    loader = DataLoader(
        TensorDataset(artifacts["x_test"], artifacts["y_test"]),
        batch_size=config.batch_size,
        shuffle=False,
    )
    raw_targets = np.asarray(artifacts["raw_y"])[artifacts["test_idx"]]
    metrics = evaluate_split(
        model=model,
        loader=loader,
        raw_targets=raw_targets,
        target_mean=float(artifacts["target_mean"]),
        target_scale=float(artifacts["target_scale"]),
    )

    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_x, _ in loader:
            predictions.append(model(batch_x).detach().cpu().numpy())
    predicted = np.concatenate(predictions) * float(artifacts["target_scale"]) + float(artifacts["target_mean"])

    meta = artifacts["meta"].iloc[artifacts["test_idx"]].reset_index(drop=True)
    prediction_frame = meta.copy()
    prediction_frame["actual_5d_return"] = raw_targets
    prediction_frame["predicted_5d_return"] = predicted
    prediction_frame["actual_direction"] = return_direction(raw_targets)
    prediction_frame["predicted_direction"] = return_direction(predicted)
    prediction_frame["absolute_error"] = np.abs(raw_targets - predicted)
    return metrics, prediction_frame


def config_from_row(model_module, row: dict):
    field_names = model_module.TrialConfig.__dataclass_fields__.keys()
    return model_module.TrialConfig(**{field: row[field] for field in field_names})


def directional_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(return_direction(targets) == return_direction(predictions)))


def return_direction(returns: np.ndarray) -> np.ndarray:
    return np.where(
        returns > tune_lstm.DIRECTION_FLAT_THRESHOLD,
        1,
        np.where(returns < -tune_lstm.DIRECTION_FLAT_THRESHOLD, -1, 0),
    ).astype(np.int64)


def safe_correlation(targets: np.ndarray, predictions: np.ndarray) -> float:
    if np.isclose(np.std(targets), 0.0) or np.isclose(np.std(predictions), 0.0):
        return float("nan")
    return float(np.corrcoef(targets, predictions)[0, 1])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
