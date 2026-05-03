from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import tune_lstm
from features.build_features import MARKET_FEATURES, SENTIMENT_FEATURES, combine_indicators_and_sentiment
from temporal.gru import HybridGRUMultitask
from temporal.lstm import HybridLSTMMultitask
from temporal.multitask import multitask_loss


SYMBOLS = ["AAPL", "AMZN", "META", "NVDA", "TSLA"]
START_DATE = "2020-01-01"
END_DATE = "2022-02-28"
TARGET_HORIZON_DAYS = 5
TARGET_COLUMN = "future_return_5d"
TRAIN_END_DATE = "2021-04-30"
VAL_END_DATE = "2021-08-31"
DIRECTION_FLAT_THRESHOLD = 0.005
EXPERIMENT_SLUG = f"{tune_lstm.EXPERIMENT_SLUG}_multitask"
DATA_PATH = tune_lstm.DATA_PATH
FINAL_REPORT_DIR = tune_lstm.FINAL_REPORT_DIR
FINAL_CHECKPOINT_DIR = tune_lstm.FINAL_CHECKPOINT_DIR
DEFAULT_TRIALS = 60
DEFAULT_TRAINING_SEED = 42
FIXED_INCLUDE_CURRENT_ROW = True
FIXED_BIDIRECTIONAL = False
FEATURE_GROUPS = {
    "combined": tune_lstm.FEATURE_COLUMNS,
    "sentiment_only": SENTIMENT_FEATURES,
}


@dataclass(frozen=True)
class TrialConfig:
    sequence_length: int
    include_current_row: bool
    hidden_size: int
    num_layers: int
    classifier_hidden: int
    head_layers: int
    bidirectional: bool
    recurrent_dropout: float
    fc_dropout: float
    activation_name: str
    pooling_name: str
    use_layer_norm: bool
    gru_bias: bool
    learning_rate: float
    weight_decay: float
    batch_size: int
    regression_loss_name: str
    smooth_l1_beta: float
    direction_loss_weight: float
    class_weight_mode: str
    optimizer_name: str
    gradient_clip: float | None
    scale_target: bool
    seed: int


def main() -> None:
    args = parse_args()
    feature_columns = FEATURE_GROUPS[args.feature_group]
    torch.set_num_threads(max(1, min(args.torch_threads, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    started_at = perf_counter()
    frame = load_frame(args.data_path)
    train_cutoff = pd.to_datetime(TRAIN_END_DATE).normalize()
    val_cutoff = pd.to_datetime(VAL_END_DATE).normalize()
    scaled_frame, feature_scaler = scale_features(frame, feature_columns, train_cutoff)
    trial_configs = sample_configs(
        build_search_options(),
        trials=args.trials,
        seed=args.search_seed,
        training_seed=args.training_seed,
    )

    print(
        f"[multitask] model={args.model} features={args.feature_group} "
        f"device={device} trials={len(trial_configs)} max_epochs={args.max_epochs}",
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
            feature_columns=feature_columns,
            sequence_length=config.sequence_length,
            include_current_row=config.include_current_row,
            train_cutoff=train_cutoff,
            val_cutoff=val_cutoff,
            scale_target=config.scale_target,
            device=device,
        )
        row, state_dict = run_trial(
            model_name=args.model,
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
            "[multitask] "
            f"trial={trial_number:03d}/{len(trial_configs):03d} "
            f"score={row['selection_score']:.6f} "
            f"val_rmse={row['val_rmse']:.6f} "
            f"val_dir_acc={row['val_direction_accuracy']:.4f} "
            f"val_macro_f1={row['val_direction_macro_f1']:.4f} "
            f"epochs={row['epochs_ran']:02d} "
            f"best={best_row['selection_score']:.6f}",
            flush=True,
        )

    if best_row is None or best_state is None or best_artifacts is None:
        raise RuntimeError("No multitask tuning trials completed.")

    results = pd.DataFrame(rows).sort_values(["selection_score", "val_direction_macro_f1", "val_rmse"])
    search_results_path, best_config_path, predictions_path, checkpoint_path = output_paths(
        model=args.model,
        feature_group=args.feature_group,
        output_suffix=args.output_suffix,
    )
    search_results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(search_results_path, index=False)

    best_config = TrialConfig(**{field.name: best_row[field.name] for field in fields(TrialConfig)})
    test_metrics, predictions = evaluate_test_best(
        model_name=args.model,
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

    print("[multitask] best validation config:")
    print(json.dumps(best_row, indent=2))
    print("[multitask] final test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"[multitask] wrote {search_results_path}")
    print(f"[multitask] wrote {best_config_path}")
    print(f"[multitask] wrote {predictions_path}")
    print(f"[multitask] wrote {checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune multitask LSTM/GRU return + direction models.")
    parser.add_argument("--model", choices=["lstm", "gru"], required=True)
    parser.add_argument("--feature-group", choices=sorted(FEATURE_GROUPS), required=True)
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-suffix", default=EXPERIMENT_SLUG)
    return parser.parse_args()


def output_paths(model: str, feature_group: str, output_suffix: str) -> tuple[Path, Path, Path, Path]:
    stem = f"{model}_{feature_group}_multitask_tuned_{output_suffix}"
    return (
        FINAL_REPORT_DIR / f"{stem}_search.csv",
        FINAL_REPORT_DIR / f"{stem}_best.json",
        FINAL_REPORT_DIR / f"{stem}_predictions.csv",
        FINAL_CHECKPOINT_DIR / f"{stem}_best.pt",
    )


def build_search_options() -> dict[str, list[object]]:
    return {
        "sequence_length": [10, 20, 30, 45],
        "hidden_size": [24, 32, 48, 64, 96],
        "num_layers": [1, 2, 3],
        "classifier_hidden": [8, 16, 32],
        "head_layers": [1, 2],
        "recurrent_dropout": [0.0, 0.1, 0.2],
        "fc_dropout": [0.0, 0.1, 0.2, 0.3],
        "activation_name": ["relu", "gelu", "tanh"],
        "pooling_name": ["last", "mean", "last_mean"],
        "use_layer_norm": [False, True],
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
        "batch_size": [16, 32, 64],
        "regression_loss_name": ["smooth_l1", "mse", "mae"],
        "smooth_l1_beta": [0.05, 0.1, 0.2],
        "direction_loss_weight": [0.1, 0.25, 0.5, 1.0, 2.0],
        "class_weight_mode": ["none", "balanced"],
        "optimizer_name": ["adam", "adamw"],
        "gradient_clip": [None, 0.25, 0.5, 1.0],
        "scale_target": [True],
    }


def sample_configs(
    options: dict[str, list[object]],
    trials: int,
    seed: int,
    training_seed: int,
) -> list[TrialConfig]:
    rng = random.Random(seed)
    sampled = curated_configs(training_seed)
    seen = set(sampled)
    while len(sampled) < trials:
        loss_name = rng.choice(options["regression_loss_name"])
        num_layers = rng.choice(options["num_layers"])
        config = TrialConfig(
            sequence_length=rng.choice(options["sequence_length"]),
            include_current_row=FIXED_INCLUDE_CURRENT_ROW,
            hidden_size=rng.choice(options["hidden_size"]),
            num_layers=num_layers,
            classifier_hidden=rng.choice(options["classifier_hidden"]),
            head_layers=rng.choice(options["head_layers"]),
            bidirectional=FIXED_BIDIRECTIONAL,
            recurrent_dropout=0.0 if num_layers == 1 else rng.choice(options["recurrent_dropout"]),
            fc_dropout=rng.choice(options["fc_dropout"]),
            activation_name=rng.choice(options["activation_name"]),
            pooling_name=rng.choice(options["pooling_name"]),
            use_layer_norm=rng.choice(options["use_layer_norm"]),
            gru_bias=True,
            learning_rate=rng.choice(options["learning_rate"]),
            weight_decay=rng.choice(options["weight_decay"]),
            batch_size=rng.choice(options["batch_size"]),
            regression_loss_name=loss_name,
            smooth_l1_beta=rng.choice(options["smooth_l1_beta"]) if loss_name == "smooth_l1" else 0.1,
            direction_loss_weight=rng.choice(options["direction_loss_weight"]),
            class_weight_mode=rng.choice(options["class_weight_mode"]),
            optimizer_name=rng.choice(options["optimizer_name"]),
            gradient_clip=rng.choice(options["gradient_clip"]),
            scale_target=rng.choice(options["scale_target"]),
            seed=training_seed,
        )
        if too_large(config) or config in seen:
            continue
        sampled.append(config)
        seen.add(config)
    return sampled[:trials]


def curated_configs(training_seed: int) -> list[TrialConfig]:
    return [
        TrialConfig(
            sequence_length=10,
            include_current_row=FIXED_INCLUDE_CURRENT_ROW,
            hidden_size=64,
            num_layers=1,
            classifier_hidden=8,
            head_layers=1,
            bidirectional=FIXED_BIDIRECTIONAL,
            recurrent_dropout=0.0,
            fc_dropout=0.3,
            activation_name="tanh",
            pooling_name="mean",
            use_layer_norm=False,
            gru_bias=True,
            learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size=32,
            regression_loss_name="mae",
            smooth_l1_beta=0.1,
            direction_loss_weight=0.5,
            class_weight_mode="balanced",
            optimizer_name="adam",
            gradient_clip=0.5,
            scale_target=True,
            seed=training_seed,
        ),
        TrialConfig(
            sequence_length=20,
            include_current_row=FIXED_INCLUDE_CURRENT_ROW,
            hidden_size=32,
            num_layers=2,
            classifier_hidden=32,
            head_layers=1,
            bidirectional=FIXED_BIDIRECTIONAL,
            recurrent_dropout=0.1,
            fc_dropout=0.1,
            activation_name="gelu",
            pooling_name="last",
            use_layer_norm=False,
            gru_bias=True,
            learning_rate=1e-3,
            weight_decay=1e-4,
            batch_size=64,
            regression_loss_name="mae",
            smooth_l1_beta=0.1,
            direction_loss_weight=1.0,
            class_weight_mode="balanced",
            optimizer_name="adamw",
            gradient_clip=1.0,
            scale_target=True,
            seed=training_seed,
        ),
        TrialConfig(
            sequence_length=30,
            include_current_row=FIXED_INCLUDE_CURRENT_ROW,
            hidden_size=24,
            num_layers=1,
            classifier_hidden=32,
            head_layers=2,
            bidirectional=FIXED_BIDIRECTIONAL,
            recurrent_dropout=0.0,
            fc_dropout=0.1,
            activation_name="gelu",
            pooling_name="last_mean",
            use_layer_norm=True,
            gru_bias=True,
            learning_rate=3e-4,
            weight_decay=1e-4,
            batch_size=32,
            regression_loss_name="mae",
            smooth_l1_beta=0.1,
            direction_loss_weight=0.25,
            class_weight_mode="none",
            optimizer_name="adam",
            gradient_clip=None,
            scale_target=True,
            seed=training_seed,
        ),
        TrialConfig(
            sequence_length=20,
            include_current_row=FIXED_INCLUDE_CURRENT_ROW,
            hidden_size=64,
            num_layers=1,
            classifier_hidden=16,
            head_layers=2,
            bidirectional=FIXED_BIDIRECTIONAL,
            recurrent_dropout=0.0,
            fc_dropout=0.3,
            activation_name="relu",
            pooling_name="mean",
            use_layer_norm=True,
            gru_bias=True,
            learning_rate=3e-4,
            weight_decay=1e-5,
            batch_size=16,
            regression_loss_name="mse",
            smooth_l1_beta=0.1,
            direction_loss_weight=2.0,
            class_weight_mode="balanced",
            optimizer_name="adam",
            gradient_clip=1.0,
            scale_target=True,
            seed=training_seed,
        ),
    ]


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
    required = [*feature_columns, TARGET_COLUMN, "ticker", "Date"]
    missing_columns = [column for column in required if column not in frame.columns]
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
    model_y = ((y - target_mean) / target_scale).astype(np.float32)
    direction_y = direction_classes(y)

    return {
        "X": X,
        "raw_y": y,
        "model_y": model_y,
        "direction_y": direction_y,
        "meta": meta,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "x_train": torch.tensor(X[train_idx], dtype=torch.float32, device=device),
        "y_train": torch.tensor(model_y[train_idx], dtype=torch.float32, device=device),
        "d_train": torch.tensor(direction_y[train_idx], dtype=torch.long, device=device),
        "x_val": torch.tensor(X[val_idx], dtype=torch.float32, device=device),
        "y_val": torch.tensor(model_y[val_idx], dtype=torch.float32, device=device),
        "d_val": torch.tensor(direction_y[val_idx], dtype=torch.long, device=device),
        "x_test": torch.tensor(X[test_idx], dtype=torch.float32, device=device),
        "y_test": torch.tensor(model_y[test_idx], dtype=torch.float32, device=device),
        "d_test": torch.tensor(direction_y[test_idx], dtype=torch.long, device=device),
    }


def build_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    include_current_row: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features = []
    targets = []
    metadata = []
    for ticker, ticker_frame in frame.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("Date").reset_index(drop=True)
        values = ticker_frame[feature_columns].to_numpy(dtype=np.float32)
        target_values = ticker_frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
        first_end = sequence_length - 1 if include_current_row else sequence_length
        for end_idx in range(first_end, len(ticker_frame)):
            start_idx = end_idx - sequence_length + 1 if include_current_row else end_idx - sequence_length
            end_slice = end_idx + 1 if include_current_row else end_idx
            target_end_idx = end_idx + TARGET_HORIZON_DAYS
            target_end_date = ticker_frame.loc[target_end_idx, "Date"] if target_end_idx < len(ticker_frame) else pd.NaT
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
    model_name: str,
    config: TrialConfig,
    artifacts: dict[str, object],
    feature_count: int,
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float,
) -> tuple[dict[str, float | int | str | bool | None], dict[str, torch.Tensor]]:
    set_seed(config.seed)
    model = build_model(model_name, config, input_size=feature_count).to(device)
    optimizer = build_optimizer(config, model)
    regression_loss_fn = build_regression_loss(config)
    direction_loss_fn = build_direction_loss(config, artifacts, device)
    train_loader = DataLoader(
        TensorDataset(artifacts["x_train"], artifacts["y_train"], artifacts["d_train"]),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(artifacts["x_val"], artifacts["y_val"], artifacts["d_val"]),
        batch_size=config.batch_size,
        shuffle=False,
    )

    best_score = float("inf")
    best_metrics = None
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y, batch_d in train_loader:
            optimizer.zero_grad()
            output = model(batch_x)
            losses = multitask_loss(
                return_prediction=output.return_prediction,
                direction_logits=output.direction_logits,
                return_targets=batch_y,
                direction_targets=batch_d,
                regression_loss_fn=regression_loss_fn,
                direction_loss_fn=direction_loss_fn,
                direction_loss_weight=config.direction_loss_weight,
            )
            losses.total_loss.backward()
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
        score = selection_score(metrics)

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
        **{f"val_{key}": value for key, value in best_metrics.items() if not isinstance(value, list)},
        "selection_score": best_score,
        "epochs_ran": epoch,
    }
    return row, best_state


def build_model(model_name: str, config: TrialConfig, input_size: int) -> nn.Module:
    kwargs = {
        "input_size": input_size,
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "classifier_hidden": config.classifier_hidden,
        "head_layers": config.head_layers,
        "bidirectional": config.bidirectional,
        "fc_dropout": config.fc_dropout,
        "activation_name": config.activation_name,
        "pooling_name": config.pooling_name,
        "use_layer_norm": config.use_layer_norm,
    }
    if model_name == "lstm":
        return HybridLSTMMultitask(lstm_dropout=config.recurrent_dropout, **kwargs)
    if model_name == "gru":
        return HybridGRUMultitask(gru_dropout=config.recurrent_dropout, gru_bias=config.gru_bias, **kwargs)
    raise ValueError(f"Unsupported model: {model_name}")


def build_optimizer(config: TrialConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


def build_regression_loss(config: TrialConfig) -> nn.Module:
    if config.regression_loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=config.smooth_l1_beta)
    if config.regression_loss_name == "mse":
        return nn.MSELoss()
    if config.regression_loss_name == "mae":
        return nn.L1Loss()
    raise ValueError(f"Unsupported regression loss: {config.regression_loss_name}")


def build_direction_loss(config: TrialConfig, artifacts: dict[str, object], device: torch.device) -> nn.Module:
    if config.class_weight_mode == "none":
        return nn.CrossEntropyLoss()
    if config.class_weight_mode == "balanced":
        train_labels = np.asarray(artifacts["direction_y"])[artifacts["train_idx"]]
        counts = np.bincount(train_labels, minlength=3).astype(np.float32)
        weights = len(train_labels) / (3.0 * np.maximum(counts, 1.0))
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    raise ValueError(f"Unsupported class weight mode: {config.class_weight_mode}")


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    raw_targets: np.ndarray,
    target_mean: float,
    target_scale: float,
) -> dict[str, float | list[list[int]]]:
    model.eval()
    return_predictions = []
    direction_predictions = []
    with torch.no_grad():
        for batch_x, _, _ in loader:
            output = model(batch_x)
            return_predictions.append(output.return_prediction.detach().cpu().numpy())
            direction_predictions.append(output.direction_logits.argmax(dim=1).detach().cpu().numpy())
    predicted_returns = np.concatenate(return_predictions) * target_scale + target_mean
    predicted_direction_classes = np.concatenate(direction_predictions)
    return multitask_metrics(raw_targets, predicted_returns, predicted_direction_classes)


def multitask_metrics(
    targets: np.ndarray,
    return_predictions: np.ndarray,
    direction_predictions: np.ndarray,
) -> dict[str, float | list[list[int]]]:
    actual_direction = direction_classes(targets)
    threshold_direction = direction_classes(return_predictions)
    direction_counts = np.bincount(actual_direction, minlength=3)
    directional_baseline_accuracy = float(direction_counts.max() / len(actual_direction))
    direction_accuracy = float(np.mean(actual_direction == direction_predictions))
    threshold_direction_accuracy = float(np.mean(actual_direction == threshold_direction))
    corr = safe_correlation(targets, return_predictions)
    return {
        "mae": float(mean_absolute_error(targets, return_predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, return_predictions))),
        "correlation": corr,
        "prediction_mean": float(np.mean(return_predictions)),
        "prediction_std": float(np.std(return_predictions)),
        "threshold_direction_accuracy": threshold_direction_accuracy,
        "direction_accuracy": direction_accuracy,
        "direction_baseline_accuracy": directional_baseline_accuracy,
        "direction_lift": float(direction_accuracy - directional_baseline_accuracy),
        "direction_macro_f1": float(f1_score(actual_direction, direction_predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "direction_weighted_f1": float(f1_score(actual_direction, direction_predictions, labels=[0, 1, 2], average="weighted", zero_division=0)),
        "direction_macro_precision": float(precision_score(actual_direction, direction_predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "direction_macro_recall": float(recall_score(actual_direction, direction_predictions, labels=[0, 1, 2], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual_direction, direction_predictions, labels=[0, 1, 2]).astype(int).tolist(),
    }


def selection_score(metrics: dict[str, float | list[list[int]]]) -> float:
    corr = float(metrics["correlation"])
    corr_reward = 0.0 if np.isnan(corr) else 0.003 * max(0.0, corr)
    corr_penalty = 0.0 if np.isnan(corr) else 0.002 * max(0.0, -corr)
    direction_lift = float(metrics["direction_lift"])
    macro_f1 = float(metrics["direction_macro_f1"])
    direction_reward = 0.03 * max(0.0, direction_lift) + 0.015 * macro_f1
    direction_penalty = 0.015 * max(0.0, -direction_lift)
    collapse_penalty = max(0.0, 0.01 - float(metrics["prediction_std"])) * 2.0
    return (
        float(metrics["rmse"])
        + 0.20 * float(metrics["mae"])
        + corr_penalty
        + direction_penalty
        + collapse_penalty
        - corr_reward
        - direction_reward
    )


def evaluate_test_best(
    model_name: str,
    config: TrialConfig,
    state_dict: dict[str, torch.Tensor],
    artifacts: dict[str, object],
    feature_count: int,
    device: torch.device,
) -> tuple[dict[str, float | list[list[int]]], pd.DataFrame]:
    model = build_model(model_name, config, input_size=feature_count).to(device)
    model.load_state_dict(state_dict)
    loader = DataLoader(
        TensorDataset(artifacts["x_test"], artifacts["y_test"], artifacts["d_test"]),
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
    return_predictions = []
    direction_predictions = []
    with torch.no_grad():
        for batch_x, _, _ in loader:
            output = model(batch_x)
            return_predictions.append(output.return_prediction.detach().cpu().numpy())
            direction_predictions.append(output.direction_logits.argmax(dim=1).detach().cpu().numpy())
    predicted_returns = np.concatenate(return_predictions) * float(artifacts["target_scale"]) + float(artifacts["target_mean"])
    predicted_direction_classes = np.concatenate(direction_predictions)

    meta = artifacts["meta"].iloc[artifacts["test_idx"]].reset_index(drop=True)
    prediction_frame = meta.copy()
    prediction_frame["actual_5d_return"] = raw_targets
    prediction_frame["predicted_5d_return"] = predicted_returns
    prediction_frame["actual_direction"] = direction_classes(raw_targets) - 1
    prediction_frame["predicted_direction_from_return"] = direction_classes(predicted_returns) - 1
    prediction_frame["predicted_direction"] = predicted_direction_classes - 1
    prediction_frame["absolute_error"] = np.abs(raw_targets - predicted_returns)
    return metrics, prediction_frame


def direction_classes(returns: np.ndarray) -> np.ndarray:
    return np.where(
        returns > DIRECTION_FLAT_THRESHOLD,
        2,
        np.where(returns < -DIRECTION_FLAT_THRESHOLD, 0, 1),
    ).astype(np.int64)


def safe_correlation(targets: np.ndarray, predictions: np.ndarray) -> float:
    if np.isclose(np.std(targets), 0.0) or np.isclose(np.std(predictions), 0.0):
        return float("nan")
    return float(np.corrcoef(targets, predictions)[0, 1])


def too_large(config: TrialConfig) -> bool:
    pooled_multiplier = 2 if config.pooling_name == "last_mean" else 1
    rough_width = config.hidden_size * pooled_multiplier
    return config.num_layers >= 3 and rough_width >= 192 and config.sequence_length >= 45


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
