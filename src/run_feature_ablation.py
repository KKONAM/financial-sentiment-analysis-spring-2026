from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train_gru
import train_lstm
import tune_gru
import tune_lstm
from features.build_features import (
    MARKET_FEATURES,
    SENTIMENT_FEATURES,
    combine_indicators_and_sentiment,
)
from temporal.gru import HybridGRU
from temporal.lstm import HybridLSTM


SYMBOLS = ["AAPL", "AMZN", "META", "NVDA", "TSLA"]
START_DATE = "2020-01-01"
END_DATE = "2022-02-28"
TARGET_HORIZON_DAYS = 5
TARGET_COLUMN = "future_return_5d"
TRAIN_END_DATE = "2021-04-30"
VAL_END_DATE = "2021-08-31"
EXPERIMENT_SLUG = "2020-01-01_to_2022-02-28_v5_purged_warmtech_finbert_sentiment"
DAILY_SENTIMENT_PATH = Path("data/processed/stocktwits_finbert_daily_sentiment.csv")
DATA_PATH = Path(
    f"data/processed/combined_features_AAPL_AMZN_META_NVDA_TSLA_{START_DATE}_to_{END_DATE}_"
    f"{TARGET_HORIZON_DAYS}d_target_v5_purged_warmtech_finbert_sentiment.csv"
)
SUMMARY_PATH = Path(f"reports/final/feature_ablation_{EXPERIMENT_SLUG}.json")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    sequence_length: int
    include_current_row: bool
    max_epochs: int
    patience: int
    min_delta: float
    batch_size: int
    learning_rate: float
    weight_decay: float
    loss_name: str
    smooth_l1_beta: float
    optimizer_name: str
    gradient_clip: float | None
    scale_target: bool
    seed: int
    build_model: Callable[[int], nn.Module]
    selection_score: Callable[[dict[str, float]], float]


def main() -> None:
    started_at = perf_counter()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = load_frame()

    feature_groups = {
        "technical_only": MARKET_FEATURES,
        "sentiment_only": SENTIMENT_FEATURES,
    }
    model_specs = [build_lstm_spec(), build_gru_spec()]

    print(f"[ablation] device={device} rows={len(frame):,}", flush=True)
    rows = []
    for spec in model_specs:
        for feature_group, feature_columns in feature_groups.items():
            print(f"[ablation] training model={spec.name} features={feature_group}", flush=True)
            result = run_experiment(
                frame=frame,
                spec=spec,
                feature_group=feature_group,
                feature_columns=feature_columns,
                device=device,
            )
            rows.append(result)
            test_metrics = result["test_metrics"]
            print(
                "[ablation] "
                f"model={spec.name} features={feature_group} "
                f"val_score={result['best_validation']['selection_score']:.6f} "
                f"test_mae={test_metrics['mae']:.6f} "
                f"test_rmse={test_metrics['rmse']:.6f} "
                f"test_dir={test_metrics['directional_accuracy']:.4f} "
                f"test_corr={test_metrics['correlation']:.4f}",
                flush=True,
            )

    summary = {
        "experiment": EXPERIMENT_SLUG,
        "data_path": str(DATA_PATH),
        "train_end_date": TRAIN_END_DATE,
        "val_end_date": VAL_END_DATE,
        "results": rows,
        "runtime_seconds": round(perf_counter() - started_at, 3),
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ablation] wrote {SUMMARY_PATH}", flush=True)


def load_frame() -> pd.DataFrame:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        frame = combine_indicators_and_sentiment(
            symbols=SYMBOLS,
            start_date=START_DATE,
            end_date=END_DATE,
            daily_sentiment_csv_path=DAILY_SENTIMENT_PATH,
            train_end=TRAIN_END_DATE,
            forecast_horizon=TARGET_HORIZON_DAYS,
            text_column="title",
            news_date_column="date",
            news_ticker_column="ticker",
        )
        frame.to_csv(DATA_PATH, index=False)
        return frame
    return pd.read_csv(DATA_PATH, parse_dates=["Date"])


def build_lstm_spec() -> ModelSpec:
    return ModelSpec(
        name="lstm",
        sequence_length=train_lstm.SEQUENCE_LENGTH,
        include_current_row=train_lstm.INCLUDE_CURRENT_ROW,
        max_epochs=train_lstm.EPOCHS,
        patience=train_lstm.EARLY_STOPPING_PATIENCE,
        min_delta=train_lstm.EARLY_STOPPING_MIN_DELTA,
        batch_size=train_lstm.BATCH_SIZE,
        learning_rate=train_lstm.LEARNING_RATE,
        weight_decay=train_lstm.WEIGHT_DECAY,
        loss_name=train_lstm.LOSS_NAME,
        smooth_l1_beta=train_lstm.SMOOTH_L1_BETA,
        optimizer_name=train_lstm.OPTIMIZER_NAME,
        gradient_clip=None,
        scale_target=train_lstm.SCALE_TARGET,
        seed=train_lstm.SEED,
        build_model=lambda input_size: HybridLSTM(
            input_size=input_size,
            hidden_size=train_lstm.HIDDEN_SIZE,
            num_layers=train_lstm.NUM_LAYERS,
            classifier_hidden=train_lstm.CLASSIFIER_HIDDEN,
            bidirectional=train_lstm.BIDIRECTIONAL,
            lstm_dropout=train_lstm.LSTM_DROPOUT,
            fc_dropout=train_lstm.FC_DROPOUT,
        ),
        selection_score=tune_lstm.selection_score,
    )


def build_gru_spec() -> ModelSpec:
    return ModelSpec(
        name="gru",
        sequence_length=train_gru.SEQUENCE_LENGTH,
        include_current_row=train_gru.INCLUDE_CURRENT_ROW,
        max_epochs=train_gru.EPOCHS,
        patience=train_gru.EARLY_STOPPING_PATIENCE,
        min_delta=train_gru.EARLY_STOPPING_MIN_DELTA,
        batch_size=train_gru.BATCH_SIZE,
        learning_rate=train_gru.LEARNING_RATE,
        weight_decay=train_gru.WEIGHT_DECAY,
        loss_name=train_gru.LOSS_NAME,
        smooth_l1_beta=train_gru.SMOOTH_L1_BETA,
        optimizer_name=train_gru.OPTIMIZER_NAME,
        gradient_clip=train_gru.GRADIENT_CLIP,
        scale_target=train_gru.SCALE_TARGET,
        seed=train_gru.SEED,
        build_model=lambda input_size: HybridGRU(
            input_size=input_size,
            hidden_size=train_gru.HIDDEN_SIZE,
            num_layers=train_gru.NUM_LAYERS,
            classifier_hidden=train_gru.CLASSIFIER_HIDDEN,
            head_layers=train_gru.HEAD_LAYERS,
            bidirectional=train_gru.BIDIRECTIONAL,
            gru_dropout=train_gru.GRU_DROPOUT,
            fc_dropout=train_gru.FC_DROPOUT,
            activation_name=train_gru.ACTIVATION_NAME,
            pooling_name=train_gru.POOLING_NAME,
            use_layer_norm=train_gru.USE_LAYER_NORM,
            gru_bias=train_gru.GRU_BIAS,
        ),
        selection_score=tune_gru.selection_score,
    )


def run_experiment(
    frame: pd.DataFrame,
    spec: ModelSpec,
    feature_group: str,
    feature_columns: list[str],
    device: torch.device,
) -> dict[str, object]:
    set_seed(spec.seed)
    train_cutoff = pd.to_datetime(TRAIN_END_DATE).normalize()
    val_cutoff = pd.to_datetime(VAL_END_DATE).normalize()

    scaled_frame, scaler = train_lstm.scale_features(
        df=frame,
        feature_columns=feature_columns,
        train_cutoff=train_cutoff,
    )
    X, y, meta = build_sequences(
        df=scaled_frame,
        sequence_length=spec.sequence_length,
        target_column=TARGET_COLUMN,
        include_current_row=spec.include_current_row,
        feature_columns=feature_columns,
    )
    train_idx, val_idx, test_idx = train_lstm.split_indices_by_date(
        meta=meta,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
    )

    target_mean, target_scale = fit_target_scaler(y=y, train_idx=train_idx, scale_target=spec.scale_target)
    model_y = scale_targets(y=y, target_mean=target_mean, target_scale=target_scale)

    loaders = {
        "train": make_loader(X, model_y, train_idx, spec.batch_size, device, shuffle=True),
        "val": make_loader(X, model_y, val_idx, spec.batch_size, device, shuffle=False),
        "test": make_loader(X, model_y, test_idx, spec.batch_size, device, shuffle=False),
    }

    model = spec.build_model(X.shape[2]).to(device)
    optimizer = build_optimizer(spec, model)
    loss_fn = build_loss(spec)

    best_score = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_val_metrics: dict[str, float] | None = None
    epochs_without_improvement = 0

    for epoch in range(1, spec.max_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            loss_fn=loss_fn,
            gradient_clip=spec.gradient_clip,
        )
        val_loss, val_predictions_scaled, _ = evaluate_loss(model, loaders["val"], loss_fn)
        val_predictions = inverse_scale_targets(val_predictions_scaled, target_mean, target_scale)
        val_targets = y[val_idx]
        val_metrics = regression_metrics(val_targets, val_predictions)
        val_metrics["loss"] = val_loss
        score = spec.selection_score(val_metrics)

        if score < best_score - spec.min_delta:
            best_score = score
            best_epoch = epoch
            best_val_metrics = val_metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            "[ablation] "
            f"model={spec.name} features={feature_group} epoch={epoch:02d}/{spec.max_epochs:02d} "
            f"train_loss={train_loss:.4f} val_score={score:.6f} best={best_score:.6f}",
            flush=True,
        )
        if epochs_without_improvement >= spec.patience:
            break

    if best_val_metrics is None:
        raise RuntimeError(f"No validation metrics produced for {spec.name} {feature_group}.")

    model.load_state_dict(best_state)
    test_loss, test_predictions_scaled, _ = evaluate_loss(model, loaders["test"], loss_fn)
    test_predictions = inverse_scale_targets(test_predictions_scaled, target_mean, target_scale)
    test_targets = y[test_idx]
    test_metrics = regression_metrics(test_targets, test_predictions)
    test_metrics["loss"] = test_loss

    prediction_frame = meta.iloc[test_idx].reset_index(drop=True)
    prediction_frame["actual_5d_return"] = test_targets
    prediction_frame["predicted_5d_return"] = test_predictions
    prediction_frame["actual_direction"] = return_direction(test_targets)
    prediction_frame["predicted_direction"] = return_direction(test_predictions)
    prediction_frame["absolute_error"] = np.abs(test_targets - test_predictions)

    prediction_path = Path(f"reports/final/{spec.name}_{feature_group}_ablation_predictions_{EXPERIMENT_SLUG}.csv")
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(prediction_path, index=False)

    checkpoint_path = Path(f"model_checkpoints/final/{spec.name}_{feature_group}_ablation_best_{EXPERIMENT_SLUG}.pt")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "model": spec.name,
            "feature_group": feature_group,
            "feature_columns": feature_columns,
            "model_config": asdict_without_callables(spec),
            "target_column": TARGET_COLUMN,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "feature_scaler_mean": scaler.mean_.tolist(),
            "feature_scaler_scale": scaler.scale_.tolist(),
            "train_cutoff": train_cutoff.isoformat(),
            "val_cutoff": val_cutoff.isoformat(),
            "validation_metrics": best_val_metrics,
            "test_metrics": test_metrics,
        },
        checkpoint_path,
    )

    return {
        "model": spec.name,
        "feature_group": feature_group,
        "feature_columns": feature_columns,
        "sequence_length": spec.sequence_length,
        "include_current_row": spec.include_current_row,
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "best_validation": {
            **best_val_metrics,
            "selection_score": best_score,
            "epoch": best_epoch,
        },
        "test_metrics": test_metrics,
        "paths": {
            "predictions": str(prediction_path),
            "checkpoint": str(checkpoint_path),
        },
    }


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader:
    features = torch.tensor(X[indices], dtype=torch.float32, device=device)
    targets = torch.tensor(y[indices], dtype=torch.float32, device=device)
    return DataLoader(TensorDataset(features, targets), batch_size=batch_size, shuffle=shuffle)


def build_sequences(
    df: pd.DataFrame,
    sequence_length: int,
    target_column: str,
    include_current_row: bool,
    feature_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    required_columns = [*feature_columns, target_column, "ticker", "Date"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns for sequence creation: {missing_columns}")

    features = []
    targets = []
    metadata = []
    for ticker, ticker_frame in df.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("Date").reset_index(drop=True)
        values = ticker_frame[feature_columns].to_numpy(dtype=np.float32)
        target_values = ticker_frame[target_column].to_numpy(dtype=np.float32)
        first_end = sequence_length - 1 if include_current_row else sequence_length
        for end_idx in range(first_end, len(ticker_frame)):
            start_idx = end_idx - sequence_length + 1 if include_current_row else end_idx - sequence_length
            end_slice = end_idx + 1 if include_current_row else end_idx
            features.append(values[start_idx:end_slice])
            targets.append(target_values[end_idx])
            metadata.append({"ticker": ticker, "Date": ticker_frame.loc[end_idx, "Date"]})

    if not features:
        raise ValueError("Not enough rows to create sequences.")
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), pd.DataFrame(metadata)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    gradient_clip: float | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        loss = loss_fn(model(batch_x), batch_y)
        loss.backward()
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += loss.item() * len(batch_x)
        total_rows += len(batch_x)
    return total_loss / total_rows


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    predictions = []
    targets = []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            predicted = model(batch_x)
            loss = loss_fn(predicted, batch_y)
            total_loss += loss.item() * len(batch_x)
            total_rows += len(batch_x)
            predictions.append(predicted.detach().cpu().numpy())
            targets.append(batch_y.detach().cpu().numpy())
    return total_loss / total_rows, np.concatenate(predictions), np.concatenate(targets)


def build_optimizer(spec: ModelSpec, model: nn.Module) -> torch.optim.Optimizer:
    if spec.optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    if spec.optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=spec.learning_rate, weight_decay=spec.weight_decay)
    raise ValueError(f"Unsupported optimizer: {spec.optimizer_name}")


def build_loss(spec: ModelSpec) -> nn.Module:
    if spec.loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=spec.smooth_l1_beta)
    if spec.loss_name == "mse":
        return nn.MSELoss()
    if spec.loss_name == "mae":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss: {spec.loss_name}")


def fit_target_scaler(y: np.ndarray, train_idx: np.ndarray, scale_target: bool) -> tuple[float, float]:
    if not scale_target:
        return 0.0, 1.0
    target_mean = float(np.mean(y[train_idx]))
    target_scale = float(np.std(y[train_idx]))
    if np.isclose(target_scale, 0.0):
        target_scale = 1.0
    return target_mean, target_scale


def scale_targets(y: np.ndarray, target_mean: float, target_scale: float) -> np.ndarray:
    return ((y - target_mean) / target_scale).astype(np.float32)


def inverse_scale_targets(y: np.ndarray, target_mean: float, target_scale: float) -> np.ndarray:
    return y * target_scale + target_mean


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "directional_accuracy": directional_accuracy(targets, predictions),
        "correlation": safe_correlation(targets, predictions),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions)),
    }


def directional_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(return_direction(targets) == return_direction(predictions)))


def return_direction(returns: np.ndarray) -> np.ndarray:
    return np.where(
        returns > train_lstm.DIRECTION_FLAT_THRESHOLD,
        1,
        np.where(returns < -train_lstm.DIRECTION_FLAT_THRESHOLD, -1, 0),
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


def asdict_without_callables(spec: ModelSpec) -> dict[str, object]:
    values = asdict(spec)
    values.pop("build_model")
    values.pop("selection_score")
    return values


if __name__ == "__main__":
    main()
