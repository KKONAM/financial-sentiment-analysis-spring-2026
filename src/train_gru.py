from pathlib import Path
import random
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from features.build_features import (
    MARKET_FEATURES,
    SENTIMENT_FEATURES,
    build_lstm_sequences,
    combine_indicators_and_sentiment,
)
from temporal.gru import HybridGRU


SEQUENCE_LENGTH = 20
TARGET_HORIZON_DAYS = 5
TARGET_COLUMN = "future_return_5d"
INCLUDE_CURRENT_ROW = True
DIRECTION_FLAT_THRESHOLD = 0.005
TRAIN_END_DATE = "2021-04-30"
VAL_END_DATE = "2021-08-31"
EPOCHS = 80
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-6
HIDDEN_SIZE = 24
NUM_LAYERS = 4
CLASSIFIER_HIDDEN = 8
HEAD_LAYERS = 2
BIDIRECTIONAL = False
GRU_DROPOUT = 0.0
FC_DROPOUT = 0.3
ACTIVATION_NAME = "tanh"
POOLING_NAME = "mean"
USE_LAYER_NORM = True
GRU_BIAS = True
LOSS_NAME = "smooth_l1"
OPTIMIZER_NAME = "adamw"
SMOOTH_L1_BETA = 0.2
GRADIENT_CLIP = 0.25
SCALE_TARGET = True
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-5
SEED = 101
CHECKPOINT_PATH = Path(f"model_checkpoints/final/gru_{TARGET_HORIZON_DAYS}d_return_best.pt")
PREDICTIONS_PATH = Path(f"reports/final/gru_{TARGET_HORIZON_DAYS}d_return_predictions.csv")
PROGRESS_EVERY_BATCHES = 10
CACHE_VERSION = "v4_purged_rawtech_labeled_sentiment"


def main():
    started_at = perf_counter()
    set_seed(SEED)
    symbols = ["AAPL", "AMZN", "META", "NVDA", "TSLA"]
    start_date = "2020-01-01"
    end_date = "2022-02-28"
    combined_features_path = build_combined_features_path(symbols, start_date, end_date, TARGET_HORIZON_DAYS)

    log_step("Loading or building combined indicator + sentiment dataframe")
    if combined_features_path.exists():
        log_step(f"Loading combined features from {combined_features_path}")
        combined_df = pd.read_csv(combined_features_path, parse_dates=["Date"])
    else:
        log_step("Building combined indicator + sentiment dataframe")
        combined_df = combine_indicators_and_sentiment(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            train_end=TRAIN_END_DATE,
            forecast_horizon=TARGET_HORIZON_DAYS,
            text_column="title",
            news_date_column="date",
            news_ticker_column="ticker",
        )

        combined_features_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(combined_features_path, index=False)
        log_step(f"Saved combined features to {combined_features_path}")

    log_step(f"Built dataframe with {len(combined_df):,} rows")

    log_step("Building raw sequence metadata for date split")
    _, _, raw_meta = build_lstm_sequences(
        df=combined_df,
        sequence_length=SEQUENCE_LENGTH,
        target_column=TARGET_COLUMN,
        include_current_row=INCLUDE_CURRENT_ROW,
    )
    train_cutoff, val_cutoff = get_date_cutoffs()
    log_step(f"Date split cutoffs: train <= {train_cutoff.date()}, val <= {val_cutoff.date()}")

    log_step("Scaling features using train period only")
    feature_columns = MARKET_FEATURES + SENTIMENT_FEATURES
    scaled_df, scaler = scale_features(
        df=combined_df,
        feature_columns=feature_columns,
        train_cutoff=train_cutoff,
    )

    log_step("Building scaled GRU sequences")
    X, y, meta = build_lstm_sequences(
        df=scaled_df,
        sequence_length=SEQUENCE_LENGTH,
        target_column=TARGET_COLUMN,
        include_current_row=INCLUDE_CURRENT_ROW,
    )

    train_idx, val_idx, test_idx = split_indices_by_date(
        meta=meta,
        train_cutoff=train_cutoff,
        val_cutoff=val_cutoff,
    )
    target_mean, target_scale = fit_target_scaler(y=y, train_idx=train_idx)
    model_y = scale_targets(y=y, target_mean=target_mean, target_scale=target_scale)

    log_step(
        "Split sizes: "
        f"train={len(train_idx):,}, validation={len(val_idx):,}, test={len(test_idx):,}"
    )
    if SCALE_TARGET:
        log_step(f"Scaling target using train mean={target_mean:.6f}, std={target_scale:.6f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_step(f"Using device: {device}")
    x_train, y_train = to_tensors(X, model_y, train_idx, device)
    x_val, y_val = to_tensors(X, model_y, val_idx, device)
    x_test, y_test = to_tensors(X, model_y, test_idx, device)

    test_meta = meta.iloc[test_idx].reset_index(drop=True)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = HybridGRU(
        input_size=X.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        classifier_hidden=CLASSIFIER_HIDDEN,
        head_layers=HEAD_LAYERS,
        bidirectional=BIDIRECTIONAL,
        gru_dropout=GRU_DROPOUT,
        fc_dropout=FC_DROPOUT,
        activation_name=ACTIVATION_NAME,
        pooling_name=POOLING_NAME,
        use_layer_norm=USE_LAYER_NORM,
        gru_bias=GRU_BIAS,
    ).to(device)
    optimizer = build_optimizer(model)
    loss_fn = build_loss()

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, EPOCHS + 1):
        epoch_start = perf_counter()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epoch=epoch,
        )
        log_step(f"Epoch {epoch:02d}: running validation")
        val_loss, _, _ = evaluate(model, val_loader, loss_fn)

        saved_checkpoint = False
        if val_loss < best_val_loss - EARLY_STOPPING_MIN_DELTA:
            best_val_loss = val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                val_loss=val_loss,
                feature_columns=feature_columns,
                train_cutoff=train_cutoff,
                val_cutoff=val_cutoff,
                target_column=TARGET_COLUMN,
                target_horizon_days=TARGET_HORIZON_DAYS,
                hidden_size=HIDDEN_SIZE,
                classifier_hidden=CLASSIFIER_HIDDEN,
                include_current_row=INCLUDE_CURRENT_ROW,
                loss_name=LOSS_NAME,
                optimizer_name=OPTIMIZER_NAME,
                smooth_l1_beta=SMOOTH_L1_BETA,
                gradient_clip=GRADIENT_CLIP,
                scale_target=SCALE_TARGET,
                target_mean=target_mean,
                target_scale=target_scale,
                seed=SEED,
            )
            saved_checkpoint = True
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint_note = " saved_checkpoint" if saved_checkpoint else ""
        print(
            f"epoch={epoch:02d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"time={format_seconds(perf_counter() - epoch_start)}"
            f"{checkpoint_note}",
            flush=True,
        )

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            log_step(
                "Early stopping: "
                f"validation loss did not improve for {EARLY_STOPPING_PATIENCE} epochs"
            )
            break

    log_step("Loading best validation checkpoint for final test evaluation")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    log_step("Running final test evaluation")
    test_loss, scaled_test_predictions, _ = evaluate(model, test_loader, loss_fn)
    test_predictions = inverse_scale_targets(
        y=scaled_test_predictions,
        target_mean=target_mean,
        target_scale=target_scale,
    )
    test_targets = y[test_idx]
    test_mae = mean_absolute_error(test_targets, test_predictions)
    test_rmse = float(np.sqrt(mean_squared_error(test_targets, test_predictions)))
    test_directional_accuracy = directional_accuracy(test_targets, test_predictions)
    baseline_rows = build_baseline_rows(
        train_targets=y[train_idx],
        test_targets=test_targets,
        model_predictions=test_predictions,
    )

    predictions = test_meta.copy()
    predictions["actual_5d_return"] = test_targets
    predictions["predicted_5d_return"] = test_predictions
    predictions["actual_direction"] = return_direction(test_targets)
    predictions["predicted_direction"] = return_direction(test_predictions)
    predictions["absolute_error"] = np.abs(test_targets - test_predictions)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(f"best checkpoint: {CHECKPOINT_PATH}")
    print(
        f"test_loss={test_loss:.6f} "
        f"test_mae={test_mae:.6f} "
        f"test_rmse={test_rmse:.6f} "
        f"directional_accuracy={test_directional_accuracy:.4f}"
    )
    print_baseline_report(baseline_rows)
    print(f"predictions saved to: {PREDICTIONS_PATH}")
    print(f"total runtime: {format_seconds(perf_counter() - started_at)}")
    print(predictions.head())


def build_combined_features_path(
    symbols: list[str],
    start_date: str,
    end_date: str,
    target_horizon_days: int,
) -> Path:
    symbol_slug = "_".join(symbol.upper() for symbol in symbols)
    return Path(
        "data/processed/"
        f"combined_features_{symbol_slug}_{start_date}_to_{end_date}_{target_horizon_days}d_target_{CACHE_VERSION}.csv"
    )


def get_date_cutoffs() -> tuple[pd.Timestamp, pd.Timestamp]:
    train_cutoff = pd.to_datetime(TRAIN_END_DATE).normalize()
    val_cutoff = pd.to_datetime(VAL_END_DATE).normalize()
    if train_cutoff >= val_cutoff:
        raise ValueError("TRAIN_END_DATE must be before VAL_END_DATE.")
    return train_cutoff, val_cutoff


def scale_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    train_cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, StandardScaler]:
    scaled_df = df.copy()
    scaled_df["Date"] = pd.to_datetime(scaled_df["Date"])

    train_rows = scaled_df["Date"] <= train_cutoff
    scaler = StandardScaler()
    scaler.fit(scaled_df.loc[train_rows, feature_columns])
    scaled_df[feature_columns] = scaler.transform(scaled_df[feature_columns])
    return scaled_df, scaler


def split_indices_by_date(
    meta: pd.DataFrame,
    train_cutoff: pd.Timestamp,
    val_cutoff: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(meta["Date"])
    train_idx = np.flatnonzero(dates <= train_cutoff)
    val_idx = np.flatnonzero((dates > train_cutoff) & (dates <= val_cutoff))
    test_idx = np.flatnonzero(dates > val_cutoff)

    if min(len(train_idx), len(val_idx), len(test_idx)) == 0:
        raise ValueError(
            "Date split produced an empty train, validation, or test set. "
            "Use a longer date range or adjust split ratios."
        )
    return train_idx, val_idx, test_idx


def to_tensors(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(X[indices], dtype=torch.float32, device=device)
    targets = torch.tensor(y[indices], dtype=torch.float32, device=device)
    return features, targets


def fit_target_scaler(y: np.ndarray, train_idx: np.ndarray) -> tuple[float, float]:
    if not SCALE_TARGET:
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


def train_one_epoch(
    model: HybridGRU,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0

    for batch_idx, (batch_x, batch_y) in enumerate(loader, start=1):
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        if GRADIENT_CLIP is not None:
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        total_loss += loss.item() * len(batch_x)
        total_rows += len(batch_x)

        if batch_idx == 1 or batch_idx == len(loader) or batch_idx % PROGRESS_EVERY_BATCHES == 0:
            print(
                f"epoch={epoch:02d}/{EPOCHS} "
                f"batch={batch_idx:03d}/{len(loader):03d} "
                f"running_train_loss={total_loss / total_rows:.4f}",
                flush=True,
            )

    return total_loss / total_rows


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    if OPTIMIZER_NAME == "adam":
        return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if OPTIMIZER_NAME == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")


def build_loss() -> nn.Module:
    if LOSS_NAME == "smooth_l1":
        return nn.SmoothL1Loss(beta=SMOOTH_L1_BETA)
    if LOSS_NAME == "mse":
        return nn.MSELoss()
    if LOSS_NAME == "mae":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss: {LOSS_NAME}")


def evaluate(
    model: HybridGRU,
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
            predicted_returns = model(batch_x)
            loss = loss_fn(predicted_returns, batch_y)

            total_loss += loss.item() * len(batch_x)
            total_rows += len(batch_x)
            predictions.append(predicted_returns.cpu().numpy())
            targets.append(batch_y.cpu().numpy())

    return total_loss / total_rows, np.concatenate(predictions), np.concatenate(targets)


def directional_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(return_direction(targets) == return_direction(predictions)))


def return_direction(returns: np.ndarray) -> np.ndarray:
    return np.where(
        returns > DIRECTION_FLAT_THRESHOLD,
        1,
        np.where(returns < -DIRECTION_FLAT_THRESHOLD, -1, 0),
    ).astype(np.int64)


def build_baseline_rows(
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    model_predictions: np.ndarray,
) -> list[dict[str, float | str]]:
    train_mean = float(np.mean(train_targets))
    baselines = {
        "model": model_predictions,
        "zero_return": np.zeros_like(test_targets),
        "train_mean_return": np.full_like(test_targets, train_mean),
    }

    rows = [regression_summary(name, predictions, test_targets) for name, predictions in baselines.items()]
    rows.append(
        {
            "name": "always_up_direction",
            "mae": np.nan,
            "rmse": np.nan,
            "directional_accuracy": float(np.mean(return_direction(test_targets) == 1)),
            "correlation": np.nan,
        }
    )
    rows.append(
        {
            "name": "always_down_direction",
            "mae": np.nan,
            "rmse": np.nan,
            "directional_accuracy": float(np.mean(return_direction(test_targets) == -1)),
            "correlation": np.nan,
        }
    )
    rows.append(
        {
            "name": "always_flat_direction",
            "mae": np.nan,
            "rmse": np.nan,
            "directional_accuracy": float(np.mean(return_direction(test_targets) == 0)),
            "correlation": np.nan,
        }
    )
    return rows


def regression_summary(
    name: str,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float | str]:
    return {
        "name": name,
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "directional_accuracy": directional_accuracy(targets, predictions),
        "correlation": safe_correlation(targets, predictions),
    }


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


def print_baseline_report(rows: list[dict[str, float | str]]) -> None:
    report = pd.DataFrame(rows)
    print("baseline comparison:")
    print(report.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def save_checkpoint(
    model: HybridGRU,
    optimizer: torch.optim.Optimizer,
    scaler: StandardScaler,
    epoch: int,
    val_loss: float,
    feature_columns: list[str],
    train_cutoff: pd.Timestamp,
    val_cutoff: pd.Timestamp,
    target_column: str,
    target_horizon_days: int,
    hidden_size: int,
    classifier_hidden: int,
    include_current_row: bool,
    loss_name: str,
    optimizer_name: str,
    smooth_l1_beta: float,
    gradient_clip: float | None,
    scale_target: bool,
    target_mean: float,
    target_scale: float,
    seed: int,
) -> None:
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_size": len(feature_columns),
            "sequence_length": SEQUENCE_LENGTH,
            "feature_columns": feature_columns,
            "target_column": target_column,
            "target_horizon_days": target_horizon_days,
            "hidden_size": hidden_size,
            "num_layers": NUM_LAYERS,
            "classifier_hidden": classifier_hidden,
            "head_layers": HEAD_LAYERS,
            "bidirectional": BIDIRECTIONAL,
            "gru_dropout": GRU_DROPOUT,
            "fc_dropout": FC_DROPOUT,
            "activation_name": ACTIVATION_NAME,
            "pooling_name": POOLING_NAME,
            "use_layer_norm": USE_LAYER_NORM,
            "gru_bias": GRU_BIAS,
            "include_current_row": include_current_row,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss_name": loss_name,
            "smooth_l1_beta": smooth_l1_beta,
            "optimizer_name": optimizer_name,
            "gradient_clip": gradient_clip,
            "scale_target": scale_target,
            "target_mean": target_mean,
            "target_scale": target_scale,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "seed": seed,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "train_cutoff": train_cutoff.isoformat(),
            "val_cutoff": val_cutoff.isoformat(),
        },
        CHECKPOINT_PATH,
    )


def log_step(message: str) -> None:
    print(f"[train_gru] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


if __name__ == "__main__":
    main()
