from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from features.build_features import combine_indicators_and_sentiment


SYMBOLS = ["AAPL", "AMZN", "META", "NVDA", "TSLA"]
START_DATE = "2020-01-01"
END_DATE = "2022-02-28"
FEATURE_COLUMNS = [
    "returns",
    "rsi",
    "macd",
    "bbp",
    "momentum",
    "volume",
    "positive_score",
    "negative_score",
    "neutral_score",
    "sentiment_score",
    "sentiment_confidence",
    "article_count",
    "message_count",
]
TARGET_COLUMN = "future_return_5d"
TARGET_HORIZON_DAYS = 5
TRAIN_END_DATE = "2021-04-30"
VAL_END_DATE = "2021-08-31"
DIRECTION_FLAT_THRESHOLD = 0.005
EXPERIMENT_SLUG = "2020-01-01_to_2022-02-28_v4_purged_rawtech_labeled_sentiment"
FINAL_REPORT_DIR = Path("reports/final")
FINAL_CHECKPOINT_DIR = Path("model_checkpoints/final")
DATA_PATH = Path(
    "data/processed/"
    f"combined_features_AAPL_AMZN_META_NVDA_TSLA_2020-01-01_to_2022-02-28_5d_target_"
    "v4_purged_rawtech_labeled_sentiment.csv"
)
SEARCH_RESULTS_PATH = FINAL_REPORT_DIR / f"lstm_combined_tuned_{EXPERIMENT_SLUG}_search.csv"
BEST_CONFIG_PATH = FINAL_REPORT_DIR / f"lstm_combined_tuned_{EXPERIMENT_SLUG}_best.json"
TUNED_PREDICTIONS_PATH = FINAL_REPORT_DIR / f"lstm_combined_tuned_{EXPERIMENT_SLUG}_predictions.csv"
TUNED_CHECKPOINT_PATH = FINAL_CHECKPOINT_DIR / f"lstm_combined_tuned_{EXPERIMENT_SLUG}_best.pt"
DEFAULT_TRAINING_SEED = 42
DEFAULT_TRIALS = 60
FIXED_INCLUDE_CURRENT_ROW = True
FIXED_BIDIRECTIONAL = False


@dataclass(frozen=True)
class TrialConfig:
    sequence_length: int
    include_current_row: bool
    hidden_size: int
    num_layers: int
    classifier_hidden: int
    head_layers: int
    bidirectional: bool
    lstm_dropout: float
    fc_dropout: float
    activation_name: str
    pooling_name: str
    use_layer_norm: bool
    learning_rate: float
    weight_decay: float
    batch_size: int
    loss_name: str
    smooth_l1_beta: float
    optimizer_name: str
    gradient_clip: float | None
    scale_target: bool
    seed: int


class TunableLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        classifier_hidden: int,
        head_layers: int,
        bidirectional: bool,
        lstm_dropout: float,
        fc_dropout: float,
        activation_name: str,
        pooling_name: str,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()
        recurrent_dropout = lstm_dropout if num_layers > 1 else 0.0
        self.pooling_name = pooling_name
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
        )
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        pooled_size = lstm_output_size * 2 if pooling_name == "last_mean" else lstm_output_size
        self.layer_norm = nn.LayerNorm(pooled_size) if use_layer_norm else nn.Identity()
        self.regressor = build_regressor(
            input_size=pooled_size,
            hidden_size=classifier_hidden,
            head_layers=head_layers,
            activation_name=activation_name,
            dropout=fc_dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(inputs)
        pooled = pool_sequence(output, self.pooling_name)
        pooled = self.layer_norm(pooled)
        return self.regressor(pooled).squeeze(-1)


def build_regressor(
    input_size: int,
    hidden_size: int,
    head_layers: int,
    activation_name: str,
    dropout: float,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_size = input_size
    for _ in range(head_layers):
        layers.append(nn.Linear(current_size, hidden_size))
        layers.append(build_activation(activation_name))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_size = hidden_size
    layers.append(nn.Linear(current_size, 1))
    return nn.Sequential(*layers)


def build_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def pool_sequence(output: torch.Tensor, pooling_name: str) -> torch.Tensor:
    if pooling_name == "last":
        return output[:, -1, :]
    if pooling_name == "mean":
        return output.mean(dim=1)
    if pooling_name == "last_mean":
        return torch.cat([output[:, -1, :], output.mean(dim=1)], dim=1)
    raise ValueError(f"Unsupported pooling: {pooling_name}")


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(args.torch_threads, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    started_at = perf_counter()
    frame = load_frame(args.data_path)
    train_cutoff = pd.to_datetime(TRAIN_END_DATE).normalize()
    val_cutoff = pd.to_datetime(VAL_END_DATE).normalize()
    scaled_frame, feature_scaler = scale_features(frame, train_cutoff)
    search_options = build_search_options()
    trial_configs = sample_configs(
        search_options,
        trials=args.trials,
        seed=args.search_seed,
        training_seed=args.training_seed,
    )

    print(
        f"[tune] device={device} trials={len(trial_configs)} "
        f"max_epochs={args.max_epochs} patience={args.patience}",
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
            "[tune] "
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
        raise RuntimeError("No tuning trials completed.")

    results = pd.DataFrame(rows).sort_values(["selection_score", "val_mae", "val_rmse"])
    args.search_results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.search_results_path, index=False)

    best_config = TrialConfig(**{field: best_row[field] for field in TrialConfig.__dataclass_fields__})
    test_metrics, predictions = evaluate_test_best(
        config=best_config,
        state_dict=best_state,
        artifacts=best_artifacts,
        device=device,
    )
    best_summary = {
        "best_validation": best_row,
        "test_metrics": test_metrics,
        "paths": {
            "search_results": str(args.search_results_path),
            "best_config": str(args.best_config_path),
            "predictions": str(args.tuned_predictions_path),
            "checkpoint": str(args.tuned_checkpoint_path),
        },
        "runtime_seconds": round(perf_counter() - started_at, 3),
    }

    args.best_config_path.parent.mkdir(parents=True, exist_ok=True)
    args.best_config_path.write_text(json.dumps(best_summary, indent=2), encoding="utf-8")
    args.tuned_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.tuned_predictions_path, index=False)
    args.tuned_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
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
        args.tuned_checkpoint_path,
    )

    print("[tune] best validation config:")
    print(json.dumps(best_row, indent=2))
    print("[tune] final test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"[tune] wrote {args.search_results_path}")
    print(f"[tune] wrote {args.best_config_path}")
    print(f"[tune] wrote {args.tuned_predictions_path}")
    print(f"[tune] wrote {args.tuned_checkpoint_path}")


def load_frame(data_path: Path) -> pd.DataFrame:
    if data_path.exists():
        return pd.read_csv(data_path, parse_dates=["Date"])

    data_path.parent.mkdir(parents=True, exist_ok=True)
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
    frame.to_csv(data_path, index=False)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-search LSTM hyperparameters on the validation split.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--search-results-path", type=Path, default=SEARCH_RESULTS_PATH)
    parser.add_argument("--best-config-path", type=Path, default=BEST_CONFIG_PATH)
    parser.add_argument("--tuned-predictions-path", type=Path, default=TUNED_PREDICTIONS_PATH)
    parser.add_argument("--tuned-checkpoint-path", type=Path, default=TUNED_CHECKPOINT_PATH)
    return parser.parse_args()


def build_search_options() -> dict[str, list[object]]:
    return {
        "sequence_length": [10, 20, 30, 45],
        "hidden_size": [24, 32, 48, 64, 96],
        "num_layers": [1, 2, 3],
        "classifier_hidden": [8, 16, 32],
        "head_layers": [1, 2],
        "lstm_dropout": [0.0, 0.1, 0.2],
        "fc_dropout": [0.0, 0.1, 0.2, 0.3],
        "activation_name": ["relu", "gelu", "tanh"],
        "pooling_name": ["last", "mean", "last_mean"],
        "use_layer_norm": [False, True],
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "weight_decay": [0.0, 1e-6, 1e-5, 1e-4],
        "batch_size": [16, 32, 64],
        "loss_name": ["smooth_l1", "mse", "mae"],
        "smooth_l1_beta": [0.05, 0.1, 0.2],
        "optimizer_name": ["adam", "adamw"],
        "gradient_clip": [None, 0.25, 0.5, 1.0],
        "scale_target": [True],
    }


def sample_configs(
    options: dict[str, list[object]],
    trials: int,
    seed: int,
    training_seed: int = DEFAULT_TRAINING_SEED,
) -> list[TrialConfig]:
    rng = random.Random(seed)
    sampled = curated_configs(training_seed=training_seed)
    seen = set(sampled)
    while len(sampled) < trials:
        loss_name = rng.choice(options["loss_name"])
        num_layers = rng.choice(options["num_layers"])
        hidden_size = rng.choice(options["hidden_size"])
        config = TrialConfig(
            sequence_length=rng.choice(options["sequence_length"]),
            include_current_row=FIXED_INCLUDE_CURRENT_ROW,
            hidden_size=hidden_size,
            num_layers=num_layers,
            classifier_hidden=rng.choice(options["classifier_hidden"]),
            head_layers=rng.choice(options["head_layers"]),
            bidirectional=FIXED_BIDIRECTIONAL,
            lstm_dropout=0.0 if num_layers == 1 else rng.choice(options["lstm_dropout"]),
            fc_dropout=rng.choice(options["fc_dropout"]),
            activation_name=rng.choice(options["activation_name"]),
            pooling_name=rng.choice(options["pooling_name"]),
            use_layer_norm=rng.choice(options["use_layer_norm"]),
            learning_rate=rng.choice(options["learning_rate"]),
            weight_decay=rng.choice(options["weight_decay"]),
            batch_size=rng.choice(options["batch_size"]),
            loss_name=loss_name,
            smooth_l1_beta=rng.choice(options["smooth_l1_beta"]) if loss_name == "smooth_l1" else 0.05,
            optimizer_name=rng.choice(options["optimizer_name"]),
            gradient_clip=rng.choice(options["gradient_clip"]),
            scale_target=rng.choice(options["scale_target"]),
            seed=training_seed,
        )
        if config in seen:
            continue
        sampled.append(config)
        seen.add(config)
    return sampled[:trials]


def curated_configs(training_seed: int = DEFAULT_TRAINING_SEED) -> list[TrialConfig]:
    baseline = TrialConfig(
        sequence_length=20,
        include_current_row=FIXED_INCLUDE_CURRENT_ROW,
        hidden_size=32,
        num_layers=2,
        classifier_hidden=16,
        head_layers=1,
        bidirectional=FIXED_BIDIRECTIONAL,
        lstm_dropout=0.1,
        fc_dropout=0.0,
        activation_name="relu",
        pooling_name="last",
        use_layer_norm=False,
        learning_rate=3e-4,
        weight_decay=1e-5,
        batch_size=32,
        loss_name="smooth_l1",
        smooth_l1_beta=0.1,
        optimizer_name="adam",
        gradient_clip=0.5,
        scale_target=True,
        seed=training_seed,
    )
    compact_fast = TrialConfig(
        sequence_length=10,
        include_current_row=FIXED_INCLUDE_CURRENT_ROW,
        hidden_size=32,
        num_layers=1,
        classifier_hidden=16,
        head_layers=1,
        bidirectional=FIXED_BIDIRECTIONAL,
        lstm_dropout=0.0,
        fc_dropout=0.1,
        activation_name="gelu",
        pooling_name="last",
        use_layer_norm=True,
        learning_rate=1e-3,
        weight_decay=1e-5,
        batch_size=32,
        loss_name="smooth_l1",
        smooth_l1_beta=0.1,
        optimizer_name="adamw",
        gradient_clip=1.0,
        scale_target=True,
        seed=training_seed,
    )
    wider_context = TrialConfig(
        sequence_length=30,
        include_current_row=FIXED_INCLUDE_CURRENT_ROW,
        hidden_size=64,
        num_layers=2,
        classifier_hidden=32,
        head_layers=2,
        bidirectional=FIXED_BIDIRECTIONAL,
        lstm_dropout=0.1,
        fc_dropout=0.1,
        activation_name="tanh",
        pooling_name="mean",
        use_layer_norm=True,
        learning_rate=3e-4,
        weight_decay=1e-5,
        batch_size=32,
        loss_name="smooth_l1",
        smooth_l1_beta=0.1,
        optimizer_name="adamw",
        gradient_clip=1.0,
        scale_target=True,
        seed=training_seed,
    )
    scaled_target = TrialConfig(
        sequence_length=45,
        include_current_row=FIXED_INCLUDE_CURRENT_ROW,
        hidden_size=64,
        num_layers=2,
        classifier_hidden=32,
        head_layers=2,
        bidirectional=FIXED_BIDIRECTIONAL,
        lstm_dropout=0.1,
        fc_dropout=0.2,
        activation_name="relu",
        pooling_name="last_mean",
        use_layer_norm=True,
        learning_rate=3e-4,
        weight_decay=1e-5,
        batch_size=32,
        loss_name="mae",
        smooth_l1_beta=0.1,
        optimizer_name="adamw",
        gradient_clip=1.0,
        scale_target=True,
        seed=training_seed,
    )
    return [baseline, compact_fast, wider_context, scaled_target]


def scale_features(frame: pd.DataFrame, train_cutoff: pd.Timestamp) -> tuple[pd.DataFrame, StandardScaler]:
    missing_columns = [column for column in [*FEATURE_COLUMNS, TARGET_COLUMN, "ticker", "Date"] if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    scaled = frame.copy()
    train_rows = scaled["Date"] <= train_cutoff
    scaler = StandardScaler()
    scaler.fit(scaled.loc[train_rows, FEATURE_COLUMNS])
    scaled[FEATURE_COLUMNS] = scaler.transform(scaled[FEATURE_COLUMNS])
    return scaled, scaler


def prepare_data(
    frame: pd.DataFrame,
    sequence_length: int,
    include_current_row: bool,
    train_cutoff: pd.Timestamp,
    val_cutoff: pd.Timestamp,
    scale_target: bool,
    device: torch.device,
) -> dict[str, object]:
    X, y, meta = build_sequences(
        frame=frame,
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
    if math.isclose(target_scale, 0.0):
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
    sequence_length: int,
    include_current_row: bool,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    features = []
    targets = []
    metadata = []
    for ticker, ticker_frame in frame.groupby("ticker"):
        ticker_frame = ticker_frame.sort_values("Date").reset_index(drop=True)
        values = ticker_frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        target_values = ticker_frame[TARGET_COLUMN].to_numpy(dtype=np.float32)
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

    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), pd.DataFrame(metadata)


def run_trial(
    config: TrialConfig,
    artifacts: dict[str, object],
    device: torch.device,
    max_epochs: int,
    patience: int,
    min_delta: float,
) -> tuple[dict[str, float | int | str | bool | None], dict[str, torch.Tensor]]:
    set_seed(config.seed)
    model = build_model(config, input_size=len(FEATURE_COLUMNS)).to(device)
    optimizer = build_optimizer(config, model)
    loss_fn = build_loss(config)
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
        **{f"val_{key}": value for key, value in best_metrics.items()},
        "selection_score": best_score,
        "epochs_ran": epoch,
    }
    return row, best_state


def build_model(config: TrialConfig, input_size: int) -> TunableLSTM:
    return TunableLSTM(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        classifier_hidden=config.classifier_hidden,
        head_layers=config.head_layers,
        bidirectional=config.bidirectional,
        lstm_dropout=config.lstm_dropout,
        fc_dropout=config.fc_dropout,
        activation_name=config.activation_name,
        pooling_name=config.pooling_name,
        use_layer_norm=config.use_layer_norm,
    )


def build_optimizer(config: TrialConfig, model: nn.Module) -> torch.optim.Optimizer:
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    if config.optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"Unsupported optimizer: {config.optimizer_name}")


def build_loss(config: TrialConfig) -> nn.Module:
    if config.loss_name == "smooth_l1":
        return nn.SmoothL1Loss(beta=config.smooth_l1_beta)
    if config.loss_name == "mse":
        return nn.MSELoss()
    if config.loss_name == "mae":
        return nn.L1Loss()
    raise ValueError(f"Unsupported loss: {config.loss_name}")


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
    return {
        "mae": float(mean_absolute_error(targets, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "directional_accuracy": float(np.mean(actual_direction == predicted_direction)),
        "directional_baseline_accuracy": directional_baseline_accuracy,
        "directional_lift": float(np.mean(actual_direction == predicted_direction) - directional_baseline_accuracy),
        "correlation": safe_correlation(targets, predictions),
        "prediction_mean": float(np.mean(predictions)),
        "prediction_std": float(np.std(predictions)),
    }


def selection_score(metrics: dict[str, float]) -> float:
    corr = metrics["correlation"]
    corr_reward = 0.0 if np.isnan(corr) else 0.004 * max(0.0, corr)
    corr_penalty = 0.0 if np.isnan(corr) else 0.002 * max(0.0, -corr)
    directional_reward = 0.02 * max(0.0, metrics["directional_lift"])
    directional_penalty = 0.01 * max(0.0, -metrics["directional_lift"])
    collapse_penalty = max(0.0, 0.01 - metrics["prediction_std"]) * 3.0
    return (
        metrics["rmse"]
        + 0.25 * metrics["mae"]
        + corr_penalty
        + directional_penalty
        + collapse_penalty
        - corr_reward
        - directional_reward
    )


def evaluate_test_best(
    config: TrialConfig,
    state_dict: dict[str, torch.Tensor],
    artifacts: dict[str, object],
    device: torch.device,
) -> tuple[dict[str, float], pd.DataFrame]:
    model = build_model(config, input_size=len(FEATURE_COLUMNS)).to(device)
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


def directional_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    return float(np.mean(return_direction(targets) == return_direction(predictions)))


def return_direction(returns: np.ndarray) -> np.ndarray:
    return np.where(
        returns > DIRECTION_FLAT_THRESHOLD,
        1,
        np.where(returns < -DIRECTION_FLAT_THRESHOLD, -1, 0),
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
