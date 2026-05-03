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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tune_lstm import (
    DATA_PATH,
    DEFAULT_TRIALS,
    DEFAULT_TRAINING_SEED,
    DIRECTION_FLAT_THRESHOLD,
    FEATURE_COLUMNS,
    FINAL_CHECKPOINT_DIR,
    FINAL_REPORT_DIR,
    FIXED_BIDIRECTIONAL,
    FIXED_INCLUDE_CURRENT_ROW,
    EXPERIMENT_SLUG,
    TARGET_COLUMN,
    TRAIN_END_DATE,
    VAL_END_DATE,
    build_loss,
    build_optimizer,
    evaluate_split,
    load_frame,
    prepare_data,
    scale_features,
    set_seed,
)


SEARCH_RESULTS_PATH = FINAL_REPORT_DIR / f"gru_combined_tuned_{EXPERIMENT_SLUG}_search.csv"
BEST_CONFIG_PATH = FINAL_REPORT_DIR / f"gru_combined_tuned_{EXPERIMENT_SLUG}_best.json"
TUNED_PREDICTIONS_PATH = FINAL_REPORT_DIR / f"gru_combined_tuned_{EXPERIMENT_SLUG}_predictions.csv"
TUNED_CHECKPOINT_PATH = FINAL_CHECKPOINT_DIR / f"gru_combined_tuned_{EXPERIMENT_SLUG}_best.pt"


@dataclass(frozen=True)
class TrialConfig:
    sequence_length: int
    include_current_row: bool
    hidden_size: int
    num_layers: int
    classifier_hidden: int
    head_layers: int
    bidirectional: bool
    gru_dropout: float
    fc_dropout: float
    activation_name: str
    pooling_name: str
    use_layer_norm: bool
    gru_bias: bool
    learning_rate: float
    weight_decay: float
    batch_size: int
    loss_name: str
    smooth_l1_beta: float
    optimizer_name: str
    gradient_clip: float | None
    scale_target: bool
    seed: int


class TunableGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        classifier_hidden: int,
        head_layers: int,
        bidirectional: bool,
        gru_dropout: float,
        fc_dropout: float,
        activation_name: str,
        pooling_name: str,
        use_layer_norm: bool,
        gru_bias: bool,
    ) -> None:
        super().__init__()
        recurrent_dropout = gru_dropout if num_layers > 1 else 0.0
        self.pooling_name = pooling_name
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=bidirectional,
            bias=gru_bias,
        )

        gru_output_size = hidden_size * (2 if bidirectional else 1)
        pooled_size = gru_output_size * 2 if pooling_name == "last_mean" else gru_output_size
        self.layer_norm = nn.LayerNorm(pooled_size) if use_layer_norm else nn.Identity()
        self.regressor = build_regressor(
            input_size=pooled_size,
            hidden_size=classifier_hidden,
            head_layers=head_layers,
            activation_name=activation_name,
            dropout=fc_dropout,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(inputs)
        pooled = pool_sequence(output, self.pooling_name)
        pooled = self.layer_norm(pooled)
        return self.regressor(pooled).squeeze(-1)


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
        f"[tune_gru] device={device} trials={len(trial_configs)} "
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
            "[tune_gru] "
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

    best_config = TrialConfig(**{field.name: best_row[field.name] for field in fields(TrialConfig)})
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

    print("[tune_gru] best validation config:")
    print(json.dumps(best_row, indent=2))
    print("[tune_gru] final test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"[tune_gru] wrote {args.search_results_path}")
    print(f"[tune_gru] wrote {args.best_config_path}")
    print(f"[tune_gru] wrote {args.tuned_predictions_path}")
    print(f"[tune_gru] wrote {args.tuned_checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-search GRU hyperparameters on the validation split.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
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
        "gru_dropout": [0.0, 0.1, 0.2],
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
        head_layers = rng.choice(options["head_layers"])
        pooling_name = rng.choice(options["pooling_name"])
        config = TrialConfig(
            sequence_length=rng.choice(options["sequence_length"]),
            include_current_row=FIXED_INCLUDE_CURRENT_ROW,
            hidden_size=hidden_size,
            num_layers=num_layers,
            classifier_hidden=rng.choice(options["classifier_hidden"]),
            head_layers=head_layers,
            bidirectional=FIXED_BIDIRECTIONAL,
            gru_dropout=0.0 if num_layers == 1 else rng.choice(options["gru_dropout"]),
            fc_dropout=rng.choice(options["fc_dropout"]),
            activation_name=rng.choice(options["activation_name"]),
            pooling_name=pooling_name,
            use_layer_norm=rng.choice(options["use_layer_norm"]),
            gru_bias=True,
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
        if too_large(config):
            continue
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
        gru_dropout=0.1,
        fc_dropout=0.0,
        activation_name="relu",
        pooling_name="last",
        use_layer_norm=False,
        gru_bias=True,
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
        gru_dropout=0.0,
        fc_dropout=0.1,
        activation_name="gelu",
        pooling_name="last",
        use_layer_norm=True,
        gru_bias=True,
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
        gru_dropout=0.1,
        fc_dropout=0.1,
        activation_name="tanh",
        pooling_name="mean",
        use_layer_norm=True,
        gru_bias=True,
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
    long_context = TrialConfig(
        sequence_length=45,
        include_current_row=FIXED_INCLUDE_CURRENT_ROW,
        hidden_size=64,
        num_layers=2,
        classifier_hidden=32,
        head_layers=2,
        bidirectional=FIXED_BIDIRECTIONAL,
        gru_dropout=0.1,
        fc_dropout=0.2,
        activation_name="relu",
        pooling_name="last_mean",
        use_layer_norm=True,
        gru_bias=True,
        learning_rate=3e-4,
        weight_decay=1e-5,
        batch_size=32,
        loss_name="mae",
        smooth_l1_beta=0.05,
        optimizer_name="adamw",
        gradient_clip=1.0,
        scale_target=True,
        seed=training_seed,
    )
    compact_mean = TrialConfig(
        sequence_length=20,
        include_current_row=FIXED_INCLUDE_CURRENT_ROW,
        hidden_size=24,
        num_layers=1,
        classifier_hidden=8,
        head_layers=1,
        bidirectional=FIXED_BIDIRECTIONAL,
        gru_dropout=0.0,
        fc_dropout=0.1,
        activation_name="tanh",
        pooling_name="mean",
        use_layer_norm=True,
        gru_bias=True,
        learning_rate=1e-3,
        weight_decay=0.0,
        batch_size=32,
        loss_name="mae",
        smooth_l1_beta=0.1,
        optimizer_name="adamw",
        gradient_clip=0.25,
        scale_target=True,
        seed=training_seed,
    )
    return [baseline, compact_fast, wider_context, long_context, compact_mean]


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


def build_model(config: TrialConfig, input_size: int) -> TunableGRU:
    return TunableGRU(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        classifier_hidden=config.classifier_hidden,
        head_layers=config.head_layers,
        bidirectional=config.bidirectional,
        gru_dropout=config.gru_dropout,
        fc_dropout=config.fc_dropout,
        activation_name=config.activation_name,
        pooling_name=config.pooling_name,
        use_layer_norm=config.use_layer_norm,
        gru_bias=config.gru_bias,
    )


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
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


def pool_sequence(output: torch.Tensor, pooling_name: str) -> torch.Tensor:
    if pooling_name == "last":
        return output[:, -1, :]
    if pooling_name == "mean":
        return output.mean(dim=1)
    if pooling_name == "max":
        return output.max(dim=1).values
    if pooling_name == "last_mean":
        return torch.cat([output[:, -1, :], output.mean(dim=1)], dim=1)
    raise ValueError(f"Unsupported pooling: {pooling_name}")


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


def return_direction(returns: np.ndarray) -> np.ndarray:
    return np.where(
        returns > DIRECTION_FLAT_THRESHOLD,
        1,
        np.where(returns < -DIRECTION_FLAT_THRESHOLD, -1, 0),
    ).astype(np.int64)


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


def too_large(config: TrialConfig) -> bool:
    output_multiplier = 2 if config.bidirectional else 1
    pooled_multiplier = 2 if config.pooling_name == "last_mean" else 1
    rough_width = config.hidden_size * output_multiplier * pooled_multiplier
    return config.num_layers == 4 and rough_width >= 192 and config.sequence_length >= 45


if __name__ == "__main__":
    main()
