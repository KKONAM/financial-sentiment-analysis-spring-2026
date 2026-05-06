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

from temporal.transformer import HybridTransformer
import tune_lstm
from tune_lstm import (
    DATA_PATH,
    DEFAULT_TRAINING_SEED,
    DIRECTION_FLAT_THRESHOLD,
    FEATURE_COLUMNS,
    FINAL_CHECKPOINT_DIR,
    FINAL_REPORT_DIR,
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
    return_direction,
    scale_features,
    selection_score,
    set_seed,
)


DEFAULT_TRIALS = 60
SEARCH_RESULTS_PATH = FINAL_REPORT_DIR / f"transformer_combined_tuned_{EXPERIMENT_SLUG}_search.csv"
BEST_CONFIG_PATH = FINAL_REPORT_DIR / f"transformer_combined_tuned_{EXPERIMENT_SLUG}_best.json"
TUNED_PREDICTIONS_PATH = FINAL_REPORT_DIR / f"transformer_combined_tuned_{EXPERIMENT_SLUG}_predictions.csv"
TUNED_CHECKPOINT_PATH = FINAL_CHECKPOINT_DIR / f"transformer_combined_tuned_{EXPERIMENT_SLUG}_best.pt"


@dataclass(frozen=True)
class TrialConfig:
    sequence_length: int
    include_current_row: bool
    d_model: int
    n_heads: int
    num_layers: int
    dim_feedforward: int
    classifier_hidden: int
    head_layers: int
    transformer_dropout: float
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


def main() -> None:
    args = parse_args()
    tune_lstm.TARGET_HORIZON_DAYS = args.target_horizon_days
    torch.set_num_threads(max(1, min(args.torch_threads, torch.get_num_threads())))
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    started_at = perf_counter()
    frame = load_frame(args.data_path)
    train_cutoff = pd.to_datetime(TRAIN_END_DATE).normalize()
    val_cutoff = pd.to_datetime(VAL_END_DATE).normalize()
    scaled_frame, feature_scaler = scale_features(frame, train_cutoff)
    trial_configs = sample_configs(
        build_search_options(),
        trials=args.trials,
        seed=args.search_seed,
        training_seed=args.training_seed,
    )

    print(
        f"[tune_transformer] device={device} trials={len(trial_configs)} "
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
            "[tune_transformer] "
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
        raise RuntimeError("No transformer tuning trials completed.")

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

    print("[tune_transformer] best validation config:")
    print(json.dumps(best_row, indent=2))
    print("[tune_transformer] final test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"[tune_transformer] wrote {args.search_results_path}")
    print(f"[tune_transformer] wrote {args.best_config_path}")
    print(f"[tune_transformer] wrote {args.tuned_predictions_path}")
    print(f"[tune_transformer] wrote {args.tuned_checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random-search Transformer hyperparameters on the validation split.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--search-seed", type=int, default=2026)
    parser.add_argument("--training-seed", type=int, default=DEFAULT_TRAINING_SEED)
    parser.add_argument("--target-horizon-days", type=int, default=tune_lstm.TARGET_HORIZON_DAYS)
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
        "d_model": [32, 48, 64, 96],
        "n_heads": [2, 3, 4, 6, 8],
        "num_layers": [1, 2, 3],
        "dim_feedforward": [64, 128, 192, 256],
        "classifier_hidden": [8, 16, 32],
        "head_layers": [1, 2],
        "transformer_dropout": [0.0, 0.1, 0.2],
        "fc_dropout": [0.0, 0.1, 0.2, 0.3],
        "activation_name": ["relu", "gelu"],
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
        d_model = rng.choice(options["d_model"])
        compatible_heads = [heads for heads in options["n_heads"] if d_model % heads == 0]
        config = TrialConfig(
            sequence_length=rng.choice(options["sequence_length"]),
            include_current_row=FIXED_INCLUDE_CURRENT_ROW,
            d_model=d_model,
            n_heads=rng.choice(compatible_heads),
            num_layers=rng.choice(options["num_layers"]),
            dim_feedforward=rng.choice(options["dim_feedforward"]),
            classifier_hidden=rng.choice(options["classifier_hidden"]),
            head_layers=rng.choice(options["head_layers"]),
            transformer_dropout=rng.choice(options["transformer_dropout"]),
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
    return [
        TrialConfig(
            sequence_length=20,
            include_current_row=FIXED_INCLUDE_CURRENT_ROW,
            d_model=64,
            n_heads=4,
            num_layers=2,
            dim_feedforward=128,
            classifier_hidden=32,
            head_layers=1,
            transformer_dropout=0.1,
            fc_dropout=0.1,
            activation_name="gelu",
            pooling_name="mean",
            use_layer_norm=True,
            learning_rate=3e-4,
            weight_decay=1e-5,
            batch_size=32,
            loss_name="smooth_l1",
            smooth_l1_beta=0.1,
            optimizer_name="adamw",
            gradient_clip=0.5,
            scale_target=True,
            seed=training_seed,
        ),
        TrialConfig(
            sequence_length=10,
            include_current_row=FIXED_INCLUDE_CURRENT_ROW,
            d_model=32,
            n_heads=4,
            num_layers=1,
            dim_feedforward=64,
            classifier_hidden=16,
            head_layers=1,
            transformer_dropout=0.1,
            fc_dropout=0.0,
            activation_name="gelu",
            pooling_name="last",
            use_layer_norm=True,
            learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size=32,
            loss_name="mae",
            smooth_l1_beta=0.1,
            optimizer_name="adam",
            gradient_clip=1.0,
            scale_target=True,
            seed=training_seed,
        ),
        TrialConfig(
            sequence_length=30,
            include_current_row=FIXED_INCLUDE_CURRENT_ROW,
            d_model=96,
            n_heads=6,
            num_layers=2,
            dim_feedforward=192,
            classifier_hidden=32,
            head_layers=2,
            transformer_dropout=0.2,
            fc_dropout=0.2,
            activation_name="relu",
            pooling_name="last_mean",
            use_layer_norm=True,
            learning_rate=3e-4,
            weight_decay=1e-4,
            batch_size=32,
            loss_name="smooth_l1",
            smooth_l1_beta=0.1,
            optimizer_name="adamw",
            gradient_clip=0.5,
            scale_target=True,
            seed=training_seed,
        ),
    ]


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


def build_model(config: TrialConfig, input_size: int) -> HybridTransformer:
    return HybridTransformer(
        input_size=input_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        classifier_hidden=config.classifier_hidden,
        head_layers=config.head_layers,
        transformer_dropout=config.transformer_dropout,
        fc_dropout=config.fc_dropout,
        activation_name=config.activation_name,
        pooling_name=config.pooling_name,
        use_layer_norm=config.use_layer_norm,
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
    prediction_frame[f"actual_{tune_lstm.TARGET_HORIZON_DAYS}d_return"] = raw_targets
    prediction_frame[f"predicted_{tune_lstm.TARGET_HORIZON_DAYS}d_return"] = predicted
    prediction_frame["actual_direction"] = return_direction(raw_targets)
    prediction_frame["predicted_direction"] = return_direction(predicted)
    prediction_frame["absolute_error"] = np.abs(raw_targets - predicted)
    return metrics, prediction_frame


if __name__ == "__main__":
    main()
