from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_SLUG = "2020-01-01_to_2022-02-28_v6_1d_purged_warmtech_finetuned_finbert_sentiment_sidequest"
DATA_PATH = Path(
    "data/processed/"
    "combined_features_AAPL_AMZN_META_NVDA_TSLA_2020-01-01_to_2022-02-28_1d_target_"
    "v6_purged_warmtech_finetuned_finbert_sentiment.csv"
)
DAILY_SENTIMENT_PATH = Path("data/processed/stocktwits_finetuned_finbert_daily_sentiment.csv")
REPORT_DIR = Path("reports/final")
CHECKPOINT_DIR = Path("model_checkpoints/final")
LOG_DIR = REPORT_DIR / "tuning_v6_1d_logs"
STATUS_PATH = LOG_DIR / "status.json"
SUMMARY_CSV_PATH = REPORT_DIR / f"tuned_v6_1d_architecture_feature_grid_summary.csv"
SUMMARY_JSON_PATH = REPORT_DIR / f"tuned_v6_1d_architecture_feature_grid_summary.json"
TARGET_HORIZON_DAYS = 1


@dataclass(frozen=True)
class TuningJob:
    name: str
    command: list[str]
    expected_report: Path
    expected_checkpoint: Path


def build_jobs(python_executable: str, trials: int, torch_threads: int) -> list[TuningJob]:
    common = [
        "--trials",
        str(trials),
        "--torch-threads",
        str(torch_threads),
        "--data-path",
        str(DATA_PATH),
        "--target-horizon-days",
        str(TARGET_HORIZON_DAYS),
    ]
    jobs = []

    for model, script in [
        ("lstm", "src/tune_lstm.py"),
        ("gru", "src/tune_gru.py"),
        ("transformer", "src/tune_transformer.py"),
    ]:
        stem = f"{model}_combined_tuned_{EXPERIMENT_SLUG}"
        search_path = REPORT_DIR / f"{stem}_search.csv"
        best_path = REPORT_DIR / f"{stem}_best.json"
        predictions_path = REPORT_DIR / f"{stem}_predictions.csv"
        checkpoint_path = CHECKPOINT_DIR / f"{stem}_best.pt"
        jobs.append(
            TuningJob(
                name=f"{model}_combined",
                command=[
                    python_executable,
                    script,
                    *common,
                    "--search-results-path",
                    str(search_path),
                    "--best-config-path",
                    str(best_path),
                    "--tuned-predictions-path",
                    str(predictions_path),
                    "--tuned-checkpoint-path",
                    str(checkpoint_path),
                ],
                expected_report=best_path,
                expected_checkpoint=checkpoint_path,
            )
        )

    for model in ["lstm", "gru", "transformer"]:
        for feature_group in ["sentiment_only", "technical_only"]:
            stem = f"{model}_{feature_group}_tuned_{EXPERIMENT_SLUG}"
            jobs.append(
                TuningJob(
                    name=f"{model}_{feature_group}",
                    command=[
                        python_executable,
                        "src/tune_feature_subset.py",
                        "--model",
                        model,
                        "--feature-group",
                        feature_group,
                        *common,
                        "--output-suffix",
                        EXPERIMENT_SLUG,
                    ],
                    expected_report=REPORT_DIR / f"{stem}_best.json",
                    expected_checkpoint=CHECKPOINT_DIR / f"{stem}_best.pt",
                )
            )
    return jobs


def main() -> None:
    args = parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    ensure_dataset()

    jobs = build_jobs(
        python_executable=args.python_executable,
        trials=args.trials,
        torch_threads=args.torch_threads,
    )
    if not args.force:
        jobs = [
            job
            for job in jobs
            if not (job.expected_report.exists() and job.expected_checkpoint.exists())
        ]

    status: dict[str, dict[str, object]] = {
        job.name: {
            "status": "pending",
            "command": job.command,
            "expected_report": str(job.expected_report),
            "expected_checkpoint": str(job.expected_checkpoint),
            "stdout": str(LOG_DIR / f"{job.name}.stdout.log"),
            "stderr": str(LOG_DIR / f"{job.name}.stderr.log"),
        }
        for job in jobs
    }
    write_status(status)

    pending = list(jobs)
    running: dict[str, tuple[TuningJob, subprocess.Popen[bytes], object, object, float]] = {}
    completed = 0
    failed = 0

    while pending or running:
        while pending and len(running) < args.max_parallel:
            job = pending.pop(0)
            stdout_path = LOG_DIR / f"{job.name}.stdout.log"
            stderr_path = LOG_DIR / f"{job.name}.stderr.log"
            stdout_handle = stdout_path.open("wb")
            stderr_handle = stderr_path.open("wb")
            process = subprocess.Popen(
                job.command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                cwd=Path.cwd(),
            )
            running[job.name] = (job, process, stdout_handle, stderr_handle, time.time())
            status[job.name]["status"] = "running"
            status[job.name]["pid"] = process.pid
            status[job.name]["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            write_status(status)
            print(f"[runner_1d] started {job.name} pid={process.pid}", flush=True)

        time.sleep(args.poll_seconds)

        for name, (job, process, stdout_handle, stderr_handle, started_at) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue

            stdout_handle.close()
            stderr_handle.close()
            elapsed = round(time.time() - started_at, 3)
            status[name]["status"] = "completed" if return_code == 0 else "failed"
            status[name]["return_code"] = return_code
            status[name]["elapsed_seconds"] = elapsed
            status[name]["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            status[name]["report_exists"] = job.expected_report.exists()
            status[name]["checkpoint_exists"] = job.expected_checkpoint.exists()
            write_status(status)
            del running[name]

            if return_code == 0:
                completed += 1
                print(f"[runner_1d] completed {name} in {elapsed:.1f}s", flush=True)
            else:
                failed += 1
                print(f"[runner_1d] failed {name} rc={return_code} in {elapsed:.1f}s", flush=True)
                if args.stop_on_failure:
                    pending.clear()

    print(f"[runner_1d] finished completed={completed} failed={failed}", flush=True)
    if failed:
        raise SystemExit(1)

    write_summary(build_jobs(args.python_executable, args.trials, args.torch_threads))


def ensure_dataset() -> None:
    if DATA_PATH.exists():
        print(f"[runner_1d] using existing dataset {DATA_PATH}", flush=True)
        return

    sys.path.insert(0, str(Path("src").resolve()))
    from features.build_features import combine_indicators_and_sentiment

    print(f"[runner_1d] building dataset {DATA_PATH}", flush=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = combine_indicators_and_sentiment(
        symbols=["AAPL", "AMZN", "META", "NVDA", "TSLA"],
        start_date="2020-01-01",
        end_date="2022-02-28",
        daily_sentiment_csv_path=DAILY_SENTIMENT_PATH,
        train_end="2021-04-30",
        forecast_horizon=TARGET_HORIZON_DAYS,
        text_column="title",
        news_date_column="date",
        news_ticker_column="ticker",
    )
    frame.to_csv(DATA_PATH, index=False)
    print(f"[runner_1d] wrote {DATA_PATH} shape={frame.shape}", flush=True)


def write_summary(jobs: list[TuningJob]) -> None:
    rows = []
    for job in jobs:
        if not job.expected_report.exists():
            continue
        summary = json.loads(job.expected_report.read_text(encoding="utf-8"))
        validation = summary["best_validation"]
        test = summary["test_metrics"]
        model, features = parse_job_name(job.name)
        rows.append(
            {
                "model": model,
                "features": features,
                "val_mae": validation["val_mae"],
                "val_rmse": validation["val_rmse"],
                "val_directional_accuracy": validation["val_directional_accuracy"],
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
                "source": str(job.expected_report),
            }
        )

    order = {
        ("LSTM", "Combined"): 0,
        ("LSTM", "Sentiment only"): 1,
        ("LSTM", "Technical only"): 2,
        ("GRU", "Combined"): 3,
        ("GRU", "Sentiment only"): 4,
        ("GRU", "Technical only"): 5,
        ("Transformer", "Combined"): 6,
        ("Transformer", "Sentiment only"): 7,
        ("Transformer", "Technical only"): 8,
    }
    rows.sort(key=lambda row: order[(row["model"], row["features"])])
    SUMMARY_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_JSON_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[runner_1d] wrote {SUMMARY_CSV_PATH}", flush=True)
    print(f"[runner_1d] wrote {SUMMARY_JSON_PATH}", flush=True)


def parse_job_name(name: str) -> tuple[str, str]:
    if name.startswith("lstm_"):
        model = "LSTM"
        remainder = name.removeprefix("lstm_")
    elif name.startswith("gru_"):
        model = "GRU"
        remainder = name.removeprefix("gru_")
    elif name.startswith("transformer_"):
        model = "Transformer"
        remainder = name.removeprefix("transformer_")
    else:
        raise ValueError(f"Unexpected job name: {name}")

    feature_label = {
        "combined": "Combined",
        "sentiment_only": "Sentiment only",
        "technical_only": "Technical only",
    }[remainder]
    return model, feature_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v6 one-day side-quest tuning grid.")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def write_status(status: dict[str, dict[str, object]]) -> None:
    STATUS_PATH.write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
