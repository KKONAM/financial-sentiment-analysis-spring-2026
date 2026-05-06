from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


EXPERIMENT_SLUG = "2020-01-01_to_2022-02-28_v6_purged_warmtech_finetuned_finbert_sentiment"
REPORT_DIR = Path("reports/final")
CHECKPOINT_DIR = Path("model_checkpoints/final")
LOG_DIR = REPORT_DIR / "tuning_v6_logs"
STATUS_PATH = LOG_DIR / "status.json"


@dataclass(frozen=True)
class TuningJob:
    name: str
    command: list[str]
    expected_report: Path
    expected_checkpoint: Path


def build_jobs(python_executable: str, trials: int, torch_threads: int) -> list[TuningJob]:
    common = ["--trials", str(trials), "--torch-threads", str(torch_threads)]
    jobs = [
        TuningJob(
            name="lstm_combined",
            command=[python_executable, "src/tune_lstm.py", *common],
            expected_report=REPORT_DIR / f"lstm_combined_tuned_{EXPERIMENT_SLUG}_best.json",
            expected_checkpoint=CHECKPOINT_DIR / f"lstm_combined_tuned_{EXPERIMENT_SLUG}_best.pt",
        ),
        TuningJob(
            name="gru_combined",
            command=[python_executable, "src/tune_gru.py", *common],
            expected_report=REPORT_DIR / f"gru_combined_tuned_{EXPERIMENT_SLUG}_best.json",
            expected_checkpoint=CHECKPOINT_DIR / f"gru_combined_tuned_{EXPERIMENT_SLUG}_best.pt",
        ),
        TuningJob(
            name="transformer_combined",
            command=[python_executable, "src/tune_transformer.py", *common],
            expected_report=REPORT_DIR / f"transformer_combined_tuned_{EXPERIMENT_SLUG}_best.json",
            expected_checkpoint=CHECKPOINT_DIR / f"transformer_combined_tuned_{EXPERIMENT_SLUG}_best.pt",
        ),
    ]

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
            print(f"[runner] started {job.name} pid={process.pid}", flush=True)

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
                print(f"[runner] completed {name} in {elapsed:.1f}s", flush=True)
            else:
                failed += 1
                print(f"[runner] failed {name} rc={return_code} in {elapsed:.1f}s", flush=True)
                if args.stop_on_failure:
                    pending.clear()

    print(f"[runner] finished completed={completed} failed={failed}", flush=True)
    if failed:
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the v6 LSTM/GRU/Transformer tuning grid.")
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
