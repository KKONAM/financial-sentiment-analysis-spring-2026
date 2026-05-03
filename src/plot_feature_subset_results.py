"""Generate feature-subset result figures for the LaTeX report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPERIMENT_SLUG = "2020-01-01_to_2022-02-28_v4_purged_rawtech_labeled_sentiment"
DEFAULT_INPUT = Path(f"reports/final/feature_subset_tuning_comparison_{EXPERIMENT_SLUG}.csv")
DEFAULT_OUTPUT_DIR = Path("figures")

FEATURE_LABELS = {
    "combined": "Combined",
    "technical_only": "Technical only",
    "sentiment_only": "Sentiment only",
}

FEATURE_COLORS = {
    "combined": "#4169a8",
    "technical_only": "#459b6b",
    "sentiment_only": "#db6b2d",
}


def require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is not installed. Install it with `pip install matplotlib` "
            "or `conda install matplotlib`, then rerun this script."
        ) from exc
    return plt


def feature_family(feature_name: str) -> str:
    return feature_name.replace("_tuned", "")


def read_results(path: Path) -> list[dict[str, str | float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        for metric in (
            "test_mae",
            "test_rmse",
            "test_directional_accuracy",
            "test_correlation",
        ):
            row[metric] = float(row[metric])
        row["feature_family"] = feature_family(str(row["features"]))
        row["plot_label"] = f"{row['model']}\n{FEATURE_LABELS[row['feature_family']]}"
    return rows


def bar_colors(rows: list[dict[str, str | float]]) -> list[str]:
    return [FEATURE_COLORS[str(row["feature_family"])] for row in rows]


def format_axes(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_value_labels(ax, bars, fmt: str = "{:.3f}") -> None:
    y_min, y_max = ax.get_ylim()
    offset = (y_max - y_min) * 0.025
    for bar in bars:
        value = bar.get_height()
        if value >= 0:
            y = value + offset
            va = "bottom"
        else:
            y = value - offset
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=8,
        )


def save_error_figure(rows: list[dict[str, str | float]], output_dir: Path) -> Path:
    plt = require_matplotlib()
    labels = [str(row["plot_label"]) for row in rows]
    colors = bar_colors(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    fig.suptitle("Feature Subset Error Metrics", fontsize=14, fontweight="bold")

    for ax, metric, title in zip(
        axes,
        ("test_mae", "test_rmse"),
        ("Test MAE", "Test RMSE"),
    ):
        values = [float(row[metric]) for row in rows]
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylim(0.0, 0.08)
        format_axes(ax, title, "Error")
        add_value_labels(ax, bars)
        ax.tick_params(axis="x", labelsize=8)

    output_path = output_dir / "feature_subset_error_metrics.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_signal_figure(rows: list[dict[str, str | float]], output_dir: Path) -> Path:
    plt = require_matplotlib()
    labels = [str(row["plot_label"]) for row in rows]
    colors = bar_colors(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    fig.suptitle("Feature Subset Signal Metrics", fontsize=14, fontweight="bold")

    direction_values = [float(row["test_directional_accuracy"]) for row in rows]
    direction_bars = axes[0].bar(labels, direction_values, color=colors)
    axes[0].axhline(
        0.5025,
        color="#9e3333",
        linestyle="--",
        linewidth=1.2,
        label="Always-down baseline",
    )
    axes[0].set_ylim(0.0, 0.60)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    format_axes(axes[0], "Test Directional Accuracy", "Accuracy")
    add_value_labels(axes[0], direction_bars)

    correlation_values = [float(row["test_correlation"]) for row in rows]
    correlation_bars = axes[1].bar(labels, correlation_values, color=colors)
    axes[1].axhline(0.0, color="#555555", linewidth=0.9)
    axes[1].set_ylim(-0.10, 0.10)
    format_axes(axes[1], "Test Return Correlation", "Pearson correlation")
    add_value_labels(axes[1], correlation_bars)

    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)

    output_path = output_dir / "feature_subset_signal_metrics.pdf"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate feature-subset result PDFs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Comparison CSV path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for PDFs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_results(args.input)
    error_path = save_error_figure(rows, args.output_dir)
    signal_path = save_signal_figure(rows, args.output_dir)
    print(f"Wrote {error_path}")
    print(f"Wrote {signal_path}")


if __name__ == "__main__":
    main()
