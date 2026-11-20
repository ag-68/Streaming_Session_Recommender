"""Charts for the README. The numbers come from the artifact files."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sessrec.paths import (
    CURVES_FILE,
    DOCS_FIGURES,
    HISTORY_LENGTH_FILE,
    LATENCY_FILE,
    METRICS_FILE,
)

COLORS = {"popularity": "#e7a8cc", "item_knn": "#ff9f1c", "sasrec": "#b388ff"}
LABELS = {"popularity": "Popularity", "item_knn": "Item-to-item", "sasrec": "SASRec"}


def _style(ax) -> None:
    ax.set_facecolor("#10140f")
    ax.figure.set_facecolor("#10140f")
    ax.tick_params(colors="#f6eef4")
    ax.xaxis.label.set_color("#f6eef4")
    ax.yaxis.label.set_color("#f6eef4")
    ax.title.set_color("#f6eef4")
    for spine in ax.spines.values():
        spine.set_color("#2c3826")


def write_figures() -> None:
    DOCS_FIGURES.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    curves = json.loads(CURVES_FILE.read_text(encoding="utf-8"))
    names = ["popularity", "item_knn", "sasrec"]
    measures = ["ndcg@10", "hr@10", "coverage@10"]
    measure_labels = ["NDCG@10", "HR@10", "Coverage@10"]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    _style(axis)
    x = np.arange(len(measures))
    width = 0.24
    for offset, name in enumerate(names):
        values = [metrics["models"][name][measure] for measure in measures]
        axis.bar(x + (offset - 1) * width, values, width, label=LABELS[name], color=COLORS[name])
    axis.set_xticks(x, measure_labels)
    axis.set_ylabel("Score")
    axis.set_title("Test split, full catalog")
    axis.legend(facecolor="#1a2118", edgecolor="#2c3826", labelcolor="#f6eef4")
    figure.tight_layout()
    figure.savefig(DOCS_FIGURES / "metrics.png", dpi=140)
    plt.close(figure)

    epochs = [row["epoch"] for row in curves["epochs"]]
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 4))
    for axis in axes:
        _style(axis)
    axes[0].plot(epochs, [row["train_loss"] for row in curves["epochs"]], color="#ff9f1c")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(epochs, [row["val_ndcg@10"] for row in curves["epochs"]], color="#b388ff")
    axes[1].axvline(curves["best_epoch"], color="#e7a8cc", linestyle="--", label="best checkpoint")
    axes[1].set_title("Validation NDCG@10")
    axes[1].set_xlabel("Epoch")
    axes[1].legend(facecolor="#1a2118", edgecolor="#2c3826", labelcolor="#f6eef4")
    figure.tight_layout()
    figure.savefig(DOCS_FIGURES / "training.png", dpi=140)
    plt.close(figure)

    if LATENCY_FILE.exists():
        latency = json.loads(LATENCY_FILE.read_text(encoding="utf-8"))
        lengths = sorted(int(key) for key in latency["by_history_length"])
        figure, axis = plt.subplots(figsize=(8, 4.5))
        _style(axis)
        for mark, color in (("p50", "#8bd450"), ("p95", "#ff9f1c"), ("p99", "#e7a8cc")):
            axis.plot(
                lengths,
                [latency["by_history_length"][str(length)][mark] for length in lengths],
                marker="o",
                color=color,
                label=mark,
            )
        axis.axhline(50, color="#ff5d73", linestyle="--", label="50 ms")
        axis.set_xlabel("Films in the request")
        axis.set_ylabel("Milliseconds")
        axis.set_title("HTTP round trip on localhost")
        axis.legend(facecolor="#1a2118", edgecolor="#2c3826", labelcolor="#f6eef4")
        figure.tight_layout()
        figure.savefig(DOCS_FIGURES / "latency.png", dpi=140)
        plt.close(figure)

    if HISTORY_LENGTH_FILE.exists():
        history = json.loads(HISTORY_LENGTH_FILE.read_text(encoding="utf-8"))
        figure, axis = plt.subplots(figsize=(8, 4.5))
        _style(axis)
        for name in names:
            axis.plot(
                history["lengths"],
                history["models"][name],
                marker="o",
                color=COLORS[name],
                label=LABELS[name],
            )
        axis.set_xlabel("Recent films shown to the method")
        axis.set_ylabel("NDCG@10")
        axis.set_title("Test score when the history is shortened")
        axis.legend(facecolor="#1a2118", edgecolor="#2c3826", labelcolor="#f6eef4")
        figure.tight_layout()
        figure.savefig(DOCS_FIGURES / "history_length.png", dpi=140)
        plt.close(figure)


def main() -> None:
    write_figures()


if __name__ == "__main__":
    main()
