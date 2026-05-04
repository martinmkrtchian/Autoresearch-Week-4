"""
prepare.py — Read experiments.json and generate performance.png summary chart.

Usage:
    python prepare.py
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LOG_FILE    = "experiments.json"
OUTPUT_FILE = "performance.png"


def main():
    if not os.path.exists(LOG_FILE):
        print(f"No {LOG_FILE} found. Run some experiments first with run.py.")
        return

    with open(LOG_FILE, "r") as f:
        log = json.load(f)

    if not log:
        print("No experiments logged yet.")
        return

    df = pd.DataFrame(log)
    df["index"] = range(1, len(df) + 1)

    # Back-fill columns that older log entries may not have
    if "timeout" not in df.columns:
        df["timeout"] = False
    df["timeout"] = df["timeout"].fillna(False)
    if "runtime_s" not in df.columns:
        df["runtime_s"] = None

    df["color"] = df.apply(
        lambda r: "#f39c12" if r["timeout"] else ("#2ecc71" if r["kept"] else "#e74c3c"),
        axis=1
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(
        "AutoResearch: Energy Stock Intraday Return Classification\nROC-AUC Progress",
        fontsize=14, fontweight="bold", y=0.98
    )

    # ── Top plot: Val ROC-AUC per experiment ────────────────────────────────
    ax1 = axes[0]

    # Only plot non-timeout rows for the bar chart
    valid = df[~df["timeout"]].copy()
    bars = ax1.bar(valid["index"], valid["val_roc_auc"],
                   color=valid["color"], edgecolor="white", linewidth=0.8)

    # Mark timeout experiments as an X on the x-axis
    for _, row in df[df["timeout"]].iterrows():
        ax1.annotate("⏱ TIMEOUT", xy=(row["index"], 0.01),
                     ha="center", va="bottom", fontsize=7, color="#f39c12")

    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1)

    # Running best line (skip timeouts)
    running_best = valid["val_roc_auc"].cummax()
    ax1.plot(valid["index"], running_best,
             color="#3498db", linewidth=2, marker="o", markersize=4, label="Running best")

    ax1.set_xlabel("Experiment #")
    ax1.set_ylabel("Val ROC-AUC")
    ax1.set_title("Validation ROC-AUC per Experiment")
    ax1.set_xticks(df["index"])
    ax1.set_xticklabels([f"#{i}" for i in df["index"]], fontsize=8)

    if not valid.empty:
        ax1.set_ylim(
            max(0,   valid["val_roc_auc"].min() - 0.05),
            min(1.0, valid["val_roc_auc"].max() + 0.05)
        )

    kept_patch    = mpatches.Patch(color="#2ecc71", label="Kept")
    disc_patch    = mpatches.Patch(color="#e74c3c", label="Discarded")
    timeout_patch = mpatches.Patch(color="#f39c12", label="Timeout")
    ax1.legend(
        handles=[kept_patch, disc_patch, timeout_patch,
                 plt.Line2D([0], [0], color="#3498db", linewidth=2, label="Running best"),
                 plt.Line2D([0], [0], color="gray", linestyle="--", linewidth=1, label="Random baseline (0.5)")],
        loc="lower right", fontsize=8
    )

    # Value labels on bars
    for bar, val in zip(bars, valid["val_roc_auc"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # ── Bottom plot: Summary table ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.axis("off")

    col_labels = ["#", "Description", "Train AUC", "Val AUC", "Runtime", "Status"]
    table_data = []
    for _, row in df.iterrows():
        if row["timeout"]:
            status        = "⏱ TIMEOUT"
            train_auc_str = "—"
            val_auc_str   = "—"
        else:
            status        = "✓ KEPT" if row["kept"] else "✗ DISC."
            train_auc_str = f"{row['train_roc_auc']:.4f}"
            val_auc_str   = f"{row['val_roc_auc']:.4f}"
        runtime_str = f"{row['runtime_s']:.0f}s" if pd.notna(row.get("runtime_s")) else "—"
        table_data.append([
            f"#{int(row['index'])}",
            row["description"],
            train_auc_str,
            val_auc_str,
            runtime_str,
            status,
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Color rows
    row_colors = {"kept": "#d5f5e3", "discarded": "#fadbd8", "timeout": "#fef3cd"}
    for i, row in df.iterrows():
        if row["timeout"]:
            color = row_colors["timeout"]
        elif row["kept"]:
            color = row_colors["kept"]
        else:
            color = row_colors["discarded"]
        for j in range(len(col_labels)):
            table[(i + 1, j)].set_facecolor(color)

    # Header style
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    ax2.set_title("Experiment Summary", fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"\nPerformance chart saved to {OUTPUT_FILE}")

    # Console summary
    if not valid.empty:
        best = valid.loc[valid["val_roc_auc"].idxmax()]
        print(f"\nBest experiment : #{int(best['index'])} — {best['description']}")
        print(f"Best Val ROC-AUC: {best['val_roc_auc']:.4f}")
    print(f"\nFull summary:")
    print(df[["index", "description", "train_roc_auc", "val_roc_auc",
              "runtime_s", "kept", "timeout"]].to_string(index=False))


if __name__ == "__main__":
    main()
