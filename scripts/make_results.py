"""Render Figure 3: RCAEval-Hard Hit@1 comparison -> report/figures/results_hitrate.png.

Grouped bars of ShardRCA vs a budgeted single-context ReAct baseline on two
RCAEval-Hard holdouts, with paired-test p-values annotated. Numbers match the
repository's Current Evidence Status table in README.md.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Liberation Serif", "serif"]
plt.rcParams["text.color"] = "#0f172a"

SHARD = "#2563eb"
BASE = "#94a3b8"
EDGE = "#334155"


def render() -> None:
    groups = ["Holdout v7", "Holdout xác nhận (n=24)"]
    shard = [0.60, 0.667]
    base = [0.20, 0.375]
    pvals = ["p = 0.008", "p = 0.039"]
    x = [0.0, 1.15]
    w = 0.34

    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    b1 = ax.bar([xi - w / 2 for xi in x], shard, width=w, color=SHARD, edgecolor=EDGE,
                linewidth=0.8, label="ShardRCA")
    b2 = ax.bar([xi + w / 2 for xi in x], base, width=w, color=BASE, edgecolor=EDGE,
                linewidth=0.8, label="Tác tử đơn ReAct")

    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9.5, weight="bold")

    # p-value brackets over each group
    for xi, p in zip(x, pvals):
        top = 0.80
        ax.plot([xi - w / 2, xi - w / 2, xi + w / 2, xi + w / 2],
                [top - 0.03, top, top, top - 0.03], color=EDGE, lw=1.0)
        ax.text(xi, top + 0.005, p, ha="center", va="bottom", fontsize=9.5, style="italic")

    ax.set_ylim(0, 0.92)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("Hit@1 (định vị nguyên nhân gốc)", fontsize=10)
    ax.set_title("Định vị nguyên nhân gốc trên holdout RCAEval-Hard", fontsize=11.5, weight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9.2, loc="center", bbox_to_anchor=(0.5, 0.60),
              frameon=True, framealpha=0.95, edgecolor="#cccccc")
    ax.tick_params(axis="y", labelsize=9)

    out = "report/figures/results_hitrate.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    os.makedirs("report/figures", exist_ok=True)
    render()
