"""Render Figure 2: posterior revision after peer interaction -> report/figures/peer_interaction.png.

Concrete before/after illustration of one worker's local posterior being revised
after evidence-backed peer support/challenge messages, so the true root becomes a
clear winner instead of a near-tie.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Liberation Serif", "serif"]
plt.rcParams["text.color"] = "#0f172a"

HL = "#3b82f6"       # highlighted (true root)
GRAY = "#cbd5e1"     # other candidates
EDGE = "#475569"


def _bars(ax, values, title):
    cats = ["A", "B", "C"]
    colors = [HL, GRAY, GRAY]
    bars = ax.bar(cats, values, color=colors, edgecolor=EDGE, linewidth=0.8, width=0.62)
    ax.set_ylim(0, 0.75)
    ax.set_title(title, fontsize=10, weight="bold", pad=8)
    ax.set_ylabel("posterior", fontsize=8.5)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5, weight="bold")


def render() -> None:
    fig = plt.figure(figsize=(7.6, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.8, 3.8, 2.8], wspace=0.6)

    ax_before = fig.add_subplot(gs[0, 0])
    _bars(ax_before, [0.40, 0.35, 0.25], "Hậu nghiệm cục bộ (Trace agent)")

    ax_mid = fig.add_subplot(gs[0, 1])
    ax_mid.axis("off")
    ax_mid.set_xlim(0, 1); ax_mid.set_ylim(0, 1)
    ax_mid.annotate("", xy=(0.94, 0.70), xytext=(0.06, 0.70),
                    arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.6, mutation_scale=15))
    ax_mid.text(0.5, 0.78, "hiệu chỉnh", ha="center", va="bottom", fontsize=9, style="italic", color=EDGE)
    ax_mid.text(0.5, 0.40,
                "Vòng tương tác peer:\n"
                "• Metric: support A (CPU bão hoà)\n"
                "• Log: challenge B (không lỗi app)",
                ha="center", va="center", fontsize=7.7,
                bbox=dict(boxstyle="round,pad=0.45", facecolor="#faf5ff", edgecolor="#a855f7", linewidth=1.0))

    ax_after = fig.add_subplot(gs[0, 2])
    _bars(ax_after, [0.62, 0.18, 0.20], "Hậu nghiệm sau hiệu chỉnh")

    fig.suptitle("Hiệu chỉnh hậu nghiệm sau tương tác peer (A = nguyên nhân gốc)",
                 fontsize=11, weight="bold", y=1.05)
    out = "report/figures/peer_interaction.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    os.makedirs("report/figures", exist_ok=True)
    render()
