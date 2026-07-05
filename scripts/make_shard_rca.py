"""Render ShardRCA architecture flowchart and explanation to report/figures/shard_rca.png.

This script produces a NeurIPS-standard academic diagram illustrating:
- The ShardRCA evidence-isolated multi-agent RCA pipeline (flowchart).
- High-level reasoning for why a single agent cannot replicate this behavior.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import textwrap

# Set academic serif font style
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Liberation Serif", "serif"]
plt.rcParams["text.color"] = "#0f172a"

# Colors
COLOR_TEXT = "#0f172a"
COLOR_ARROW = "#475569"

# Soft fills and borders
FILL_GRAY = "#f1f5f9"
BORDER_GRAY = "#64748b"

FILL_BLUE = "#eff6ff"
BORDER_BLUE = "#3b82f6"

FILL_AMBER = "#fffbeb"
BORDER_AMBER = "#f59e0b"

FILL_PURPLE = "#faf5ff"
BORDER_PURPLE = "#a855f7"

FILL_ROSE = "#fff1f2"
BORDER_ROSE = "#f43f5e"

FILL_GREEN = "#f0fdf4"
BORDER_GREEN = "#10b981"


def _box(ax, x, y, w, h, text, fill_color, border_color, fs=9.5, weight="normal", style="normal"):
    # Draw rounded rectangle
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03",
        linewidth=1.2, edgecolor=border_color, facecolor=fill_color
    ))
    # Add text
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fs, color=COLOR_TEXT,
        weight=weight, style=style
    )


def _arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_ARROW, lw=1.2, mutation_scale=12, shrinkA=0, shrinkB=0)
    )
    if label:
        if abs(y1 - y2) < 1e-4:  # Horizontal arrow
            ax.text((x1 + x2) / 2, y1 + 0.05, label, ha="center", va="bottom", fontsize=8.8, color="#475569", weight="bold")
        else:  # Vertical or diagonal arrow
            ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, label, ha="left", va="center", fontsize=8.8, color="#475569")


def render_shard_rca() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 7.5))
    ax.set_xlim(-0.3, 6.5)
    ax.set_ylim(0.0, 7.5)
    ax.axis("off")

    # Title
    ax.text(
        3.1, 7.25, "ShardRCA Architecture: Evidence-Isolated Multi-Agent RCA",
        ha="center", va="center", fontsize=13, weight="bold", color="#0f172a"
    )

    # ==========================================
    # FLOWCHART
    # ==========================================
    
    # 1. Catalog
    _box(
        ax, 1.3, 6.4, 3.6, 0.6,
        "Task & Telemetry Catalog\n(Target incident & global data feeds)",
        FILL_GRAY, BORDER_GRAY, fs=9.0
    )
    
    # 2. Planner
    _box(
        ax, 1.3, 5.45, 3.6, 0.7,
        "PLANNER (1 LLM call)\nFormulates domain-specific query shards",
        FILL_BLUE, BORDER_BLUE, fs=9.2, weight="bold"
    )
    _arrow(ax, 3.1, 6.4, 3.1, 6.15)
    
    # Label pointing to shard plan
    _arrow(ax, 4.9, 5.8, 5.5, 5.8, label="shard plan")
    
    # 3. Parallel Miners (Dashed background box)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.25, 4.05), 5.7, 1.25, boxstyle="round,pad=0,rounding_size=0.03",
        linewidth=1.0, edgecolor="#94a3b8", facecolor="#f8fafc", ls="--"
    ))
    ax.text(
        0.35, 5.18, "Parallel Isolated Execution",
        fontsize=8.0, color="#64748b", weight="bold", style="italic",
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0)
    )
    
    # Miner boxes
    _box(
        ax, 0.4, 4.2, 1.7, 0.8,
        "Metric Miner\n(pandas tools)\nCPU, RAM, bandwidth",
        FILL_AMBER, BORDER_AMBER, fs=8.2
    )
    _box(
        ax, 2.25, 4.2, 1.7, 0.8,
        "Log Miner\n(template/anom)\nSyslog patterns",
        FILL_AMBER, BORDER_AMBER, fs=8.2
    )
    _box(
        ax, 4.1, 4.2, 1.7, 0.8,
        "Trace Miner\n(graph/latency)\nRPC dependencies",
        FILL_AMBER, BORDER_AMBER, fs=8.2
    )
    
    # Split arrows
    ax.plot([3.1, 3.1], [5.45, 5.2], color=COLOR_ARROW, lw=1.2)
    ax.plot([1.25, 4.95], [5.2, 5.2], color=COLOR_ARROW, lw=1.2)
    _arrow(ax, 1.25, 5.2, 1.25, 5.0)
    _arrow(ax, 3.1, 5.2, 3.1, 5.0)
    _arrow(ax, 4.95, 5.2, 4.95, 5.0)
    
    # 4. Blackboard
    _box(
        ax, 1.3, 3.0, 3.6, 0.7,
        "BLACKBOARD (Structured findings)\nAggregates compact, typed domain evidence",
        FILL_GRAY, BORDER_GRAY, fs=9.0, weight="bold"
    )
    
    # Merge arrows
    ax.plot([1.25, 1.25], [4.2, 3.9], color=COLOR_ARROW, lw=1.2)
    ax.plot([3.1, 3.1], [4.2, 3.9], color=COLOR_ARROW, lw=1.2)
    ax.plot([4.95, 4.95], [4.2, 3.9], color=COLOR_ARROW, lw=1.2)
    ax.plot([1.25, 4.95], [3.9, 3.9], color=COLOR_ARROW, lw=1.2)
    _arrow(ax, 3.1, 3.9, 3.1, 3.7)
    
    # 5. Synthesizer
    _box(
        ax, 1.3, 2.05, 3.6, 0.7,
        "SYNTHESIZER (k=3 samples)\nGenerates candidate root-causes\nand tallies majority vote",
        FILL_PURPLE, BORDER_PURPLE, fs=9.0
    )
    _arrow(ax, 3.1, 3.0, 3.1, 2.75)
    
    # 6. Falsifier
    _box(
        ax, 1.3, 1.1, 3.6, 0.7,
        "FALSIFIER (LLM disproof)\nExecutes adversarial sanity checks\nto challenge top candidate",
        FILL_ROSE, BORDER_ROSE, fs=9.0
    )
    _arrow(ax, 3.1, 2.05, 3.1, 1.8)
    
    # 7. Final RCA
    _box(
        ax, 1.3, 0.1, 3.6, 0.75,
        "Final RCA Answer\n(Root-cause localization & remediation plan)",
        FILL_GREEN, BORDER_GREEN, fs=9.2, weight="bold"
    )
    _arrow(ax, 3.1, 1.1, 3.1, 0.85)

    out = "report/figures/shard_rca.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    os.makedirs("report/figures", exist_ok=True)
    render_shard_rca()
