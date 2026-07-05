"""Render architectural evolution timeline to report/figures/architecture_timeline.png.

This script produces a NeurIPS-standard academic diagram illustrating the three phases:
- Phase 1 (v1-v3): Persona Committee
- Phase 2 (v4): Information Sharding
- Phase 3 (v5-v7): ShardRCA Deployed & Benchmark Rigor
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Set academic serif font style
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Liberation Serif", "serif"]
plt.rcParams["text.color"] = "#0f172a"

# Colors
COLOR_TEXT = "#0f172a"
COLOR_ARROW = "#475569"

# Fills & Borders for 3 phases
FILL_P1 = "#f1f5f9"    # Soft Gray
BORDER_P1 = "#64748b"

FILL_P2 = "#eff6ff"    # Soft Blue
BORDER_P2 = "#3b82f6"

FILL_P3 = "#f0fdf4"    # Soft Green
BORDER_P3 = "#10b981"


import textwrap

def _card(ax, x, y, w, h, title, points, fill_color, border_color):
    # Draw card outline
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04",
        linewidth=1.5, edgecolor=border_color, facecolor=fill_color
    ))

    # Title
    ax.text(
        x + w / 2, y + h - 0.4, title,
        ha="center", va="center", fontsize=12.0, color=COLOR_TEXT,
        weight="bold"
    )

    # Divider line
    ax.plot([x + 0.2, x + w - 0.2], [y + h - 0.65, y + h - 0.65], color=border_color, lw=0.8, ls="--")

    # Bullet points
    y_text = y + h - 0.8
    line_h = 0.190
    
    for heading, body in points:
        # Heading
        ax.text(
            x + 0.2, y_text, f"\u2022 {heading}:",
            ha="left", va="top", fontsize=9.2, color=COLOR_TEXT, weight="bold"
        )
        y_text -= 0.19
        
        # Wrap body text manually (width=35 fits w=3.2 beautifully with larger font)
        wrapped_lines = textwrap.wrap(body, width=35)
        for line in wrapped_lines:
            ax.text(
                x + 0.35, y_text, line,
                ha="left", va="top", fontsize=8.7, color="#334155"
            )
            y_text -= line_h
            
        y_text -= 0.12 # spacing between bullets


def _arrow(ax, x1, y1, x2, y2, label=""):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_ARROW, lw=2.0, mutation_scale=15, shrinkA=0, shrinkB=0)
    )
    if label:
        ax.text((x1 + x2) / 2, y1 + 0.15, label, ha="center", va="bottom", fontsize=9.5, weight="bold", color="#475569")


def render_timeline() -> None:
    fig, ax = plt.subplots(figsize=(13.0, 5.0))
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0.5, 5.5)
    ax.axis("off")

    # Card 1: Phase 1
    p1_points = [
        ("Context", "LLM agents assigned domain roles (RAN, Core) but looking at the SAME global telemetry."),
        ("Finding", "Zero information diversity. A single LLM with a global view performs equally well but at 1/3 the cost/latency."),
        ("Key Lesson", "Persona-only diversity on shared data is a design overhead in fully-observable simulators.")
    ]
    _card(ax, 0.4, 1.2, 3.2, 3.8, "Phase 1: Persona Committee", p1_points, FILL_P1, BORDER_P1)

    # Card 2: Phase 2
    p2_points = [
        ("Design Shift", "Physically partition observability. RAN expert queries RAN KPIs; Core expert queries Core telemetry."),
        ("Mechanism", "Ensures independent reasoning. No expert can read other domains' data directly."),
        ("Finding", "Solves duplicate reasoning but introduces domain boundaries and tie-vote conflicts.")
    ]
    _card(ax, 4.9, 1.2, 3.2, 3.8, "Phase 2: Information Sharding", p2_points, FILL_P2, BORDER_P2)

    # Card 3: Phase 3
    p3_points = [
        ("Solution", "Deployed calibrated evidence-weighted consensus to resolve expert voting ties."),
        ("Methodology", "Audited and fixed label leakage bugs; integrated 5 strong academic baselines."),
        ("Benchmark", "Evaluated on RCAEval-Hard holdout dataset for real-world robustness.")
    ]
    _card(ax, 9.4, 1.2, 3.2, 3.8, "Phase 3: ShardRCA Deployed", p3_points, FILL_P3, BORDER_P3)

    # Timeline connection arrows
    _arrow(ax, 3.9, 3.1, 4.6, 3.1, label="Redesign")
    _arrow(ax, 8.4, 3.1, 9.1, 3.1, label="Rigor & Fix")

    # Takeaway box
    ax.text(
        6.5, 0.65,
        "Core Conclusion: Multi-Agent Systems only outperform single-agent baselines when:\n"
        "(a) Data volume exceeds the context window, or (b) Information is genuinely disjoint (physically partitioned).",
        ha="center", va="center", fontsize=11.0, weight="bold", color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.4", edgecolor="#94a3b8", facecolor="#f8fafc", lw=1.0)
    )

    out = "report/figures/architecture_timeline.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    os.makedirs("report/figures", exist_ok=True)
    render_timeline()
