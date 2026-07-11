"""Render the up-to-date ShardRCA architecture flowchart to report/figures/shard_rca.png.

The diagram reflects the current autonomous peer-interaction MAS pipeline:
Task+candidate catalog -> planner/shard builder -> parallel isolated investigator
agents (each emits a local posterior + evidence pointers on a shared blackboard)
-> autonomous peer-interaction round (publish -> critique -> revise posterior)
-> correlation-aware fusion -> causal re-rank / targeted refinement
-> adversarial falsifier -> final RCA answer.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Academic serif style
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman", "Liberation Serif", "serif"]
plt.rcParams["text.color"] = "#0f172a"

COLOR_TEXT = "#0f172a"
COLOR_ARROW = "#64748b"

FILL_GRAY = "#f8fafc"; BORDER_GRAY = "#94a3b8"
FILL_BLUE = "#f0f4ff"; BORDER_BLUE = "#5c7cfa"
FILL_AMBER = "#fff9db"; BORDER_AMBER = "#fcc419"
FILL_PURPLE = "#f3f0ff"; BORDER_PURPLE = "#845ef7"
FILL_TEAL = "#e6fcf5"; BORDER_TEAL = "#20c997"
FILL_AZURE = "#e7f5ff"; BORDER_AZURE = "#339af0"
FILL_ROSE = "#fff5f5"; BORDER_ROSE = "#ff6b6b"
FILL_GREEN = "#ebfbee"; BORDER_GREEN = "#51cf66"


def _box(ax, x, y, w, h, text, fill_color, border_color, fs=9.0, weight="normal", style="normal", lw=1.2):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.03",
        linewidth=lw, edgecolor=border_color, facecolor=fill_color,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=COLOR_TEXT, weight=weight, style=style)


def _arrow(ax, x1, y1, x2, y2, label="", lw=1.2, color=COLOR_ARROW):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=12,
                                shrinkA=0, shrinkB=0))
    if label:
        if abs(y1 - y2) < 1e-4:
            ax.text((x1 + x2) / 2, y1 + 0.08, label, ha="center", va="bottom",
                    fontsize=8.5, color="#475569", weight="bold")
        else:
            ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, label, ha="left", va="center",
                    fontsize=8.5, color="#475569")


def render_shard_rca() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 12.0))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-0.1, 12.1)
    ax.axis("off")

    cx = 4.0
    fw_x, fw_w = 0.2, 7.6  # full-width boxes

    ax.text(cx, 11.75, "ShardRCA: Evidence-Isolated Autonomous Multi-Agent RCA",
            ha="center", va="center", fontsize=13.0, weight="bold", color="#0f172a")

    # 1. Task & candidate catalog
    _box(ax, fw_x, 10.85, fw_w, 0.80,
         "Task & Candidate Catalog\nCMDB component × fault reason universe + multi-source telemetry feeds",
         FILL_GRAY, BORDER_GRAY, fs=8.5)

    # 2. Planner / shard builder
    _box(ax, fw_x, 9.40, fw_w, 0.95,
         "PLANNER / SHARD BUILDER\nPartition observability by modality × component groups × active time window",
         FILL_BLUE, BORDER_BLUE, fs=8.5, weight="bold")
    _arrow(ax, cx, 10.85, cx, 10.35, label="Telemetry stream")

    # 3. Parallel isolated investigator agents (dashed container)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0.2, 7.0), 7.6, 1.9, boxstyle="round,pad=0,rounding_size=0.02",
        linewidth=1.0, edgecolor="#cbd5e1", facecolor="#f8fafc", ls="--"))
    ax.text(0.3, 8.7, "Parallel Isolated Execution  (each agent sees ONLY its planned shard)",
            fontsize=8.0, color="#64748b", weight="bold", style="italic",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

    # Agents with width 2.2, spanning 0.5 -> 2.7, 2.9 -> 5.1, 5.3 -> 7.5
    agents = [
        (0.5, "Metric Agent\n(KPI anomalies)\nCPU / RAM\nBandwidth", 1.6),
        (2.9, "Log Agent\n(pattern shifts)\nsyslog &\nstdout patterns", 4.0),
        (5.3, "Trace Agent\n(RPC latencies)\nspan graphs &\ndependencies", 6.4),
    ]
    for ax0, label, center_x in agents:
        _box(ax, ax0, 7.15, 2.2, 1.40, label, FILL_AMBER, BORDER_AMBER, fs=8.5)

    # split arrows planner -> agents
    ax.plot([cx, cx], [9.40, 9.0], color=COLOR_ARROW, lw=1.2)
    ax.plot([1.6, 6.4], [9.0, 9.0], color=COLOR_ARROW, lw=1.2)
    for _, _, center_x in agents:
        _arrow(ax, center_x, 9.0, center_x, 8.55)

    ax.text(7.6, 7.9, "each emits\nlocal posterior\n+ evidence ptrs\nto blackboard",
            fontsize=7.5, color="#64748b", va="center", ha="left")

    # 4. Autonomous peer interaction
    _box(ax, fw_x, 5.35, fw_w, 1.00,
         "AUTONOMOUS PEER INTERACTION  (MAS review round)\n"
         "Exchange proposals   ⇄   Peer critique (support / challenge)   ⇄   Refine posteriors",
         FILL_PURPLE, BORDER_PURPLE, fs=8.5, weight="bold")
         
    # merge agents -> interaction
    for _, _, center_x in agents:
        ax.plot([center_x, center_x], [7.15, 6.75], color=COLOR_ARROW, lw=1.0)
    ax.plot([1.6, 6.4], [6.75, 6.75], color=COLOR_ARROW, lw=1.0)
    _arrow(ax, cx, 6.75, cx, 6.35)

    # 5. Correlation-aware fusion
    _box(ax, fw_x, 4.05, fw_w, 1.00,
         "CORRELATION-AWARE FUSION\n"
         "Redundancy-discounted Log-Opinion Pool (Product of Experts)",
         FILL_TEAL, BORDER_TEAL, fs=8.5, weight="bold")
    _arrow(ax, cx, 5.35, cx, 5.05)

    # 6. Causal re-rank + refinement
    _box(ax, fw_x, 2.80, fw_w, 0.95,
         "CAUSAL RE-RANK  +  TARGETED REFINEMENT\n"
         "Score boosted by call-graph topology coverage & temporal anomaly precedence",
         FILL_AZURE, BORDER_AZURE, fs=8.5)
    _arrow(ax, cx, 4.05, cx, 3.75)

    # 7. Adversarial falsifier
    _box(ax, fw_x, 1.60, fw_w, 0.95,
         "ADVERSARIAL FALSIFICATION GATE\n"
         "Top-candidate verification against Blackboard runner-up scores",
         FILL_ROSE, BORDER_ROSE, fs=8.5)
    _arrow(ax, cx, 2.80, cx, 2.55)

    # 8. Final answer
    _box(ax, fw_x, 0.40, fw_w, 0.95,
         "FINAL RCA ANSWER\n"
         "Root-cause component + fault reason + occurrence timestamp  (+ audit trail)",
         FILL_GREEN, BORDER_GREEN, fs=8.5, weight="bold")
    _arrow(ax, cx, 1.60, cx, 1.35)

    out = "report/figures/shard_rca.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    os.makedirs("report/figures", exist_ok=True)
    render_shard_rca()
