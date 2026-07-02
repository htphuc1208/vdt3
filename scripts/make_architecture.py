"""Render the TelcoMAS architecture diagram to report/figures/architecture.png.

Reproducible (no data, just a diagram): run `python scripts/make_architecture.py`.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BLUE, GREEN, ORANGE, GREY, PURPLE = "#2563eb", "#16a34a", "#f59e0b", "#e5e7eb", "#7c3aed"


def _box(ax, x, y, w, h, text, color, fs=9, tc="white"):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0, facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def _arrow(ax, x1, y1, x2, y2, color="#374151"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("TelcoMAS — Multi-Agent architecture for telecom incident RCA", fontsize=13, weight="bold")

    # Orchestrator band
    _box(ax, 0.3, 6.0, 11.4, 0.7, "Flow-of-Action Orchestrator  (SOP-driven controller)", BLUE, fs=11)

    # Pipeline stages
    stages = [
        ("Detection\n/ Triage", 0.4),
        ("Correlation\n(RAG: SOPs +\nhistory)", 2.35),
        ("Diagnosis experts\nRAN · Transport ·\nCore", 4.3),
        ("Consensus\n(weighted vote\n+ arbiter)", 6.55),
        ("Remediation\n(SOP plan)", 8.6),
        ("Validation\n(apply + verify)", 10.35),
    ]
    y, h, w = 4.4, 1.1, 1.6
    centers = []
    for i, (label, x) in enumerate(stages):
        color = GREEN if i == 2 else (PURPLE if i == 3 else BLUE)
        _box(ax, x, y, w, h, label, color, fs=8.5)
        centers.append((x + w / 2, x))
    for i in range(len(stages) - 1):
        _arrow(ax, stages[i][1] + w, y + h / 2, stages[i + 1][1], y + h / 2)

    # Tool layer
    _box(ax, 0.4, 2.7, 11.3, 0.75,
         "Tool layer (OpenAI function-calling):  query_alarms · query_kpis · query_logs · "
         "query_topology · search_knowledge_base · get_historical_incidents · run_diagnostic · apply_remediation",
         ORANGE, fs=8.2, tc="black")
    for _, x in stages[:5]:
        _arrow(ax, x + w / 2, y, x + w / 2, 3.45, color="#9ca3af")

    # Environment + knowledge base
    _box(ax, 0.4, 1.2, 5.4, 1.1,
         "Simulated 5G network\nCore → Transport → RAN + Power\ntelemetry + fault injection (ground truth)",
         GREY, fs=8.5, tc="black")
    _box(ax, 6.3, 1.2, 5.4, 1.1,
         "Knowledge base\nSOP playbooks + historical incidents\n(TF-IDF retrieval)",
         GREY, fs=8.5, tc="black")
    _arrow(ax, 3.1, 2.7, 3.1, 2.3, color="#9ca3af")
    _arrow(ax, 9.0, 2.7, 9.0, 2.3, color="#9ca3af")

    # LLM note
    ax.text(6, 0.5, "Every agent reasons via a live LLM (OpenAI / DeepSeek, OpenAI-compatible API)",
            ha="center", va="center", fontsize=9, style="italic", color="#374151")

    os.makedirs("report/figures", exist_ok=True)
    out = "report/figures/architecture.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
