"""Generate benchmark charts (multi-agent vs single-agent) for the report."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402

_LABELS = {
    "multi_agent": "Multi-Agent (TelcoMAS)", "full": "Multi-Agent (full)", "multi": "Multi-Agent",
    "single_agent": "Single-Agent baseline", "single": "Single-Agent",
    "no_rag": "− RAG", "no_consensus": "− consensus", "no_arbiter": "− arbiter",
    "no_partition": "− partition", "no_debate": "− debate",
}
_COLORS = {
    "multi_agent": "#2563eb", "full": "#2563eb", "multi": "#2563eb",
    "single_agent": "#9ca3af", "single": "#9ca3af",
    "no_rag": "#f59e0b", "no_consensus": "#ef4444", "no_arbiter": "#8b5cf6",
    "no_partition": "#10b981", "no_debate": "#0ea5e9",
}
_ORDER = [
    "multi_agent", "full", "multi", "single_agent", "single",
    "no_rag", "no_consensus", "no_arbiter", "no_partition", "no_debate",
]


def _label(system: str) -> str:
    return _LABELS.get(system, system)


def make_charts(summary: dict, outdir: str = "report/figures") -> list[str]:
    os.makedirs(outdir, exist_ok=True)
    systems = [s for s in _ORDER if s in summary] or list(summary)
    paths: list[str] = []

    # --- accuracy comparison ---
    metrics = [
        ("localization_accuracy", "Localization"),
        ("fault_type_accuracy", "Fault type"),
        ("causal_explanation_accuracy", "Causal"),
        ("diagnosis_accuracy", "Strict diagnosis"),
        ("end_to_end_accuracy", "End-to-end"),
        ("resolution_rate", "Resolution"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n = len(systems)
    width = 0.8 / n
    for i, system in enumerate(systems):
        vals = [summary[system][m] * 100 for m, _ in metrics]
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(metrics))]
        bars = ax.bar(xs, vals, width=width, label=_label(system), color=_COLORS.get(system, None))
        ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=9)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([lbl for _, lbl in metrics])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Incident RCA accuracy: multi-agent vs single-agent")
    ax.legend()
    fig.tight_layout()
    p1 = os.path.join(outdir, "accuracy_comparison.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    # --- efficiency (tokens + calls) ---
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9, 4))
    tok = [summary[s]["avg_total_tokens"] for s in systems]
    b1 = axl.bar([_label(s) for s in systems], tok, color=[_COLORS.get(s) for s in systems])
    axl.bar_label(b1, fmt="%.0f", fontsize=9)
    axl.set_title("Avg tokens per incident")
    axl.set_ylabel("tokens")
    axl.tick_params(axis="x", labelsize=8)

    tool = [summary[s]["avg_tool_calls"] for s in systems]
    llm = [summary[s]["avg_llm_calls"] for s in systems]
    x = range(len(systems))
    w = 0.35
    b2 = axr.bar([i - w / 2 for i in x], tool, width=w, label="tool calls", color="#2563eb")
    b3 = axr.bar([i + w / 2 for i in x], llm, width=w, label="LLM calls", color="#f59e0b")
    axr.bar_label(b2, fmt="%.0f", fontsize=8)
    axr.bar_label(b3, fmt="%.0f", fontsize=8)
    axr.set_xticks(list(x))
    axr.set_xticklabels([_label(s) for s in systems], fontsize=8)
    axr.set_title("Avg calls per incident")
    axr.legend()
    fig.tight_layout()
    p2 = os.path.join(outdir, "efficiency_comparison.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)
    return paths
