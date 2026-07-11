"""Redraw the headline results figure (new design, not reusing make_results.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "report/figures/fig3_results.png"

groups = ["RCAEval-Hard\n(v7 holdout)", "RCAEval-Hard\n(holdout xác nhận)", "OpenRCA Telecom\n(n=51, strict)"]
shardrca = [0.60, 0.667, 0.196]
baseline = [0.20, 0.375, 0.157]
pvals = ["p = 0.008", "p = 0.039", ""]

x = np.arange(len(groups))
width = 0.32

fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=200)
b1 = ax.bar(x - width/2, shardrca, width, label="ShardRCA", color="#2b6cb0")
b2 = ax.bar(x + width/2, baseline, width, label="Baseline tác tử đơn", color="#a0aec0")

for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.015, f"{h:.2f}",
                ha="center", fontsize=9, color="#2d3748")

for xi, p in zip(x, pvals):
    if p:
        ax.text(xi, max(shardrca[xi], baseline[xi]) + 0.09, p,
                ha="center", fontsize=8.6, style="italic", color="#4a5568")

ax.set_ylabel("Hit@1", fontsize=10)
ax.set_ylim(0, 0.85)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("ShardRCA so với baseline tác tử đơn ReAct", fontsize=11.5, fontweight="bold", color="#1a202c")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved", OUT)
