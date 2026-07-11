"""Redraw the before/after peer-interaction posterior figure (new design, not reusing make_interaction.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "report/figures/fig2_interaction.png"

candidates = ["Dịch vụ\nthượng nguồn\n(root)", "Endpoint\nphát hiện lỗi", "Dịch vụ\nlân cận"]
before = [0.38, 0.34, 0.28]
after = [0.71, 0.19, 0.10]

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), dpi=200, sharey=True)
colors = ["#2f855a", "#c53030", "#a0aec0"]

for ax, values, title in zip(axes, [before, after], ["Trước tương tác peer", "Sau tương tác peer"]):
    bars = ax.bar(candidates, values, color=colors, width=0.58, edgecolor="white")
    ax.set_ylim(0, 0.85)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#1a202c")
    ax.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=9, color="#2d3748")
    ax.tick_params(axis="x", labelsize=8.4)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])

axes[0].set_ylabel("Xác suất hậu nghiệm", fontsize=9.5)

fig.suptitle("Hiệu chỉnh hậu nghiệm sau một vòng phản biện chéo",
             fontsize=11.5, y=1.02, color="#1a202c")

# annotate the swing on the root cause bar
axes[1].annotate("phản biện có bằng chứng\ntừ tác tử trace",
                  xy=(0.32, after[0] - 0.04), xytext=(1.05, 0.55),
                  fontsize=8, color="#2f855a", ha="left",
                  arrowprops=dict(arrowstyle="->", color="#2f855a", lw=1.0))

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved", OUT)
