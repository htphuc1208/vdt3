"""Redraw the ShardRCA pipeline figure for the VDT2026 report (new design, not reusing make_shard_rca.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches

OUT = "report/figures/fig1_architecture.png"

STAGES = [
    ("Planner /\nShard builder", "#2b6cb0"),
    ("Tác tử điều tra\ncô lập (song song)", "#2f855a"),
    ("Tương tác peer\n(phản biện + hiệu chỉnh)", "#c05621"),
    ("Hợp nhất\n(log-opinion pool)", "#6b46c1"),
    ("Tái xếp hạng\nnhân quả", "#2b6cb0"),
    ("Xác minh bằng chứng\n(top vs runner-up)", "#b83280"),
]

fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=200)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

box_w, box_h = 1.42, 1.35
gap = 0.28
n = len(STAGES)
total_w = n * box_w + (n - 1) * gap
x0 = (10 - total_w) / 2
y_mid = 3.15

centers = []
for i, (label, color) in enumerate(STAGES):
    x = x0 + i * (box_w + gap)
    centers.append((x + box_w / 2, y_mid))
    box = FancyBboxPatch(
        (x, y_mid - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.4, edgecolor=color, facecolor=color + "22",
    )
    ax.add_patch(box)
    ax.text(x + box_w / 2, y_mid, label, ha="center", va="center",
            fontsize=8.6, color="#1a202c", fontweight="medium", linespacing=1.3)

for i in range(n - 1):
    x1 = centers[i][0] + box_w / 2
    x2 = centers[i + 1][0] - box_w / 2
    ax.add_patch(FancyArrowPatch((x1, y_mid), (x2, y_mid),
                                  arrowstyle="-|>", mutation_scale=14,
                                  linewidth=1.3, color="#4a5568"))

# investigator agents fan-out under stage 2 to show parallelism
agent_y = 1.15
ax2x = centers[1][0]
for dx, tag in zip([-0.85, 0, 0.85], ["Metric", "Log", "Trace"]):
    ax.add_patch(FancyArrowPatch((ax2x, y_mid - box_h / 2), (ax2x + dx, agent_y + 0.32),
                                  arrowstyle="-", linewidth=0.9, color="#2f855a", alpha=0.6))
    small = FancyBboxPatch((ax2x + dx - 0.42, agent_y - 0.28), 0.84, 0.56,
                            boxstyle="round,pad=0.03,rounding_size=0.06",
                            linewidth=1.0, edgecolor="#2f855a", facecolor="#f0fff4")
    ax.add_patch(small)
    ax.text(ax2x + dx, agent_y, tag, ha="center", va="center", fontsize=7.6, color="#22543d")

# blackboard label under the fan-out
ax.text(ax2x, 0.35, "Blackboard chung (typed, có con trỏ bằng chứng)",
        ha="center", va="center", fontsize=7.6, style="italic", color="#4a5568")

# top annotation: input / output
ax.text(centers[0][0], y_mid + box_h / 2 + 0.35, "Ca sự cố + danh mục\nứng viên quan sát được",
        ha="center", va="bottom", fontsize=7.4, color="#2d3748", linespacing=1.25)
ax.annotate("", xy=(centers[0][0], y_mid + box_h/2), xytext=(centers[0][0], y_mid + box_h/2 + 0.3),
            arrowprops=dict(arrowstyle="-|>", color="#2d3748", lw=1.1))

ax.text(centers[-1][0], y_mid + box_h / 2 + 0.35, "Nguyên nhân gốc\n(component, reason) + transcript",
        ha="center", va="bottom", fontsize=7.4, color="#2d3748", linespacing=1.25)
ax.annotate("", xy=(centers[-1][0], y_mid + box_h/2), xytext=(centers[-1][0], y_mid + box_h/2 + 0.3),
            arrowprops=dict(arrowstyle="-|>", color="#2d3748", lw=1.1))

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved", OUT)
