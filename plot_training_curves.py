"""
Plot MaskablePPO training reward curves from TensorBoard CSVs.
Output: logs/v3/TensorBoard/reward_curves.svg
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import scienceplots

plt.style.use(["science", "nature", "grid"])
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "Arial"

from tum_colors import TUM_BLUE, TUM_DARK_BLUE_1, TUM_GREEN, TUM_ORANGE, GRAY_80

CSV_DIR = "logs/v3/TensorBoard"

# Ordered highest→lowest mean reward so legend matches visual top→bottom
SCENARIOS = [
    ("S4", "Scenario 4", TUM_BLUE),
    ("S2", "Scenario 2", TUM_ORANGE),
    ("SQ", "Status Quo",  GRAY_80),
    ("S3", "Scenario 3", TUM_GREEN),
    ("S1", "Scenario 1", TUM_DARK_BLUE_1),
]

fig, ax = plt.subplots(figsize=(6, 3.5))

max_step = 0
for tag, label, color in SCENARIOS:
    df = pd.read_csv(f"{CSV_DIR}/{tag}_MaskablePPO_1.csv")
    df = df.sort_values("Step")
    ax.plot(df["Step"], df["Value"], color=color, linewidth=0.9, label=label)
    max_step = max(max_step, int(df["Step"].max()))

ax.axvline(max_step, color="black", linewidth=0.8, linestyle="--", zorder=5)


def x_fmt(x, _):
    if x == 0:
        return "0"
    return f"{int(x / 1e6)}M"


def y_fmt(y, _):
    val = y / 1e7
    rounded = round(val, 1)
    if rounded == int(rounded):
        return f"{int(rounded)}e+7"
    return f"{rounded}e+7"


ax.xaxis.set_major_locator(MultipleLocator(1e6))
ax.xaxis.set_major_formatter(FuncFormatter(x_fmt))
ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))

ax.set_xlabel("Training step")
ax.set_ylabel("Mean episodic reward")
ax.legend(fontsize=7, loc="lower right", frameon=True)

plt.tight_layout()
out_path = os.path.join(CSV_DIR, "reward_curves.svg")
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved: {out_path}")
plt.show()
