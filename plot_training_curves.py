"""
Plot MaskablePPO training reward curves from TensorBoard CSVs.
Output: logs/v3/TensorBoard/reward_curves.svg
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import scienceplots

plt.style.use(["science", "nature", "grid"])
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.latex.preamble"] = (
    r"\usepackage{helvet}"
    r"\renewcommand\familydefault{\sfdefault}"
    r"\usepackage[T1]{fontenc}"
    r"\usepackage{eurosym}"
)
plt.rcParams["axes.labelsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8

from tum_colors import TUM_BLUE, TUM_DARK_BLUE_1, TUM_GREEN, TUM_ORANGE, GRAY_80

CSV_DIR = "logs/v3/TensorBoard"

# Ordered highest→lowest mean reward so legend matches visual top→bottom
SCENARIOS = [
    ("SQ", "Status Quo",   GRAY_80),
    ("S1", "Scenario 1",   TUM_DARK_BLUE_1),
    ("S2", "Scenario 2",   TUM_ORANGE),
    ("S3", "Scenario 3",   TUM_GREEN),
    ("S4", "Scenario 4",   TUM_BLUE),
]

# Load all data upfront to compute axis limits
dfs = {}
for tag, _label, _color in SCENARIOS:
    df = pd.read_csv(f"{CSV_DIR}/{tag}_MaskablePPO_1.csv").sort_values("Step")
    dfs[tag] = df

all_vals = np.concatenate([dfs[tag]["Value"].values for tag, _, _ in SCENARIOS])
max_step  = max(int(dfs[tag]["Step"].max()) for tag, _, _ in SCENARIOS)

TICK = 0.5e7
bot_lo = np.floor(all_vals.min() / TICK) * TICK   # -7e7
bot_hi = -3.7e7
top_lo = -1.5 * TICK                                # -0.75e7
top_hi = 0.0                                        # 0 at the top

# Height ratios: equal pixel height per TICK in both panels
n_bot = (bot_hi - bot_lo) / TICK                   # 6.6
n_top = 1.5

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1,
    sharex=True,
    figsize=(6, 4.2),
    gridspec_kw={"height_ratios": [n_top, n_bot]},
)
fig.subplots_adjust(left=0.13, right=0.97, top=0.97, bottom=0.11, hspace=0.04)

for tag, label, color in SCENARIOS:
    df = dfs[tag]
    kw = dict(color=color, linewidth=0.9)
    ax_top.plot(df["Step"], df["Value"], **kw, label=label)
    ax_bot.plot(df["Step"], df["Value"], **kw)

for ax in (ax_top, ax_bot):
    ax.axvline(max_step, color="black", linewidth=0.8, linestyle="--", zorder=5)

# Panel limits — small padding so edge ticks aren't clipped
ax_top.set_ylim(top_lo, top_hi)
ax_bot.set_ylim(bot_lo, bot_hi)


def x_fmt(x, _):
    return str(int(x / 1e6))


def y_fmt(y, _):
    val = y / 1e6
    return str(int(val)) if val == int(val) else str(round(val, 1))


for ax in (ax_top, ax_bot):
    ax.xaxis.set_major_locator(MultipleLocator(1e6))
    ax.xaxis.set_major_formatter(FuncFormatter(x_fmt))
for ax in (ax_top, ax_bot):
    ax.yaxis.set_major_locator(MultipleLocator(TICK))
    ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))

# Hide inner spines to create the visual break
ax_top.spines["bottom"].set_visible(False)
ax_bot.spines["top"].set_visible(False)
ax_top.xaxis.tick_top()
ax_top.tick_params(labeltop=True, bottom=False)
ax_bot.xaxis.set_ticks_position("bottom")
ax_bot.tick_params(axis="x", labeltop=False, labelbottom=False)

# Break markers (identical to compare_to_baselines.py)
d = 0.5
bk = dict(marker=[(-1, -d), (1, d)], markersize=12,
          linestyle="none", color="k", mec="k", mew=1, clip_on=False)
ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **bk)
ax_bot.plot([0, 1], [1, 1], transform=ax_bot.transAxes, **bk)

ax_bot.set_xlim(left=0)
ax_top.set_xlabel(r"Training step ($10^6$)")
ax_top.xaxis.set_label_position("top")
ax_bot.set_ylabel(r"Mean episodic reward ($10^6$\,\euro)")
center_y = 0.5 + n_top / (2 * n_bot)
ax_bot.yaxis.set_label_coords(-0.06, center_y)
ax_top.legend(fontsize=7, loc="upper right", frameon=True,
              labelspacing=0.25, borderpad=0.3)

# Hide gridlines and tick marks at axis boundaries (keep labels)
fig.canvas.draw()
_tol = 1e3
for ax, bounds in [(ax_top, [top_hi]), (ax_bot, [bot_lo])]:
    for gl in ax.yaxis.get_gridlines():
        if any(abs(gl.get_ydata()[0] - b) < _tol for b in bounds):
            gl.set_visible(False)
    for tick in ax.yaxis.get_major_ticks():
        if any(abs(tick.get_loc() - b) < _tol for b in bounds):
            tick.tick1line.set_visible(False)
            tick.tick2line.set_visible(False)

for ext, kw in [("pdf", {}), ("png", {"dpi": 150})]:
    out_path = os.path.join(CSV_DIR, f"reward_curves.{ext}")
    fig.savefig(out_path, bbox_inches="tight", **kw)
    print(f"Saved: {out_path}")
plt.show()
