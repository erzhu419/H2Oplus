#!/usr/bin/env python3
"""Plot the ten verified paired block effects, without invented uncertainty.

Run from any directory. Values are read only from data/verified_results.json.
No smoothing, confidence interval, resampling, or inference is introduced.
"""

import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "h2oplus_tits_matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "verified_results.json"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
data = json.loads(DATA.read_text())
blocks = sorted(data["block_differences"], key=int)
assert len(blocks) == 10
primary = np.array([data["block_differences"][b]["primary"] for b in blocks])
completion = 100 * np.array([data["block_differences"][b]["completion"] for b in blocks])
assert np.all(primary < 0)
assert np.all(completion > 0)
assert abs(primary.mean() - data["means"]["primary"]) < 1e-10
assert abs(completion.mean() - 100 * data["means"]["completion"]) < 1e-10

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.65,
    "ytick.major.width": 0.65,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.facecolor": "white",
})

fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), sharey=True)
fig.subplots_adjust(left=0.077, right=0.987, bottom=0.235, top=0.805, wspace=0.205)
positions = np.arange(1, 11)
dot_color = "#225879"
mean_color = "#454545"

for ax in axes:
    ax.set_ylim(10.55, 0.45)
    ax.set_yticks(positions)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=3, pad=3)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    ax.axvline(0, color="#222222", lw=0.8, zorder=2)
    ax.set_xlabel("Proposed minus matched reference", labelpad=6)

axes[0].scatter(primary, positions, s=22, marker="o", color=dot_color,
                edgecolors="white", linewidths=0.45, zorder=4)
axes[0].axvline(primary.mean(), color=mean_color, lw=0.9,
                linestyle=(0, (5, 3)), zorder=3)
axes[0].set_xlim(-120, 8)
axes[0].set_xticks([-120, -90, -60, -30, 0])
axes[0].set_ylabel("Independent test block", labelpad=6)
axes[0].set_title("(a) Generalized passenger time (s/departed passenger)\n"
                   f"Descriptive mean: {primary.mean():.2f} s", pad=8)

axes[1].scatter(completion, positions, s=22, marker="s", color=dot_color,
                edgecolors="white", linewidths=0.45, zorder=4)
axes[1].axvline(completion.mean(), color=mean_color, lw=0.9,
                linestyle=(0, (5, 3)), zorder=3)
axes[1].axvline(-0.3, color="#222222", lw=1.1,
                linestyle=(0, (1, 2)), zorder=3)
axes[1].set_xlim(-0.37, 0.62)
axes[1].set_xticks([-0.3, 0.0, 0.3, 0.6])
axes[1].set_xticklabels(["−0.3", "0", "+0.3", "+0.6"])
axes[1].set_title("(b) Journey completion (percentage points)\n"
                   f"Descriptive mean: +{completion.mean():.3f} pp", pad=8)
axes[1].tick_params(axis="y", left=False)

legend = [
    Line2D([0], [0], color="#222222", lw=0.8, label="No difference"),
    Line2D([0], [0], color=mean_color, lw=0.9, linestyle=(0, (5, 3)),
           label="Descriptive mean"),
    Line2D([0], [0], color="#222222", lw=1.1, linestyle=(0, (1, 2)),
           label="Completion noninferiority margin (−0.3 pp)"),
]
fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.52, 0.017),
           ncol=3, frameon=False, fontsize=8, handlelength=2.5,
           columnspacing=1.5, handletextpad=0.6)

metadata = {
    "Title": "Paired effects of the added completion rule in ten independent evaluation blocks",
    "Subject": "Existing verified endpoint data; block means over five training seeds and three demand levels",
    "Creator": "Matplotlib; plot_block_effects.py",
    "Keywords": "bus holding, passenger time, completion, independent block, paired comparison",
}
fig.savefig(OUT / "block_effects.pdf", metadata=metadata)
fig.savefig(OUT / "block_effects.png", dpi=300)
fig.savefig(OUT / "block_effects.svg")
plt.close(fig)
print(f"Saved vector PDF/SVG and 300-dpi PNG to {OUT}")
