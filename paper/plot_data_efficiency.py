"""
plot_data_efficiency.py — Group E data-efficiency curve for the paper.
Pure offline (Cal-QL) vs H2O+ at 4 offline buffer sizes.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "figures", "data_efficiency")

# From paper §5.4 tab:data_size (5 seeds each)
sizes = np.array([100, 200, 400, 675])  # K transitions

pure_off_mean = np.array([-789, -750, -749, -750])  # K reward
pure_off_std  = np.array([  42,   14,   24,   28])

h2o_mean = np.array([-697, -686, -689, -705])
h2o_std  = np.array([  61,   48,   32,   38])

ep39 = -666
zero_hold = -1600
pure_online = -1654
h2o_sumo_best = -646       # sumo_darc 4-seed mean — best H2O+ (with SUMO-online)
h2o_sumo_best_std = 16

fig, ax = plt.subplots(figsize=(8.5, 5.2))

# Two H2O+ curves at varying offline size
ax.errorbar(sizes, pure_off_mean, yerr=pure_off_std,
            marker="s", markersize=8, capsize=5, lw=1.8,
            color="#8c6d31", label="Pure offline RL (Cal-QL)", zorder=4)
ax.errorbar(sizes, h2o_mean, yerr=h2o_std,
            marker="o", markersize=8, capsize=5, lw=1.8,
            color="#2c5d8a",
            label="H2O$^+$ — SIM-online (cheap, $40\\times$ faster)", zorder=5)

# H2O+ SUMO-online ceiling — single point at full data, but plot as a band
sumo_x = np.array([sizes[-1]])
ax.errorbar(sumo_x, [h2o_sumo_best], yerr=[h2o_sumo_best_std],
            marker="*", markersize=18, capsize=5, lw=0,
            color="#1a5e1a", markeredgecolor="black", markeredgewidth=0.7,
            label="H2O$^+$ — SUMO-online (full fidelity, ours best)",
            zorder=6)
# horizontal line at h2o_sumo_best for visual ceiling
ax.axhline(h2o_sumo_best, color="#1a5e1a", lw=1.0, ls="-.", alpha=0.6, zorder=2)

# Reference baselines as horizontal lines
ax.axhline(ep39, color="#a06030", lw=1.4, ls="--", alpha=0.95, zorder=3)
ax.text(sizes[0]-5, ep39+8, f"RE-SAC SUMO expert ep39 ({ep39}K)",
        fontsize=9, color="#a06030", va="bottom", weight="bold")

ax.axhline(pure_online, color="#a04040", lw=1.2, ls=":", alpha=0.7, zorder=2)
ax.text(sizes[0]-5, pure_online+30,
        f"Pure online RL ($-${-pure_online}K, 5 seeds)",
        fontsize=8.5, color="#a04040", va="bottom")

# Annotate the data-efficiency claim (SIM-online H2O+ vs Pure offline)
ax.annotate("", xy=(200, -686), xytext=(675, -750),
            arrowprops=dict(arrowstyle="->", color="#2c5d8a", lw=1.3, ls="--"))
ax.text(440, -706,
        "SIM-online H2O$^+$ at 200K\n$\\geq$ pure offline at 675K",
        fontsize=8, color="#2c5d8a", ha="center", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#2c5d8a", alpha=0.9))

# Annotate the SUMO-online H2O+ ceiling
ax.annotate("Best H2O$^+$ ($-646\\pm 16$K)\nslightly exceeds ep39 on default scenario",
            xy=(sumo_x[0], h2o_sumo_best), xytext=(450, -625),
            fontsize=8.5, color="#1a5e1a",
            ha="center", style="italic", weight="bold",
            arrowprops=dict(arrowstyle="->", color="#1a5e1a", lw=1.3),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f4ffe8",
                      edgecolor="#1a5e1a", alpha=0.95))

ax.set_xlabel("Offline buffer size  (K transitions)", fontsize=10)
ax.set_ylabel("SUMO eval cumulative reward (K)", fontsize=10)
ax.set_xticks(sizes)
ax.set_xticklabels(["100K", "200K", "400K", "675K (full)"])
ax.grid(True, alpha=0.3, ls=":")
ax.set_ylim(-1700, -550)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)
ax.set_title("Data efficiency on SUMO eval: SIM-online H2O$^+$ closes most of the gap to the RE-SAC SUMO expert;\nfull-fidelity SUMO-online H2O$^+$ slightly exceeds ep39 on the default scenario",
             fontsize=9.5)

plt.tight_layout()
plt.savefig(OUT + ".pdf", bbox_inches="tight")
plt.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}.pdf / .png")
