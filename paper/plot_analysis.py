"""
plot_analysis.py — Analysis figures for §5.5:
  Panel A: IS-weight evolution (sumo_darc vs sumo_is)
  Panel B: Discriminator loss convergence (TransDisc / DynDisc / Contrastive)
  Panel C: Q-floor effect on training reward stability
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPDIR = "/home/erzhu419/mine_code/sumo-rl/H2Oplus/experiment_output"
OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "figures", "analysis")

# ── helper: collect train_steps.csv per config across seeds ────────────────
def find_runs(name_str_prefix):
    """Return list of (name_str, seed, csv_path) matching prefix."""
    runs = []
    for d in sorted(glob.glob(os.path.join(EXPDIR, "h2oplus_bus_seed*"))):
        cfg_path = os.path.join(d, "config.json")
        csv_path = os.path.join(d, "train_steps.csv")
        if not os.path.exists(cfg_path) or not os.path.exists(csv_path):
            continue
        try:
            cfg = json.load(open(cfg_path))
            name = cfg.get("name_str", "")
            if name.startswith(name_str_prefix) and "_s" in name:
                seed = int(name.rsplit("_s", 1)[1])
                runs.append((name, seed, csv_path))
        except Exception:
            continue
    return runs

def aggregate_metric(runs, col, smooth=20):
    """Read CSV from each run; return list of (seed, smoothed_array)."""
    out = []
    for name, seed, path in runs:
        try:
            df = pd.read_csv(path)
            if col not in df.columns:
                continue
            arr = df[col].values.astype(float)
            arr = arr[~np.isnan(arr)]
            if len(arr) < smooth:
                continue
            # rolling mean
            kernel = np.ones(smooth) / smooth
            sm = np.convolve(arr, kernel, mode="valid")
            out.append((seed, sm))
        except Exception as e:
            print(f"  skip {name}: {e}")
    return out

# ── PANEL A: IS weight evolution (sqrt_IS_ratio² = w) ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

# DARC = sumo_darc_s* (TransitionDiscriminator default)
darc = aggregate_metric(find_runs("sumo_darc_s"), "sqrt_IS_ratio")
contr = aggregate_metric(find_runs("sumo_is_s"),  "sqrt_IS_ratio")
dyn   = aggregate_metric(find_runs("sumo_dyn_s"), "sqrt_IS_ratio")
ax = axes[0]
for label, runs, color in [
    ("TransitionDisc",   darc, "#8c6d31"),
    ("DynamicsDisc",     dyn,  "#a04040"),
    ("Contrastive",      contr,"#2c5d8a"),
]:
    if not runs: continue
    # squared = w (raw IS); sqrt_IS_ratio is sqrt of clipped w used in loss
    # Plot mean across seeds with shaded std band
    minlen = min(len(arr) for _, arr in runs)
    M = np.stack([arr[:minlen]**2 for _, arr in runs])
    x = np.arange(minlen)
    mean = M.mean(axis=0); std = M.std(axis=0)
    ax.plot(x, mean, color=color, label=f"{label} (n={len(runs)})", lw=1.6)
    ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.18)
ax.axhline(1.0, color="grey", ls=":", lw=0.8, alpha=0.7)
ax.set_xlabel("Training step (smoothed)")
ax.set_ylabel("IS weight $w_{\\rm IS}$")
ax.set_title("(a) IS weight evolution by discriminator")
ax.set_yscale("log")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3, ls=":")

# ── PANEL B: discriminator loss convergence ────────────────────────────────
ax = axes[1]
for label, runs, color in [
    ("TransitionDisc", darc, "#8c6d31"),
    ("DynamicsDisc",   dyn,  "#a04040"),
    ("Contrastive",    contr,"#2c5d8a"),
]:
    runs2 = aggregate_metric(find_runs(label.lower().replace("transitiondisc","sumo_darc_s").replace("dynamicsdisc","sumo_dyn_s").replace("contrastive","sumo_is_s")), "disc_loss", smooth=20)
    if not runs2: continue
    minlen = min(len(arr) for _, arr in runs2)
    M = np.stack([arr[:minlen] for _, arr in runs2])
    x = np.arange(minlen)
    mean = M.mean(axis=0); std = M.std(axis=0)
    ax.plot(x, mean, color=color, label=f"{label}", lw=1.6)
    ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.18)
ax.set_xlabel("Training step (smoothed)")
ax.set_ylabel("Discriminator loss")
ax.set_title("(b) Discriminator convergence")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3, ls=":")

# ── PANEL C: rolling reward (proxy for policy stability) ───────────────────
ax = axes[2]
configs_panel_c = [
    ("sumo_baseline_s", "no IS, no Q-floor",   "#a04040"),
    ("sumo_calql_s",    "no IS, Q-floor only", "#5a3a86"),
    ("sumo_darc_s",     "DARC IS",              "#8c6d31"),
    ("sumo_full_s",     "Contrastive + Q-floor + KL", "#2a6f2a"),
]
for prefix, label, color in configs_panel_c:
    runs = aggregate_metric(find_runs(prefix), "mean_real_rewards", smooth=50)
    if not runs: continue
    minlen = min(len(arr) for _, arr in runs)
    M = np.stack([arr[:minlen] for _, arr in runs])
    x = np.arange(minlen)
    mean = M.mean(axis=0); std = M.std(axis=0)
    ax.plot(x, mean, color=color, label=label, lw=1.5)
    ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.18)
ax.set_xlabel("Training step (smoothed)")
ax.set_ylabel("Mean real-batch reward")
ax.set_title("(c) Training stability across H2O$^+$ variants")
ax.legend(fontsize=7, loc="lower right")
ax.grid(True, alpha=0.3, ls=":")

plt.tight_layout()
plt.savefig(OUT + ".pdf", bbox_inches="tight")
plt.savefig(OUT + ".png", dpi=150, bbox_inches="tight")
print(f"Saved: {OUT}.pdf / .png")
print(f"Runs found: darc={len(darc)} dyn={len(dyn)} contrastive={len(contr)}")
