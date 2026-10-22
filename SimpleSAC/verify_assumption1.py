#!/usr/bin/env python
"""verify_assumption1.py — empirical check of action-conditional domain invariance.

Assumption 1 from the paper states:
    P(domain | s, a, s') = P(domain | s, s')

Equivalently, the density ratio P_real(s' | s, a) / P_sim(s' | s, a) does not
depend on a beyond its dependence on (s, s').

This script tests that assumption from *data* (not from a learned model — the
learned model passing the test is tautological, since we train it to ignore a).

Method:
    1. Load real transitions (SUMO offline h5) and sim transitions (sim-core
       rollout h5; generate with rollout_sim_for_verification.py if needed).
    2. Discretize action into K_a bins.
    3. Project (s, s') concatenation to 2D via PCA, then bin into K_ss^2 grid.
    4. For each cell (bin_ss_in, bin_ss_out, bin_a), estimate transition
       probability P_d(s' ∈ bin_ss_out | s ∈ bin_ss_in, a ∈ bin_a) per domain.
    5. Compute ratio ρ = P_real / P_sim per cell.
    6. Assumption 1 holds ⟺ ρ varies primarily across (bin_ss_in, bin_ss_out)
       and little across bin_a. Report:
          action_dep_ratio = mean over (ss_in, ss_out) [ std_a(log ρ) ]
                           / mean over a            [ std_(ss_in, ss_out)(log ρ) ]
       Small ratio → assumption supported.

Usage:
    python verify_assumption1.py \\
        --real_h5 ../bus_h2o/datasets_v2/merged_all_v2.h5 \\
        --sim_h5 ../experiment_output/sim_rollouts_for_verification.h5 \\
        --out_dir ../experiment_output/assumption1_check

Outputs:
    action_dep_ratio.txt    — single scalar, the headline metric
    heatmap_per_action.pdf  — log ρ heatmaps across action bins (for figure)
    summary_table.csv       — per-cell ratio, variance breakdown
"""

import argparse
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_transitions(path, max_n=None):
    """Load (s, a, s') from h5."""
    with h5py.File(path, "r") as f:
        n = f["observations"].shape[0]
        if max_n and n > max_n:
            idx = np.random.RandomState(0).permutation(n)[:max_n]
            idx.sort()
            s = np.array(f["observations"][idx])
            a = np.array(f["actions"][idx])
            sp = np.array(f["next_observations"][idx])
        else:
            s = np.array(f["observations"])
            a = np.array(f["actions"])
            sp = np.array(f["next_observations"])
    return s, a, sp


def bin_actions(a, n_bins=3):
    """Discretize a into n_bins bins based on the hold decision (a[:, 0]).

    For our transit MDP, a[:, 0] > 0 triggers holding. We split into:
        bin 0: no-hold (a[:, 0] ≤ 0)
        bin 1..n-1: hold, split by a[:, 1] (duration) quantiles among hold decisions.
    """
    hold_mask = a[:, 0] > 0
    bins = np.zeros(a.shape[0], dtype=int)
    if hold_mask.any():
        hold_dur = a[hold_mask, 1]
        if n_bins > 2:
            q = np.quantile(hold_dur, np.linspace(0, 1, n_bins)[1:-1])
            bins[hold_mask] = 1 + np.searchsorted(q, hold_dur)
        else:
            bins[hold_mask] = 1
    return bins


def bin_states(sspair, scaler, pca, grid_edges):
    """Standardize → PCA 2D → bin into a 2D grid."""
    proj = pca.transform(scaler.transform(sspair))
    bin_x = np.clip(np.searchsorted(grid_edges[0], proj[:, 0]), 0, len(grid_edges[0])) - 1
    bin_y = np.clip(np.searchsorted(grid_edges[1], proj[:, 1]), 0, len(grid_edges[1])) - 1
    return bin_x * len(grid_edges[1]) + bin_y  # flatten to single bin id


def compute_cell_counts(s_bins, a_bins, n_s_bins, n_a_bins):
    """Count occurrences per (state_bin, action_bin) cell."""
    counts = np.zeros((n_s_bins, n_a_bins), dtype=np.int64)
    for sb, ab in zip(s_bins, a_bins):
        counts[sb, ab] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_h5", required=True)
    ap.add_argument("--sim_h5", required=True)
    ap.add_argument("--out_dir", default="./assumption1_check")
    ap.add_argument("--n_action_bins", type=int, default=3)
    ap.add_argument("--grid_size", type=int, default=8)
    ap.add_argument("--max_n", type=int, default=200000,
                    help="subsample per domain to this many transitions (memory cap)")
    ap.add_argument("--min_cell_count", type=int, default=30,
                    help="drop cells with fewer than this many samples in either domain")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────
    print(f"Loading real from {args.real_h5} (cap {args.max_n}) ...")
    s_r, a_r, sp_r = load_transitions(args.real_h5, args.max_n)
    print(f"Loading sim  from {args.sim_h5} (cap {args.max_n}) ...")
    s_s, a_s, sp_s = load_transitions(args.sim_h5, args.max_n)
    print(f"  real N={len(s_r)}, sim N={len(s_s)}")

    # ── Standardize then fit PCA on pooled (s, s') for consistent projection ──
    sspair_r = np.concatenate([s_r, sp_r], axis=1)
    sspair_s = np.concatenate([s_s, sp_s], axis=1)
    pooled = np.concatenate([sspair_r, sspair_s], axis=0)
    scaler = StandardScaler()
    pooled_scaled = scaler.fit_transform(pooled)
    pca = PCA(n_components=2)
    pca.fit(pooled_scaled)
    print(f"PCA explained variance ratio (post-standardize): {pca.explained_variance_ratio_}")

    # ── Grid edges from quantiles of pooled projection ──────────
    pooled_proj = pca.transform(pooled_scaled)
    qs = np.linspace(0, 1, args.grid_size + 1)
    edges_x = np.quantile(pooled_proj[:, 0], qs)[1:-1]
    edges_y = np.quantile(pooled_proj[:, 1], qs)[1:-1]
    n_s_bins = args.grid_size * args.grid_size

    # ── Bin ──────────────────────────────────────────────────────
    sb_r = bin_states(sspair_r, scaler, pca, (edges_x, edges_y))
    sb_s = bin_states(sspair_s, scaler, pca, (edges_x, edges_y))
    ab_r = bin_actions(a_r, args.n_action_bins)
    ab_s = bin_actions(a_s, args.n_action_bins)

    counts_r = compute_cell_counts(sb_r, ab_r, n_s_bins, args.n_action_bins)
    counts_s = compute_cell_counts(sb_s, ab_s, n_s_bins, args.n_action_bins)

    # ── Compute cell probabilities per domain ───────────────────
    # For each action bin a_k, P_real(ss_cell | a_k) = count_real(ss_cell, a_k) / sum_cell count_real(·, a_k)
    p_r = counts_r / (counts_r.sum(axis=0, keepdims=True) + 1e-9)
    p_s = counts_s / (counts_s.sum(axis=0, keepdims=True) + 1e-9)

    # Log-ratio per cell (mask sparse cells)
    mask = (counts_r >= args.min_cell_count) & (counts_s >= args.min_cell_count)
    log_ratio = np.full_like(p_r, np.nan, dtype=np.float64)
    log_ratio[mask] = np.log(p_r[mask] / p_s[mask])

    # ── Action-dependence metric ─────────────────────────────────
    # For each ss-cell, variance of log_ratio across action bins (only cells with enough coverage)
    std_across_a = np.nanstd(log_ratio, axis=1)  # (n_s_bins,)
    std_across_s = np.nanstd(log_ratio, axis=0)  # (n_a_bins,)

    n_rows_valid = int(np.isfinite(std_across_a).sum())
    n_cols_valid = int(np.isfinite(std_across_s).sum())
    mean_std_a = np.nanmean(std_across_a)
    mean_std_s = np.nanmean(std_across_s)

    action_dep_ratio = mean_std_a / (mean_std_s + 1e-9)

    # ── Write summary ────────────────────────────────────────────
    print("─" * 60)
    print(f"Valid cells (ss-bins × action-bins with ≥{args.min_cell_count} samples each side):")
    print(f"  ss-bins with valid rows: {n_rows_valid} / {n_s_bins}")
    print(f"  action-bins with valid columns: {n_cols_valid} / {args.n_action_bins}")
    print(f"mean std_a (log ρ)   = {mean_std_a:.4f}")
    print(f"mean std_ss (log ρ)  = {mean_std_s:.4f}")
    print(f"action-dependence ratio = {action_dep_ratio:.4f}")
    if action_dep_ratio < 0.3:
        print("  → Small ratio: log ρ varies primarily with (s, s'), not a.")
        print("  → Assumption 1 supported on this data.")
    elif action_dep_ratio < 0.6:
        print("  → Moderate ratio: Assumption 1 is borderline.")
    else:
        print("  → Large ratio: Assumption 1 LIKELY VIOLATED on this data.")
    print("─" * 60)

    with open(os.path.join(args.out_dir, "action_dep_ratio.txt"), "w") as f:
        f.write(f"action_dep_ratio={action_dep_ratio:.6f}\n")
        f.write(f"mean_std_a={mean_std_a:.6f}\n")
        f.write(f"mean_std_ss={mean_std_s:.6f}\n")
        f.write(f"n_action_bins={args.n_action_bins}\n")
        f.write(f"grid_size={args.grid_size}\n")
        f.write(f"min_cell_count={args.min_cell_count}\n")
        f.write(f"n_real={len(s_r)}\n")
        f.write(f"n_sim={len(s_s)}\n")

    np.savetxt(os.path.join(args.out_dir, "log_ratio_matrix.csv"),
               log_ratio, delimiter=",", fmt="%.4f")

    # ── Heatmap figure ────────────────────────────────────────────
    fig, axes = plt.subplots(1, args.n_action_bins, figsize=(4 * args.n_action_bins, 3.5), sharey=True)
    if args.n_action_bins == 1:
        axes = [axes]
    vmin = np.nanpercentile(log_ratio, 5)
    vmax = np.nanpercentile(log_ratio, 95)
    for k, ax in enumerate(axes):
        grid = log_ratio[:, k].reshape(args.grid_size, args.grid_size)
        im = ax.imshow(grid, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"a-bin {k}")
        ax.set_xlabel("PCA-1 bin")
        if k == 0:
            ax.set_ylabel("PCA-2 bin")
    fig.colorbar(im, ax=axes, shrink=0.8, label=r"$\log \rho(\mathrm{cell}, a)$")
    fig.suptitle(f"Log-ratio per (s,s') cell across action bins  "
                 f"(action-dependence ratio = {action_dep_ratio:.3f})")
    plt.savefig(os.path.join(args.out_dir, "heatmap_per_action.pdf"),
                bbox_inches="tight")
    print(f"Saved heatmaps to {os.path.join(args.out_dir, 'heatmap_per_action.pdf')}")


if __name__ == "__main__":
    main()
