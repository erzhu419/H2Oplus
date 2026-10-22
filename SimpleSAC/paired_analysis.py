"""paired_analysis.py
Compute paired-scenario statistics and Pareto plot from multiseed_eval/*.json.

Outputs:
  - experiment_output/multiseed_eval/paired_stats.csv   (Wilcoxon paired tests + bootstrap CI)
  - paper/figures/pareto.pdf                              (reward vs passenger wait Pareto)
  - paper/figures/per_scenario_heatmap.pdf                (9-cell reward heatmap per method)

Reads existing 54 JSONs (6 methods x 3 SUMO seeds x 3 OD scales) plus any newer
v3/v4/v5/r2 results in the same directory.
"""
import json, glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = '/home/erzhu419/mine_code/sumo-rl/H2Oplus'
JSON_DIR = f'{ROOT}/experiment_output/multiseed_eval'
FIG_DIR = f'{ROOT}/paper/figures'
os.makedirs(FIG_DIR, exist_ok=True)

# Canonical method names (matches Tab V row labels)
CANON = {
    'h2oplus_full':       'H2O+ TransDisc only',
    'h2oplus_darc_calql': 'H2O+ TransDisc + Q-floor',
    'pure_online_sac':    'Pure-online SAC (200ep, legacy)',
    'pure_online_sac_1000ep_s42':  'Pure-online SAC 1000ep s42',
    'pure_online_sac_1000ep_s123': 'Pure-online SAC 1000ep s123',
    'pure_online_sac_1000ep_s789': 'Pure-online SAC 1000ep s789',
    'bc_full':            'BC (full Doff)',
    'daganzo':            'Daganzo (alpha=0.6)',
    'daganzo_a03':        'Daganzo (alpha=0.3)',
    'daganzo_a04':        'Daganzo (alpha=0.4)',
    'zero_hold':          'Zero-hold',
    'ep39':               'RE-SAC SUMO expert ep39',
}

PARETO_GROUPS = {
    'ep39': 'RE-SAC SUMO expert',
    'h2oplus_full': 'H2O+ TransDisc',
    'h2oplus_darc_calql': 'H2O+ TransDisc + Q-floor',
    'sim_contrastive': 'H2O+ Contrastive (SIM)',
    'rlpd_nosnap': 'RLPD',
    'wsrl_nosnap': 'WSRL',
    'td3bc': 'TD3+BC',
    'iql': 'IQL',
    'awac': 'AWAC',
    'pure_online_sac_1000ep': 'Pure-online SAC 1000ep',
    'pure_online_sac': 'Pure-online SAC 200ep',
    'bc_ep39': 'BC ep39-only',
    'bc_full': 'BC full Doff',
    'zero_hold': 'Zero-hold',
    'daganzo_a03': 'Daganzo alpha=0.3',
    'daganzo_a04': 'Daganzo alpha=0.4',
    'daganzo': 'Daganzo alpha=0.6',
}

PARETO_ORDER = [
    'ep39',
    'h2oplus_full',
    'h2oplus_darc_calql',
    'sim_contrastive',
    'rlpd_nosnap',
    'wsrl_nosnap',
    'td3bc',
    'iql',
    'awac',
    'pure_online_sac_1000ep',
    'pure_online_sac',
    'bc_ep39',
    'bc_full',
    'zero_hold',
    'daganzo_a03',
    'daganzo_a04',
    'daganzo',
]


def pareto_group(method):
    """Map training-seed-specific eval files to the paper-level Figure 5 rows."""
    prefix_groups = (
        ('sim_contrastive_s', 'sim_contrastive'),
        ('rlpd_nosnap_s', 'rlpd_nosnap'),
        ('wsrl_nosnap_s', 'wsrl_nosnap'),
        ('td3bc_s', 'td3bc'),
        ('iql_s', 'iql'),
        ('awac_s', 'awac'),
        ('bc_ep39_s', 'bc_ep39'),
    )
    for prefix, group in prefix_groups:
        if method.startswith(prefix):
            return group
    if method in PARETO_GROUPS:
        return method
    return None

def load_all():
    rows = []
    for fp in sorted(glob.glob(f'{JSON_DIR}/*.json')):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        base = os.path.basename(fp).replace('.json', '')
        if '_sumo' not in base:
            continue
        method, rest = base.rsplit('_sumo', 1)
        seed_str, od_str = rest.split('_od')
        # Group the 3 training seeds of pure_online 1000ep into a single canonical method
        canonical = method
        if method.startswith('pure_online_sac_1000ep_s'):
            canonical = 'pure_online_sac_1000ep'
        # Compute headway-CV avg across lines
        cv_d = d.get('headway_cv_per_line', {}) or {}
        hw_cv_avg = float(np.mean(list(cv_d.values()))) if cv_d else np.nan
        rows.append({
            'method': canonical,
            'method_raw': method,    # keep training-seed distinction for hierarchical analysis
            'sumo_seed': int(seed_str),
            'od_scale': float(od_str),
            'cum_reward': d.get('cumulative_reward', np.nan),
            'pax_wait_mean': d.get('passenger_wait_mean', np.nan),
            'pax_wait_p90': d.get('passenger_wait_p90', np.nan),
            'large_gap_rate': d.get('large_gap_rate', np.nan),
            'jain_fairness': d.get('jain_fairness', np.nan),
            'hw_cv_avg': hw_cv_avg,
        })
    return pd.DataFrame(rows)

def hw_cv_avg(d):
    cv = d.get('headway_cv_per_line', {})
    if not cv: return np.nan
    return float(np.mean(list(cv.values())))

def main():
    df = load_all()
    # Add per-method headway CV from JSONs
    df['n'] = 1
    print(f"Loaded {len(df)} eval JSONs across {df['method'].nunique()} methods")
    print(df.groupby('method').size())

    # Pivot wide for paired analysis: rows = (sumo_seed, od_scale), cols = method, values = cum_reward
    pivot = df.pivot_table(index=['sumo_seed', 'od_scale'], columns='method',
                           values='cum_reward', aggfunc='mean')

    # Paired Wilcoxon: H2O+ TransDisc-only vs each other method, on the 9 paired scenarios
    BASE = 'h2oplus_full'
    if BASE not in pivot.columns:
        print(f"WARN: base method {BASE} missing")
        return
    others = [c for c in pivot.columns if c != BASE]
    rows = []
    base_vals = pivot[BASE].dropna()
    for m in others:
        joint = pivot[[BASE, m]].dropna()
        if len(joint) < 3:
            continue
        diff = joint[BASE] - joint[m]   # positive = base better (higher reward)
        try:
            stat, p = wilcoxon(diff)
        except ValueError:
            stat, p = (np.nan, np.nan)
        # Bootstrap 95% CI for mean diff
        rng = np.random.default_rng(42)
        boots = [rng.choice(diff.values, size=len(diff), replace=True).mean()
                 for _ in range(2000)]
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
        rows.append({
            'method': m,
            'n_paired': len(diff),
            'mean_diff_K': diff.mean()/1000,
            'ci95_lo_K': ci_lo/1000,
            'ci95_hi_K': ci_hi/1000,
            'wilcoxon_p': p,
        })
    paired = pd.DataFrame(rows).sort_values('mean_diff_K', ascending=False)
    out = f'{JSON_DIR}/paired_stats.csv'
    paired.to_csv(out, index=False)
    print(f"\nPaired Wilcoxon (base = {BASE}, positive mean_diff = base reward higher):")
    print(paired.to_string(index=False))
    print(f"saved {out}")

    # Per-method summary table (replaces Tab V data + paired SE)
    summary = df.groupby('method').agg(
        cum_reward_mean=('cum_reward', 'mean'),
        cum_reward_se=('cum_reward', lambda x: x.std() / np.sqrt(len(x)) if len(x)>1 else np.nan),
        pax_wait=('pax_wait_mean', 'mean'),
        hw_cv=('hw_cv_avg', 'mean'),
        large_gap=('large_gap_rate', 'mean'),
        jain=('jain_fairness', 'mean'),
        n_scenarios=('cum_reward', 'count'),
    ).dropna(subset=['cum_reward_mean'])
    summary.to_csv(f'{JSON_DIR}/method_summary.csv')
    print("\n=== Per-method summary (mean ± SE across scenarios) ===")
    for m, row in summary.iterrows():
        print(f"  {m:36s}: reward={row['cum_reward_mean']/1000:>7.0f}±{row['cum_reward_se']/1000:>5.0f}K  "
              f"pax_wait={row['pax_wait']:>5.1f}s  hw_cv={row['hw_cv']:>.3f}  "
              f"jain={row['jain']:>.3f}  n={row['n_scenarios']}")
    print(f"saved {JSON_DIR}/method_summary.csv")
    pareto_df = df.copy()
    pareto_df['pareto_group'] = pareto_df['method'].map(pareto_group)
    pareto_df = pareto_df.dropna(subset=['pareto_group'])
    pareto_summary = pareto_df.groupby('pareto_group').agg(
        cum_reward_mean=('cum_reward', 'mean'),
        pax_wait=('pax_wait_mean', 'mean'),
    ).reindex(PARETO_ORDER).dropna()

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    cmap = plt.get_cmap('tab20')
    markers = ['o', 's', '^', 'D', 'P', 'X', 'v', '<', '>', 'h', '*']
    for idx, (m, row) in enumerate(pareto_summary.iterrows()):
        label = PARETO_GROUPS[m]
        ax.scatter(
            row['pax_wait'],
            row['cum_reward_mean'] / 1000,
            s=78,
            color=cmap(idx % cmap.N),
            marker=markers[idx % len(markers)],
            alpha=0.9,
            edgecolor='black',
            linewidth=0.55,
            label=label,
        )
    ax.set_xlabel('Mean per-leg passenger waiting time (s) — lower is better')
    ax.set_ylabel('Cumulative reward (K) — higher is better')
    ax.set_title('Reward vs operational utility (multi-scenario means)')
    ax.grid(True, alpha=0.3)
    ax.legend(
        title='Method',
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7.4,
        title_fontsize=8,
        frameon=False,
        borderaxespad=0.0,
        handletextpad=0.6,
        labelspacing=0.55,
    )
    fig.subplots_adjust(left=0.10, right=0.64, top=0.90, bottom=0.17)
    out_pdf = f'{FIG_DIR}/pareto.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_pdf}")

    # Per-scenario heatmap: rows = methods, cols = (seed, scale), values = cum_reward
    pivot_full = df.pivot_table(index='method', columns=['sumo_seed', 'od_scale'],
                                 values='cum_reward', aggfunc='mean') / 1000
    # Reorder rows for readability
    method_order = [m for m in CANON.keys() if m in pivot_full.index]
    pivot_full = pivot_full.reindex(method_order)
    fig, ax = plt.subplots(figsize=(8, max(3, 0.5*len(pivot_full)+1.5)))
    im = ax.imshow(pivot_full.values, aspect='auto', cmap='RdYlGn', vmin=-3000, vmax=-500)
    ax.set_xticks(range(pivot_full.shape[1]))
    ax.set_xticklabels([f"s{s}\nod{o:.1f}" for (s, o) in pivot_full.columns], fontsize=7)
    ax.set_yticks(range(len(pivot_full)))
    ax.set_yticklabels([CANON.get(m, m) for m in pivot_full.index], fontsize=8)
    for i in range(pivot_full.shape[0]):
        for j in range(pivot_full.shape[1]):
            v = pivot_full.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{int(v)}", ha='center', va='center', fontsize=6,
                        color='black' if -1500 < v < -800 else 'white')
    ax.set_title('Per-scenario cumulative reward (K)')
    fig.colorbar(im, ax=ax, label='Cum. reward (K)')
    fig.tight_layout()
    out_pdf = f'{FIG_DIR}/per_scenario_heatmap.pdf'
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    print(f"saved {out_pdf}")

if __name__ == '__main__':
    main()
