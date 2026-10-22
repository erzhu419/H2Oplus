"""
aggregate_multiseed.py
======================
Aggregate per-(method, seed, scale) JSONs produced by run_multiseed_eval.sh
into a single CSV plus a per-method summary table for paper §5.4.

Inputs:  experiment_output/multiseed_eval/{method}_sumo{S}_od{D}.json
Outputs:
  - experiment_output/multiseed_eval/multiseed_results.csv      (per-run rows)
  - experiment_output/multiseed_eval/multiseed_summary.csv      (mean ± SE per method)

Issue 4 / R-? (single-seed/single-scenario complaint).

CSV columns (per run):
    method, sumo_seed, od_scale,
    cum_reward, per_step_reward,
    passenger_wait_mean, passenger_wait_p90,
    headway_cv_avg, large_gap_rate, jain_fairness,
    completed_trips, n_decisions, wall_time_sec, json_path

Summary table (per method): mean ± SE across the (seed × scale) grid.
"""
import os
import sys
import json
import glob
import csv
import math
from collections import defaultdict


def _avg(values):
    if not values:
        return None
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _se(values):
    """Standard error of the mean, ignoring None."""
    vals = [v for v in values if v is not None]
    n = len(vals)
    if n < 2:
        return None
    mu = sum(vals) / n
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return math.sqrt(var / n)


def parse_one(json_path):
    """Return a flat dict with the columns we want for one JSON file."""
    with open(json_path) as f:
        d = json.load(f)

    fname = os.path.basename(json_path)
    # Filenames look like {method}_sumo{S}_od{D}.json
    base = fname[:-5] if fname.endswith('.json') else fname
    parts = base.rsplit('_sumo', 1)
    method = parts[0]
    seed_scale = parts[1] if len(parts) > 1 else ''
    sumo_seed = ''
    od_scale = ''
    if '_od' in seed_scale:
        s, sc = seed_scale.split('_od', 1)
        sumo_seed = s
        od_scale = sc

    # Prefer values inside the JSON for provenance correctness.
    sumo_seed_in = d.get('sumo_seed', sumo_seed)
    od_scale_in = d.get('od_scale', od_scale)

    # Headway-CV average across (line, dir).
    cv_dict = d.get('headway_cv_per_line', {}) or {}
    cv_vals = [v for v in cv_dict.values() if isinstance(v, (int, float)) and v > 0]
    headway_cv_avg = (sum(cv_vals) / len(cv_vals)) if cv_vals else None

    return {
        'method': method,
        'sumo_seed': sumo_seed_in,
        'od_scale': od_scale_in,
        'cum_reward': d.get('cumulative_reward'),
        'per_step_reward': d.get('per_step_reward'),
        'passenger_wait_mean': d.get('passenger_wait_mean'),
        'passenger_wait_p90': d.get('passenger_wait_p90'),
        'headway_cv_avg': headway_cv_avg,
        'large_gap_rate': d.get('large_gap_rate'),
        'jain_fairness': d.get('jain_fairness'),
        'completed_trips': d.get('completed_trips'),
        'n_decisions': d.get('n_decisions'),
        'wall_time_sec': d.get('wall_time_sec'),
        'json_path': json_path,
    }


def main(in_dir, out_csv, summary_csv):
    paths = sorted(glob.glob(os.path.join(in_dir, '*.json')))
    if not paths:
        print(f"[aggregate] No JSONs found under {in_dir}")
        return 1

    rows = []
    for p in paths:
        try:
            rows.append(parse_one(p))
        except Exception as e:
            print(f"[aggregate] WARN failed to parse {p}: {e}")

    cols = ['method', 'sumo_seed', 'od_scale',
            'cum_reward', 'per_step_reward',
            'passenger_wait_mean', 'passenger_wait_p90',
            'headway_cv_avg', 'large_gap_rate', 'jain_fairness',
            'completed_trips', 'n_decisions', 'wall_time_sec', 'json_path']

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] Wrote per-run CSV: {out_csv}  ({len(rows)} rows)")

    # ── Per-method summary (mean ± SE across seed × scale) ──
    metric_cols = ['cum_reward', 'per_step_reward',
                   'passenger_wait_mean', 'passenger_wait_p90',
                   'headway_cv_avg', 'large_gap_rate', 'jain_fairness',
                   'completed_trips']

    by_method = defaultdict(list)
    for r in rows:
        by_method[r['method']].append(r)

    summary_cols = ['method', 'n_runs']
    for m in metric_cols:
        summary_cols.append(f'{m}_mean')
        summary_cols.append(f'{m}_se')

    with open(summary_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        for method, runs in sorted(by_method.items()):
            row = {'method': method, 'n_runs': len(runs)}
            for m in metric_cols:
                vals = [r[m] for r in runs]
                row[f'{m}_mean'] = _avg(vals)
                row[f'{m}_se']   = _se(vals)
            w.writerow(row)
    print(f"[aggregate] Wrote per-method summary: {summary_csv}")

    # Pretty-print summary to stdout
    print()
    print("─" * 100)
    print(f"  {'method':22s} {'n':>3s} {'reward':>13s} {'pax_wait':>10s} {'hw_cv':>7s} {'large_gap':>10s} {'jain':>7s}")
    print("─" * 100)
    for method, runs in sorted(by_method.items()):
        rew_mean = _avg([r['cum_reward'] for r in runs])
        rew_se   = _se([r['cum_reward'] for r in runs])
        pw  = _avg([r['passenger_wait_mean'] for r in runs])
        cv  = _avg([r['headway_cv_avg'] for r in runs])
        lg  = _avg([r['large_gap_rate'] for r in runs])
        jf  = _avg([r['jain_fairness'] for r in runs])
        rew_str = f"{rew_mean:>9,.0f}±{rew_se:,.0f}" if rew_mean is not None and rew_se is not None else (f"{rew_mean:>9,.0f}" if rew_mean is not None else 'n/a')
        print(f"  {method:22s} {len(runs):>3d} {rew_str:>13s} "
              f"{(pw or 0):>10.1f} {(cv or 0):>7.3f} {(lg or 0):>10.3f} {(jf or 0):>7.3f}")
    print("─" * 100)
    return 0


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    h2o_root = os.path.dirname(here)
    in_dir = os.path.join(h2o_root, 'experiment_output', 'multiseed_eval')
    out_csv = os.path.join(in_dir, 'multiseed_results.csv')
    summary_csv = os.path.join(in_dir, 'multiseed_summary.csv')
    if len(sys.argv) > 1:
        in_dir = sys.argv[1]
        out_csv = os.path.join(in_dir, 'multiseed_results.csv')
        summary_csv = os.path.join(in_dir, 'multiseed_summary.csv')
    sys.exit(main(in_dir, out_csv, summary_csv))
