"""
calibrate_speeds.py
====================
Calibrate sim route_news.xlsx V_max values using SUMO-observed seg_travel_times.

Logic:
  • seg_travel_time = depart(prev_stop) → arrive(this_stop) = pure driving time
  • For each (line_id, stop_idx), compute median driving time across all buses
  • Calibrated V_max = segment_distance / median_travel_time
  • Clip to [V_MIN, V_MAX] to avoid extreme values from sparse SUMO data
  • For segments with no SUMO data (bus never reached them), keep existing V_max

Usage:
    cd H2Oplus/bus_h2o
    python calibrate_speeds.py [--dry-run]
"""

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────
SUMO_CSV   = "compare_results/sumo_stop_metrics.csv"
CAL_DIR    = "calibrated_env"
LINE_ENVS  = os.path.join(CAL_DIR, "_line_envs")
V_MIN      = 1.5   # m/s  (5.4 km/h) — floor for very congested segments
V_MAX      = 15.0  # m/s  (54 km/h) — ceiling
BACKUP_SFX = ".bak" # backup suffix for original files


def load_sumo_seg_times(csv_path: str):
    """Return {line_id: {stop_idx: [seg_travel_time, ...]}}.

    Only includes rows where seg_travel_time > 0 (first stop has no travel).
    """
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            st = row.get("seg_travel_time", "")
            if not st or st in ("nan", ""):
                continue
            t = float(st)
            if t <= 0:
                continue
            lid  = row["line_id"]
            sidx = int(float(row["stop_idx"]))
            data[lid][sidx].append(t)
    return data


def calibrate_line(line_id, route_df, sumo_times, dry_run=False):
    """Return updated route_df with calibrated V_max per segment."""
    new_vmaxs = []
    for seg_idx, row in route_df.iterrows():
        dist = float(row["distance"])
        old_vmax = float(row["V_max"])

        # SUMO stop_idx = seg_idx + 1  (stop 0 is origin — no travel time)
        sumo_idx = seg_idx + 1
        times = sumo_times.get(sumo_idx, [])

        if len(times) >= 2:
            # Use median for robustness against outliers
            med_time = float(np.median(times))
            cal_v = dist / med_time
            cal_v = float(np.clip(cal_v, V_MIN, V_MAX))
            flag = f"N={len(times)} median={med_time:.0f}s → {cal_v:.2f}m/s"
        elif len(times) == 1:
            # Single observation — clamp more conservatively
            med_time = times[0]
            cal_v = dist / med_time
            cal_v = float(np.clip(cal_v, V_MIN, V_MAX))
            flag = f"N=1 → {cal_v:.2f}m/s (single obs)"
        else:
            cal_v = old_vmax
            flag = f"N=0 → keep {old_vmax:.2f}m/s"

        print(f"    seg {seg_idx:2d} ({row['start_stop']}→{row['end_stop']}) "
              f"dist={dist:.0f}m  old={old_vmax:.1f}  {flag}")
        new_vmaxs.append(cal_v)

    updated = route_df.copy()
    updated["V_max"] = new_vmaxs
    return updated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show calibration results without writing files")
    args = parser.parse_args()

    if not os.path.exists(SUMO_CSV):
        print(f"ERROR: {SUMO_CSV} not found. Run compare_envs.py first.")
        sys.exit(1)

    print(f"Loading SUMO stop metrics from {SUMO_CSV} …")
    sumo_times = load_sumo_seg_times(SUMO_CSV)
    lines_in_sumo = sorted(sumo_times.keys())
    print(f"  Lines with SUMO data: {lines_in_sumo}\n")

    summary_rows = []

    for line_id in sorted(os.listdir(LINE_ENVS)):
        line_path = os.path.join(LINE_ENVS, line_id)
        route_xlsx = os.path.join(line_path, "data", "route_news.xlsx")
        if not os.path.exists(route_xlsx):
            continue

        print(f"=== {line_id} {'(no SUMO data)' if line_id not in sumo_times else ''} ===")
        route_df = pd.read_excel(route_xlsx)

        if line_id not in sumo_times:
            print("  Skipping — no SUMO observations for this line.\n")
            continue

        updated_df = calibrate_line(
            line_id, route_df, sumo_times[line_id], dry_run=args.dry_run)

        # Summary stats
        old_travel = (route_df["distance"] / route_df["V_max"]).sum()
        new_travel = (updated_df["distance"] / updated_df["V_max"]).sum()
        print(f"  Route total sim travel time: {old_travel:.0f}s → {new_travel:.0f}s "
              f"({new_travel/old_travel:.2f}x)\n")
        summary_rows.append((line_id, old_travel, new_travel))

        if not args.dry_run:
            # Backup original
            bak = route_xlsx + BACKUP_SFX
            if not os.path.exists(bak):
                shutil.copy2(route_xlsx, bak)
                print(f"  Backed up → {bak}")
            # Write calibrated
            updated_df.to_excel(route_xlsx, index=False)
            print(f"  Written  → {route_xlsx}")

            # ── Invalidate env_bus._DATA_CACHE for this line ──────────────
            # The calibrated V_max is in route_news.xlsx which is loaded at
            # env_bus.__init__ time.  The class-level cache stores the df read
            # at first import — we need to clear it so next run reads the new file.
            try:
                sys.path.insert(0, os.path.abspath("."))
                from sim_core.sim import env_bus
                data_dir = os.path.abspath(os.path.join(line_path, "data"))
                if data_dir in env_bus._DATA_CACHE:
                    del env_bus._DATA_CACHE[data_dir]
                    print(f"  Cleared  _DATA_CACHE for {data_dir}")
            except Exception:
                pass

    print("\n=== Summary ===")
    print(f"{'Line':8s}  {'Old sim travel(s)':>18s}  {'New sim travel(s)':>18s}  {'Ratio':>6s}")
    for lid, old_t, new_t in summary_rows:
        print(f"{lid:8s}  {old_t:>18.0f}  {new_t:>18.0f}  {new_t/old_t:>6.2f}x")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified.")
    else:
        print("\nDone. Restart training / re-import MultiLineSimEnv to pick up new speeds.")


if __name__ == "__main__":
    main()
