from __future__ import annotations
"""
compare_envs.py
===============
Zero-control (action=0 holding) comparison:
  - MultiLineSimEnv (sim_core)
  - SumoRLBridge (SUMO)

Metrics collected per episode, per bus, per stop:
  - Segment travel time (s)  = depart_prev_stop → arrive_this_stop
  - Dwell time (s)           = arrive_this_stop → depart_this_stop
  - Boardings                = board_num_d[stop] / board_num
  - Alightings               = alight_num_d[stop] / alight_num
  - Load on departure        = passenger_num_n / len(passengers)
  - Full-load events         = load >= capacity

Global per-episode:
  - Total passengers served  = sum of all alightings (proxy for service)
  - Episode wall time (s)
  - Speed histogram          (low: <5 m/s, med: 5-12, high: >12)

Usage:
    cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o
    python compare_envs.py --episodes 20 [--skip_sumo] [--sumo_root <path>]

Run sim-only smoke:
    python compare_envs.py --episodes 3 --skip_sumo
"""

import os, sys, time, csv, argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
SUMO_DIR    = os.path.join(REPO_ROOT, 'SUMO_ruiguang', 'online_control')
OUT_DIR     = os.path.join(SCRIPT_DIR, 'compare_results')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── CLI ─────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument('--episodes',   type=int, default=20)
p.add_argument('--skip_sumo',  action='store_true')
p.add_argument('--sumo_root',  type=str, default=SUMO_DIR)
p.add_argument('--max_steps',  type=int, default=23000,
               help='SUMO max_steps (default 18000+5000=23000)')
p.add_argument('--env_config', type=str, default='calibrated_env')
args = p.parse_args()

# ─── Output CSVs ─────────────────────────────────────────────────────────────
SIM_STOP_CSV   = os.path.join(OUT_DIR, 'sim_stop_metrics.csv')
SUMO_STOP_CSV  = os.path.join(OUT_DIR, 'sumo_stop_metrics.csv')
SIM_EP_CSV     = os.path.join(OUT_DIR, 'sim_episode_metrics.csv')
SUMO_EP_CSV    = os.path.join(OUT_DIR, 'sumo_episode_metrics.csv')
ANALYSIS_MD    = os.path.join(OUT_DIR, 'analysis_report.md')

STOP_HEADER = ['episode', 'line_id', 'bus_id', 'stop_id', 'stop_idx',
               'arrive_time', 'depart_time', 'dwell_time',
               'seg_travel_time',   # depart(prev) → arrive(this)
               'board', 'alight', 'strand',
               'load_on_departure', 'full_load']

EP_HEADER   = ['episode', 'env', 'total_boarded', 'total_alighted',
               'full_load_events', 'wall_sec',
               'speed_low_frac', 'speed_med_frac', 'speed_high_frac',
               'n_trip_records']


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: collect sim-env metrics
# ═══════════════════════════════════════════════════════════════════════════════
def run_sim_episode(env, ep_idx: int) -> tuple[list[dict], dict]:
    """Run one episode with zero holding. Return (stop_rows, ep_summary)."""
    env.reset()
    env.initialize_state()

    # Zero-action dict
    action_dict = {lid: {k: 0.0 for k in range(le.max_agent_num)}
                   for lid, le in env.line_map.items()}
    done = False
    wall_t0 = time.time()

    # Speed samples (m/s) — sampled every step from all alive buses
    speed_samples: list[float] = []

    while not done:
        state, reward, done = env.step(action_dict)
        # Sample speeds
        for lid, le in env.line_map.items():
            for bus in le.bus_all:
                if bus.on_route:
                    speed_samples.append(float(bus.current_speed))

    wall_sec = time.time() - wall_t0

    # ── Harvest stop records from all sub-envs ─────────────────────────────
    stop_rows: list[dict] = []
    total_boarded = total_alighted = full_load_events = 0

    for lid, le in env.line_map.items():
        cap = 50  # Bus.capacity
        for bus in le.bus_all:
            prev_depart: float | None = None
            prev_stop: str | None = None
            # stop_records: [station_name, arrive_time, depart_time]
            for rec_idx, rec in enumerate(bus.stop_records):
                stop_name, arr, dep = rec[0], rec[1], rec[2]
                dwell = dep - arr
                seg_travel = (arr - prev_depart) if (prev_depart is not None) else float('nan')

                # board/alight from per-station sums (sim tracks cumulative)
                board  = float(bus.board_num)  if rec_idx == len(bus.stop_records)-1 else 0.0
                alight = float(bus.alight_num) if rec_idx == len(bus.stop_records)-1 else 0.0

                # Use trajectory_dict for per-stop arrive time to infer prev board
                # (Sim doesn't store per-stop board/alight separately;
                #  best proxy: change in passenger count)
                load_dep = len(bus.passengers)
                full = 1 if load_dep >= cap else 0
                full_load_events += full

                stop_rows.append(dict(
                    episode=ep_idx, line_id=lid,
                    bus_id=bus.bus_id, stop_id=stop_name,
                    stop_idx=rec_idx,
                    arrive_time=arr, depart_time=dep,
                    dwell_time=dwell, seg_travel_time=seg_travel,
                    board=0, alight=0, strand=0,  # sim doesn't log per-stop
                    load_on_departure=load_dep, full_load=full,
                ))
                prev_depart = dep
                prev_stop   = stop_name

    # Speed distribution
    arr = np.array(speed_samples, dtype=float)
    lo = float(np.mean(arr < 5.0))   if len(arr) else 0.0
    md = float(np.mean((arr >= 5.0) & (arr <= 12.0))) if len(arr) else 0.0
    hi = float(np.mean(arr > 12.0))  if len(arr) else 0.0

    ep_summary = dict(
        episode=ep_idx, env='sim',
        total_boarded=total_boarded, total_alighted=total_alighted,
        full_load_events=full_load_events, wall_sec=wall_sec,
        speed_low_frac=lo, speed_med_frac=md, speed_high_frac=hi,
        n_trip_records=len(stop_rows),
    )
    return stop_rows, ep_summary


# Better sim stop metrics: harvest after episode using bus.board_num_d / alight_num_d etc.
def run_sim_episode_v2(env, ep_idx: int) -> tuple[list[dict], dict]:
    """Run one sim episode with zero holding. Return (stop_rows, ep_summary).

    Uses bus.board_num_d / alight_num_d / strand_num_d / load_dep_d (added to
    sim_core/bus.py exchange_passengers) for per-stop passenger counts.
    Uses bus.stop_records for per-stop timing.
    """
    env.reset()
    env.initialize_state()

    # Disable plotting in each sub-env
    for le in env.line_map.values():
        le.enable_plot = False

    action_dict = {lid: {k: 0.0 for k in range(le.max_agent_num)}
                   for lid, le in env.line_map.items()}
    done = False
    wall_t0 = time.time()
    speed_samples: list[float] = []

    while not done:
        for lid, le in env.line_map.items():
            for bus in le.bus_all:
                if bus.on_route:
                    speed_samples.append(float(bus.current_speed))
        _, _, done = env.step(action_dict)

    wall_sec = time.time() - wall_t0

    # ── Post-episode harvest: iterate all buses in all sub-envs ───────────
    stop_rows: list[dict] = []
    total_boarded = total_alighted = full_load_events = 0
    row_idx = 0

    for lid, le in env.line_map.items():
        cap = 50  # Bus.capacity
        for bus in le.bus_all:
            # stop_records: list of [station_name, arrive_time, depart_time]
            prev_depart: float | None = None
            for s_idx, rec in enumerate(bus.stop_records):
                stop_name, arr, dep = rec[0], rec[1], rec[2]
                dwell  = dep - arr
                seg_tt = (arr - prev_depart) if (prev_depart is not None) else float('nan')
                prev_depart = dep

                board   = bus.stop_board_l[s_idx]  if s_idx < len(bus.stop_board_l)  else 0
                alight  = bus.stop_alight_l[s_idx] if s_idx < len(bus.stop_alight_l) else 0
                strand  = bus.stop_strand_l[s_idx] if s_idx < len(bus.stop_strand_l) else 0
                load_d  = bus.stop_load_l[s_idx]   if s_idx < len(bus.stop_load_l)   else 0
                full    = 1 if load_d >= cap else 0

                total_boarded   += board
                total_alighted  += alight
                full_load_events += full

                stop_rows.append(dict(
                    episode=ep_idx, line_id=lid,
                    bus_id=bus.bus_id, stop_id=stop_name,
                    stop_idx=s_idx,
                    arrive_time=arr, depart_time=dep,
                    dwell_time=dwell, seg_travel_time=seg_tt,
                    board=board, alight=alight, strand=strand,
                    load_on_departure=load_d, full_load=full,
                ))
                row_idx += 1

    # Speed distribution.
    # Sim speed is in m/s (per-step current_speed, which is speed_limit m/s for buses on Travel)
    arr_sp = np.array(speed_samples, dtype=float)
    lo = float(np.mean(arr_sp < 5.0))                        if len(arr_sp) else 0.0
    md = float(np.mean((arr_sp >= 5.0) & (arr_sp <= 12.0))) if len(arr_sp) else 0.0
    hi = float(np.mean(arr_sp > 12.0))                       if len(arr_sp) else 0.0

    ep_summary = dict(
        episode=ep_idx, env='sim',
        total_boarded=total_boarded, total_alighted=total_alighted,
        full_load_events=full_load_events, wall_sec=wall_sec,
        speed_low_frac=lo, speed_med_frac=md, speed_high_frac=hi,
        n_trip_records=len(stop_rows),
    )
    return stop_rows, ep_summary


# ═══════════════════════════════════════════════════════════════════════════════
#  Helper: collect SUMO metrics
# ═══════════════════════════════════════════════════════════════════════════════

def run_sumo_episode(sumo_env, bridge, ep_idx: int):
    """Run one SUMO episode with zero holding via SumoBusHoldingEnv.

    Uses build_bridge() callbacks (decision_provider / action_executor) via
    SumoBusHoldingEnv.step(), exactly like the legacy SAC training loop.
    Zero-hold policy: apply 0.0 seconds of extra holding at every decision.

    IMPORTANT: must pass hold=0.0 for EACH pending bus explicitly.
    An empty dict causes _apply_actions() to skip all buses, leaving them
    stuck at the pre-emptive 3600 s hold that rl_bridge injects on arrival,
    which falsely inflates seg_travel_time by hundreds of seconds.
    """
    sumo_env.reset()
    wall_t0 = time.time()
    done = False
    n_decisions = 0

    while not done:
        # Build zero-hold action for every pending bus decision.
        # sumo_env._pending_events keys = (line_id, bus_id)
        if hasattr(sumo_env, '_pending_events') and sumo_env._pending_events:
            zero_action = {}
            for (line_id, bus_id) in list(sumo_env._pending_events.keys()):
                zero_action.setdefault(line_id, {})[bus_id] = 0.0
        else:
            zero_action = {}

        state, rewards, done, _ = sumo_env.step(zero_action)
        n_decisions += 1
        if n_decisions > 50000:  # safety guard (~10x normal episode length)
            print(f"  WARNING: ep {ep_idx} exceeded 50K decisions, forcing done")
            break

    wall_sec = time.time() - wall_t0

    # ── Harvest per-stop records from bridge.bus_obj_dic ─────────────────
    stop_rows: list[dict] = []
    total_boarded = total_alighted = full_load_events = 0
    BusCap = bridge.BusCap

    for bus_id, bus_obj in bridge.bus_obj_dic.items():
        lid = bus_obj.belong_line_id_s
        line = bridge.line_obj_dic.get(lid)
        stops = line.stop_id_l if line else []

        prev_depart: float | None = None
        for s_idx, stop_id in enumerate(stops):
            arr = bus_obj.arriver_stop_time_d.get(stop_id)
            dep = bus_obj.depart_stop_time_d.get(stop_id)
            if arr is None:
                continue  # bus didn't reach this stop
            dep = dep if dep is not None else arr

            dwell  = dep - arr
            seg_tt = (arr - prev_depart) if prev_depart is not None else float('nan')
            prev_depart = dep

            board  = bus_obj.board_num_d.get(stop_id, 0)
            alight = bus_obj.alight_num_d.get(stop_id, 0)
            strand = bus_obj.strand_num_d.get(stop_id, 0)
            load_dep = int(bus_obj.passenger_num_n)
            full = 1 if board + max(0, load_dep - alight) >= BusCap else 0

            total_boarded  += board
            total_alighted += alight
            full_load_events += full

            stop_rows.append(dict(
                episode=ep_idx, line_id=lid,
                bus_id=bus_id, stop_id=stop_id, stop_idx=s_idx,
                arrive_time=arr, depart_time=dep,
                dwell_time=dwell, seg_travel_time=seg_tt,
                board=board, alight=alight, strand=strand,
                load_on_departure=load_dep, full_load=full,
            ))

    # Speed fractions: SUMO bus speeds are in m/s from bus_speed_n
    # Approximate from dwell/seg_tt distribution since we don't sample per tick
    # (speed sampling would require hooking into _advance_one_step which is internal)
    lo = md = hi = 0.0  # cannot sample speed without raw step access

    ep_summary = dict(
        episode=ep_idx, env='sumo',
        total_boarded=total_boarded, total_alighted=total_alighted,
        full_load_events=full_load_events, wall_sec=wall_sec,
        speed_low_frac=lo, speed_med_frac=md, speed_high_frac=hi,
        n_trip_records=len(stop_rows),
    )
    return stop_rows, ep_summary


# ═══════════════════════════════════════════════════════════════════════════════
#  Analysis & report
# ═══════════════════════════════════════════════════════════════════════════════

def write_csv(path: str, header: list, rows: list[dict]) -> None:
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def compute_stop_agg(rows: list[dict]) -> dict:
    """Compute per-stop-id mean travel time, dwell, board, alight."""
    from collections import defaultdict
    agg = defaultdict(list)
    for r in rows:
        sid = r['stop_id']
        agg[f"{r['line_id']}|{sid}|travel"].append(r['seg_travel_time'])
        agg[f"{r['line_id']}|{sid}|dwell" ].append(r['dwell_time'])
        agg[f"{r['line_id']}|{sid}|board" ].append(r['board'])
        agg[f"{r['line_id']}|{sid}|alight"].append(r['alight'])
        agg[f"{r['line_id']}|{sid}|load"  ].append(r['load_on_departure'])
        agg[f"{r['line_id']}|{sid}|full"  ].append(r['full_load'])
    return {k: [x for x in v if not (isinstance(x, float) and np.isnan(x))]
            for k, v in agg.items()}


def generate_analysis(sim_rows, sim_eps, sumo_rows, sumo_eps) -> str:
    """Generate markdown analysis report."""
    lines = ["# Sim vs SUMO Zero-Control Comparison\n"]
    n_ep_sim  = len(sim_eps)
    n_ep_sumo = len(sumo_eps)

    def ep_mean(eps, key):
        vals = [e[key] for e in eps if key in e]
        return np.mean(vals) if vals else float('nan')

    def bold(x): return f"**{x}**"

    lines.append("## Episode-Level Summary\n")
    lines.append("| Metric | Sim | SUMO |")
    lines.append("|--------|-----|------|")
    metrics = [
        ('Episodes', n_ep_sim, n_ep_sumo),
        ('Avg total_boarded',  round(ep_mean(sim_eps,'total_boarded'),1),  round(ep_mean(sumo_eps,'total_boarded'),1)),
        ('Avg total_alighted', round(ep_mean(sim_eps,'total_alighted'),1), round(ep_mean(sumo_eps,'total_alighted'),1)),
        ('Avg full_load_events',round(ep_mean(sim_eps,'full_load_events'),1),round(ep_mean(sumo_eps,'full_load_events'),1)),
        ('Avg wall_sec',        round(ep_mean(sim_eps,'wall_sec'),1),       round(ep_mean(sumo_eps,'wall_sec'),1)),
        ('Avg speed_low_frac',  round(ep_mean(sim_eps,'speed_low_frac'),3), round(ep_mean(sumo_eps,'speed_low_frac'),3)),
        ('Avg speed_med_frac',  round(ep_mean(sim_eps,'speed_med_frac'),3), round(ep_mean(sumo_eps,'speed_med_frac'),3)),
        ('Avg speed_high_frac', round(ep_mean(sim_eps,'speed_high_frac'),3),round(ep_mean(sumo_eps,'speed_high_frac'),3)),
        ('Avg trip_records',    round(ep_mean(sim_eps,'n_trip_records'),0), round(ep_mean(sumo_eps,'n_trip_records'),0)),
    ]
    for label, sv, sov in metrics:
        lines.append(f"| {label} | {sv} | {sov} |")

    lines.append("\n## Per-Stop Travel Time (segment avg, top-20 by deviation)\n")

    if sim_rows and sumo_rows:
        sim_agg  = compute_stop_agg(sim_rows)
        sumo_agg = compute_stop_agg(sumo_rows)

        devs = []
        all_keys = set(k.split('|',2)[0] + '|' + k.split('|',2)[1]
                       for k in sim_agg if k.endswith('|travel'))
        for lk in sorted(all_keys):
            sk_t  = f"{lk}|travel"
            sv_t  = sim_agg.get(sk_t,  [])
            sov_t = sumo_agg.get(sk_t, [])
            if not sv_t or not sov_t:
                continue
            sm, som = np.nanmean(sv_t), np.nanmean(sov_t)
            dev = abs(sm - som)
            devs.append((lk, sm, som, dev))

        devs.sort(key=lambda x: -x[3])
        lines.append("| line|stop | Sim travel_t (s) | SUMO travel_t (s) | |diff| |")
        lines.append("|-----|------|------|------|")
        for lk, sm, som, dev in devs[:20]:
            flag = " ⚠️" if dev > 60 else ""
            lines.append(f"| {lk} | {sm:.1f} | {som:.1f} | {dev:.1f}{flag} |")

    lines.append("\n## Potential Bugs\n")
    bugs = []

    # Check 1: boarded vs alighted global balance
    tb_sim  = ep_mean(sim_eps, 'total_boarded')
    ta_sim  = ep_mean(sim_eps, 'total_alighted')
    tb_sumo = ep_mean(sumo_eps, 'total_boarded')
    ta_sumo = ep_mean(sumo_eps, 'total_alighted')

    if not np.isnan(tb_sim) and not np.isnan(ta_sim):
        if tb_sim > 0 and abs(tb_sim - ta_sim) / (tb_sim + 1) > 0.2:
            bugs.append(f"⚠️ **Sim**: board ({tb_sim:.0f}) vs alight ({ta_sim:.0f}) diverge >20% → potential passenger leak")

    if not np.isnan(tb_sumo) and not np.isnan(ta_sumo):
        if tb_sumo > 0 and abs(tb_sumo - ta_sumo) / (tb_sumo + 1) > 0.2:
            bugs.append(f"⚠️ **SUMO**: board ({tb_sumo:.0f}) vs alight ({ta_sumo:.0f}) diverge >20% → potential passenger leak")

    # Check 2: travel time plausibility (should be 60–600s for typical 500m segments at 5-12 m/s)
    if sim_rows:
        sim_tt = [r['seg_travel_time'] for r in sim_rows
                  if not np.isnan(r.get('seg_travel_time', float('nan'))) and r['seg_travel_time'] > 0]
        if sim_tt:
            p5, p95 = np.percentile(sim_tt, [5, 95])
            if p5 < 20 or p95 > 1200:
                bugs.append(f"⚠️ **Sim**: segment travel time p5={p5:.0f}s p95={p95:.0f}s (expected 20-600s)")

    if sumo_rows:
        sumo_tt = [r['seg_travel_time'] for r in sumo_rows
                   if not np.isnan(r.get('seg_travel_time', float('nan'))) and r['seg_travel_time'] > 0]
        if sumo_tt:
            p5, p95 = np.percentile(sumo_tt, [5, 95])
            if p5 < 20 or p95 > 1200:
                bugs.append(f"⚠️ **SUMO**: segment travel time p5={p5:.0f}s p95={p95:.0f}s (expected 20-600s)")

    # Check 3: full-load rate
    fl_sim  = ep_mean(sim_eps, 'full_load_events')
    fl_sumo = ep_mean(sumo_eps, 'full_load_events')
    if not np.isnan(fl_sim) and not np.isnan(fl_sumo):
        if fl_sim == 0 and fl_sumo > 5:
            bugs.append("⚠️ Sim has 0 full-load events but SUMO has many → Sim passenger demand may be too low")
        elif fl_sumo == 0 and fl_sim > 5:
            bugs.append("⚠️ SUMO has 0 full-load events but Sim has many → SUMO capacity or demand mismatch")

    # Check 4: speed distribution
    sl_sim  = ep_mean(sim_eps, 'speed_low_frac')
    sh_sim  = ep_mean(sim_eps, 'speed_high_frac')
    sl_sumo = ep_mean(sumo_eps, 'speed_low_frac')
    sh_sumo = ep_mean(sumo_eps, 'speed_high_frac')
    if not np.isnan(sl_sim) and not np.isnan(sl_sumo):
        if abs(sl_sim - sl_sumo) > 0.3:
            bugs.append(f"⚠️ Low-speed fraction: Sim={sl_sim:.2f} SUMO={sl_sumo:.2f} (>0.3 diff)")
        if abs(sh_sim - sh_sumo) > 0.3:
            bugs.append(f"⚠️ High-speed fraction: Sim={sh_sim:.2f} SUMO={sh_sumo:.2f} (>0.3 diff)")

    if bugs:
        for b in bugs:
            lines.append(f"- {b}")
    else:
        lines.append("No major anomalies detected ✅")

    lines.append("\n---\n*Generated by compare_envs.py*\n")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    N = args.episodes
    print(f"[compare_envs] episodes={N}  skip_sumo={args.skip_sumo}")

    all_sim_stop: list[dict]  = []
    all_sim_ep:   list[dict]  = []
    all_sumo_stop: list[dict] = []
    all_sumo_ep:   list[dict] = []

    # ── Sim Episodes ──────────────────────────────────────────────────────
    print("\n=== SIM EPISODES ===")
    sys.path.insert(0, SCRIPT_DIR)
    from envs.bus_sim_env import MultiLineSimEnv
    sim_env = MultiLineSimEnv(args.env_config)
    sim_env.enable_plot = False  # no plots

    for ep in range(N):
        t0 = time.time()
        try:
            stop_rows, ep_sum = run_sim_episode_v2(sim_env, ep + 1)
            all_sim_stop.extend(stop_rows)
            all_sim_ep.append(ep_sum)
            print(f"  Ep {ep+1:3d}: stops={ep_sum['n_trip_records']:5d}  "
                  f"board={ep_sum['total_boarded']:6.0f}  "
                  f"wall={ep_sum['wall_sec']:.1f}s")
        except Exception as exc:
            print(f"  Ep {ep+1}: ERROR — {exc}")
            import traceback; traceback.print_exc()

    write_csv(SIM_STOP_CSV, STOP_HEADER, all_sim_stop)
    write_csv(SIM_EP_CSV,   EP_HEADER,   all_sim_ep)
    print(f"Sim CSVs → {SIM_STOP_CSV}, {SIM_EP_CSV}")

    # ── SUMO Episodes ─────────────────────────────────────────────────────
    if not args.skip_sumo:
        print("\n=== SUMO EPISODES ===")
        # sim_obj/ (passenger etc) lives under SUMO_ruiguang/online_control
        sumo_root = os.path.abspath(args.sumo_root)
        if sumo_root not in sys.path:
            sys.path.insert(0, sumo_root)
        from sumo_env.rl_bridge import build_bridge
        from sumo_env.rl_env import SumoBusHoldingEnv

        bridge_cbs = build_bridge(root_dir=sumo_root, gui=False)
        _bridge_obj = bridge_cbs['_bridge'] if '_bridge' in bridge_cbs else None

        # Reconstruct internal bridge reference for post-ep metric harvest.
        # build_bridge() closes over a SumoRLBridge instance — grab it via the
        # decision_provider closure's __closure__ cell.
        from sumo_env.rl_bridge import SumoRLBridge as _BridgeCls
        _dp = bridge_cbs['decision_provider']
        for cell in getattr(_dp, '__closure__', None) or []:
            try:
                obj = cell.cell_contents
                if isinstance(obj, _BridgeCls):
                    _bridge_obj = obj
                    break
            except ValueError:
                pass

        sumo_env_obj = SumoBusHoldingEnv(
            root_dir=sumo_root,
            decision_provider=bridge_cbs['decision_provider'],
            action_executor=bridge_cbs['action_executor'],
            reset_callback=bridge_cbs['reset_callback'],
            close_callback=bridge_cbs['close_callback'],
            reward_type='linear_penalty',
        )
        try:
            for ep in range(N):
                try:
                    stop_rows, ep_sum = run_sumo_episode(sumo_env_obj, _bridge_obj, ep + 1)
                    all_sumo_stop.extend(stop_rows)
                    all_sumo_ep.append(ep_sum)
                    print(f"  Ep {ep+1:3d}: stops={ep_sum['n_trip_records']:5d}  "
                          f"board={ep_sum['total_boarded']:6.0f}  "
                          f"wall={ep_sum['wall_sec']:.1f}s")
                except Exception as exc:
                    print(f"  Ep {ep+1}: ERROR — {exc}")
                    import traceback; traceback.print_exc()
        finally:
            try:
                sumo_env_obj.close()
            except Exception:
                pass

        write_csv(SUMO_STOP_CSV, STOP_HEADER, all_sumo_stop)
        write_csv(SUMO_EP_CSV,   EP_HEADER,   all_sumo_ep)
        print(f"SUMO CSVs → {SUMO_STOP_CSV}, {SUMO_EP_CSV}")
    else:
        print("\n[SUMO skipped]")

    # ── Analysis Report ─────────────────────────────────────────────────
    print("\nGenerating analysis report...")
    report = generate_analysis(all_sim_stop, all_sim_ep, all_sumo_stop, all_sumo_ep)
    with open(ANALYSIS_MD, 'w') as f:
        f.write(report)
    print(f"Report → {ANALYSIS_MD}")
    print("\n--- SUMMARY ---")
    print(report[:3000])


if __name__ == '__main__':
    # Also disable sim plotting globally
    for attr_name in ('enable_plot',):
        pass  # handled in main
    main()
