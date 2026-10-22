"""
eval_daganzo.py
===============
Evaluate the Daganzo cooperative-holding analytical baseline on the SUMO
RL bridge using the same harness as `eval_with_metrics.py`.

Usage:
    SUMO_HOME=/usr/share/sumo LIBSUMO_AS_TRACI=1 \
    conda run -n LSTM-RL python eval_daganzo.py \
        --alpha 0.6 \
        --output ../experiment_output/daganzo_smoketest.json
"""

import os, sys, time, json, argparse
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)

from common.data_utils import build_edge_linear_map, set_route_length
from daganzo_policy import DaganzoPolicy

SUMO_DIR = os.path.normpath(os.path.join(
    _BUS_H2O, os.pardir, os.pardir, "SUMO_ruiguang", "online_control"))
sys.path.insert(0, SUMO_DIR)
sys.path.insert(0, os.path.join(SUMO_DIR, "sim_obj"))
EDGE_XML = os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml")
SCHEDULE_XML = os.path.join(SUMO_DIR, "initialize_obj", "save_obj_bus.add.xml")

import xml.etree.ElementTree as ET


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--alpha', type=float, default=0.6,
                   help='Daganzo cooperation strength in [0,1] (default 0.6).')
    p.add_argument('--output', type=str, default='')
    p.add_argument('--max_steps', type=int, default=18000)
    p.add_argument('--bunching_threshold', type=float, default=180.0)
    p.add_argument('--single_sided', action='store_true',
                   help='Use only h_prev (textbook variant when h_next is unobserved).')
    p.add_argument('--sumo_seed', type=int, default=42,
                   help='SUMO RNG seed (--seed); also recorded for provenance.')
    p.add_argument('--od_scale', type=float, default=1.0,
                   help='Multiplicative factor on passenger demand intensity '
                        '(propagated as SUMO --scale).')
    p.add_argument('--large_gap_factor', type=float, default=1.5,
                   help='headway >= factor*scheduled is a large gap (default 1.5x).')
    p.add_argument('--method_tag', type=str, default='daganzo',
                   help='Label written into the JSON for provenance.')
    return p.parse_args()


def _patch_bridge_for_seed_and_scale(bridge_cls, seed: int, scale: float):
    """Same patch as eval_with_metrics.py — appends --seed and uses bridge.scale
    (which the constructor already wires to --scale)."""
    orig_start = bridge_cls._start_traci

    def _start_traci_with_seed(self):
        from sumolib import checkBinary
        import traci as _traci
        try:
            import libsumo as _libsumo
            _libsumo_ok = True
        except ImportError:
            _libsumo_ok = False
        common = ["-c", self.sumo_cfg, "--no-warnings",
                  "--duration-log.disable", "--log", "/dev/null",
                  "--scale", str(self.scale),
                  "--seed", str(seed)]
        if self.gui:
            binary = checkBinary('sumo-gui')
            _traci.start([binary] + common)
            self.sumo_binary = binary
        elif _libsumo_ok:
            sys.modules['traci'] = _libsumo
            import sumo_env.rl_bridge as _rb
            _rb.traci = _libsumo
            _libsumo.start(["sumo"] + common)
            self.sumo_binary = "libsumo"
        else:
            binary = checkBinary('sumo')
            _traci.start([binary] + common)
            self.sumo_binary = binary

    bridge_cls._start_traci = _start_traci_with_seed
    return orig_start


def build_sumo_indices(schedule_xml):
    from collections import defaultdict
    tree = ET.parse(schedule_xml); root = tree.getroot()
    line_deps = defaultdict(list)
    for elem in root.findall(".//bus_obj"):
        lid = elem.get("belong_line_id_s"); bid = elem.get("bus_id_s")
        st = float(elem.get("start_time_n", "0"))
        if lid and bid:
            line_deps[lid].append((st, bid))
    for entries in line_deps.values():
        entries.sort(key=lambda p: p[0])
    line_idx = {lid: i for i, lid in enumerate(sorted(line_deps.keys()))}
    bus_idx, counter = {}, 0
    for lid, deps in line_deps.items():
        for _, bid in deps:
            if bid not in bus_idx:
                bus_idx[bid] = counter; counter += 1
    return line_idx, bus_idx


def event_to_obs(ev, line_idx_map, bus_idx_map, line_headway, station_index, time_period_index):
    line_idx = line_idx_map.get(ev.line_id, 0)
    bus_idx = bus_idx_map.get(str(ev.bus_id), 0)
    sk = (ev.line_id, ev.stop_id)
    if sk not in station_index:
        station_index[sk] = ev.stop_idx if ev.stop_idx is not None and ev.stop_idx >= 0 else len(station_index)
    tp = int(ev.sim_time // 3600)
    if tp not in time_period_index:
        time_period_index[tp] = len(time_period_index)
    target_hw = line_headway.get(ev.line_id, 360.0)
    dyn_target = getattr(ev, 'target_forward_headway', target_hw)
    fp = getattr(ev, 'forward_bus_present', True)
    gap = (dyn_target - ev.forward_headway) if fp else 0.0
    return np.array([
        float(line_idx), float(bus_idx), float(station_index[sk]),
        float(time_period_index[tp]), float(int(ev.direction)),
        float(ev.forward_headway), float(ev.backward_headway),
        float(ev.waiting_passengers), float(target_hw),
        float(ev.base_stop_duration), float(ev.sim_time), float(gap),
        float(ev.co_line_forward_headway), float(ev.co_line_backward_headway),
        float(ev.segment_mean_speed),
    ], dtype=np.float32)


def compute_reward(ev, headway_fallback=360.0):
    def hr(hw, t): return -abs(hw - t)
    t_f = getattr(ev, 'target_forward_headway', headway_fallback)
    t_b = getattr(ev, 'target_backward_headway', headway_fallback)
    fp = getattr(ev, 'forward_bus_present', True)
    bp = getattr(ev, 'backward_bus_present', True)
    rf = hr(ev.forward_headway, t_f) if fp else None
    rb = hr(ev.backward_headway, t_b) if bp else None
    if rf is not None and rb is not None:
        fd, bd = abs(ev.forward_headway - t_f), abs(ev.backward_headway - t_b)
        w = fd / (fd + bd + 1e-6)
        R = t_f / max(t_b, 1e-6)
        sb = -abs(ev.forward_headway - R * ev.backward_headway) * 0.5 / ((1 + R) / 2)
        reward = rf * w + rb * (1 - w) + sb
    elif rf is not None: reward = rf
    elif rb is not None: reward = rb
    else: return -50.0
    f_pen = 20.0 * np.tanh((abs(ev.forward_headway - t_f) - 0.5 * t_f) / 30.0) if fp and t_f > 0 else 0.0
    b_pen = 20.0 * np.tanh((abs(ev.backward_headway - t_b) - 0.5 * t_b) / 30.0) if bp and t_b > 0 else 0.0
    reward -= max(0.0, f_pen + b_pen)
    return reward


def main():
    args = parse_args()
    t_start = time.time()

    line_idx_map, bus_idx_map = build_sumo_indices(SCHEDULE_XML)
    em = build_edge_linear_map(EDGE_XML, '7X')
    set_route_length(max(em.values()) if em else 13119.0)

    from sumo_env.rl_bridge import SumoRLBridge
    _patch_bridge_for_seed_and_scale(SumoRLBridge, args.sumo_seed, args.od_scale)
    bridge = SumoRLBridge(root_dir=SUMO_DIR, gui=False, max_steps=args.max_steps,
                          scale=args.od_scale)

    policy = DaganzoPolicy(alpha=args.alpha, use_two_sided=not args.single_sided)
    print(f"[Daganzo] alpha={policy.alpha} two_sided={policy.use_two_sided} "
          f"loaded headways for {len(policy.line_headways)} lines: "
          f"{ {k: round(v,1) for k,v in sorted(policy.line_headways.items())} }")

    station_index, time_period_index = {}, {}
    bridge.reset()
    line_headway = dict(bridge.line_headways)
    # Update policy headways from the env's actual computed values (more
    # accurate than the timetable for whatever scheduling is in effect).
    for k, v in line_headway.items():
        policy.line_headways.setdefault(k, v)

    pending = {}
    last_action = {}
    events_data = []

    for _ in range(100000):
        events, done, departed = bridge.fetch_events()
        for bid in departed:
            pending.pop(bid, None)
        if done:
            break
        if not events:
            continue

        for ev in events:
            bid = ev.bus_id
            obs = event_to_obs(ev, line_idx_map, bus_idx_map,
                               line_headway, station_index, time_period_index)
            si = int(obs[2])
            rew = compute_reward(ev)

            if bid in pending:
                prev = pending.pop(bid)
                if si != prev["si"]:
                    events_data.append({
                        "time": ev.sim_time, "line": ev.line_id,
                        "bus_id": str(bid), "station": si,
                        "fwd_hw": ev.forward_headway, "bwd_hw": ev.backward_headway,
                        "target_hw": getattr(ev, 'target_forward_headway', 360),
                        "reward": rew,
                        "hold": float(prev.get("hold", 0)),
                        "speed": float(prev.get("speed", 1.0)),
                    })

            raw_action = policy(ev, obs, bid, last_action)
            hold = float(np.clip(30.0 * raw_action[0] + 30.0, 0.0, 60.0))
            a_sp = float(raw_action[1])
            speed = 1.2 if a_sp > 0.6 else 1.1 if a_sp > 0.2 else 1.0 if a_sp > -0.2 else 0.9 if a_sp > -0.6 else 0.8

            bridge.apply_action(ev, [hold, speed])
            pending[bid] = {"si": si, "hold": hold, "speed": speed}
            last_action[bid] = raw_action.copy()

    # Snapshot passenger + bus dicts BEFORE close
    passenger_obj_dic = dict(getattr(bridge, 'passenger_obj_dic', {}))
    bus_obj_dic = dict(getattr(bridge, 'bus_obj_dic', {}))

    wall_time = time.time() - t_start
    bridge.close()

    if not events_data:
        print("No events collected!")
        return

    fwd_hws = np.array([e["fwd_hw"] for e in events_data])
    targets = np.array([e["target_hw"] for e in events_data])
    rewards = np.array([e["reward"] for e in events_data])
    holds = np.array([e["hold"] for e in events_data])

    n = len(events_data)
    cum_reward = float(rewards.sum())
    per_step = cum_reward / n
    hw_dev = np.abs(fwd_hws - targets)
    hw_std = float(fwd_hws.std())
    bunching_rate = float((fwd_hws < args.bunching_threshold).mean())
    severe_bunching = float((fwd_hws < args.bunching_threshold * 0.5).mean())

    per_line = {}
    for e in events_data:
        per_line.setdefault(e["line"], {"r": [], "h": [], "a": []})
        per_line[e["line"]]["r"].append(e["reward"])
        per_line[e["line"]]["h"].append(e["fwd_hw"])
        per_line[e["line"]]["a"].append(e["hold"])
    per_line_summary = {}
    for lid, d in sorted(per_line.items()):
        r = np.array(d["r"]); h = np.array(d["h"]); a = np.array(d["a"])
        per_line_summary[lid] = {
            "n": len(r), "reward_sum": float(r.sum()),
            "reward_mean": float(r.mean()),
            "hw_mean": float(h.mean()), "hw_std": float(h.std()),
            "bunching_rate": float((h < args.bunching_threshold).mean()),
            "hold_mean": float(a.mean()),
        }

    # ── Passenger-side metrics (mirrors eval_with_metrics.py) ──
    pax_waits, pax_travels, pax_in_vehicle = [], [], []
    for p in passenger_obj_dic.values():
        for leg in getattr(p, 'travel_data_l', []):
            try:
                arr, board, alight, wait, travel, _bid = leg
            except Exception:
                continue
            if wait is None or wait < 0 or wait > 7200:
                continue
            pax_waits.append(float(wait))
            pax_travels.append(float(travel))
            in_veh = float(alight) - float(board)
            if in_veh >= 0:
                pax_in_vehicle.append(in_veh)

    def _stats(arr):
        if not arr:
            return None
        a = np.asarray(arr, dtype=float)
        return {"n": int(a.size), "mean": float(a.mean()),
                "std": float(a.std()), "p50": float(np.percentile(a, 50)),
                "p90": float(np.percentile(a, 90)), "max": float(a.max())}

    pax_wait_stats = _stats(pax_waits)
    pax_travel_stats = _stats(pax_travels)
    pax_in_veh_stats = _stats(pax_in_vehicle)

    # Per-(line,direction) headway CV
    by_ld = {}
    for e in events_data:
        # eval_daganzo currently does not store ev.direction in events_data;
        # fall back to line-only key when missing.
        d = e.get("direction", 0)
        key = f"{e['line']}_dir{d}"
        by_ld.setdefault(key, []).append(e["fwd_hw"])
    headway_cv_per_line = {}
    for key, vals in by_ld.items():
        a = np.asarray(vals, dtype=float)
        if a.size > 1 and a.mean() > 1e-6:
            headway_cv_per_line[key] = float(a.std() / a.mean())

    large_mask = fwd_hws >= (args.large_gap_factor * targets)
    large_gap_rate = float(large_mask.mean()) if large_mask.size else 0.0

    completed_trips = 0
    for b in bus_obj_dic.values():
        traj = getattr(b, 'trajectory_dict', None)
        if traj and len(traj) >= 1:
            completed_trips += 1

    cv_vals = [v for v in headway_cv_per_line.values() if v > 0]
    if len(cv_vals) >= 2:
        s = sum(cv_vals); sq = sum(v * v for v in cv_vals)
        jain_fairness = float((s * s) / (len(cv_vals) * sq + 1e-12))
    else:
        jain_fairness = 1.0

    results = {
        "policy": "daganzo_cooperative_holding",
        "method_tag": args.method_tag,
        "alpha": args.alpha,
        "two_sided": not args.single_sided,
        "sumo_seed": int(args.sumo_seed),
        "od_scale": float(args.od_scale),
        "n_decisions": n,
        "cumulative_reward": cum_reward,
        "per_step_reward": per_step,
        "wall_time_sec": float(wall_time),
        "headway": {
            "fwd_mean": float(fwd_hws.mean()),
            "fwd_std": hw_std,
            "deviation_mean": float(hw_dev.mean()),
            "bunching_rate": bunching_rate,
            "severe_bunching_rate": severe_bunching,
            "large_gap_rate": large_gap_rate,
            "large_gap_factor": float(args.large_gap_factor),
        },
        "action": {
            "hold_mean": float(holds.mean()),
            "hold_std": float(holds.std()),
            "hold_min": float(holds.min()),
            "hold_max": float(holds.max()),
        },
        "per_line": per_line_summary,
        "line_headways_used": {k: float(v) for k, v in policy.line_headways.items()},
        # Aggregator-required fields:
        "passenger_wait_mean": (pax_wait_stats or {}).get("mean"),
        "passenger_wait_p90":  (pax_wait_stats or {}).get("p90"),
        "passenger_wait_stats": pax_wait_stats,
        "in_vehicle_delay_stats": pax_in_veh_stats,
        "total_travel_time_stats": pax_travel_stats,
        "headway_cv_per_line": headway_cv_per_line,
        "large_gap_rate": large_gap_rate,
        "completed_trips": int(completed_trips),
        "jain_fairness": jain_fairness,
    }

    print("=" * 60)
    print(f"Daganzo (alpha={args.alpha}, two_sided={not args.single_sided})")
    print("=" * 60)
    print(f"  Reward:         {cum_reward:>12,.0f} (per-step {per_step:.1f})")
    print(f"  Decisions:      {n:>12,d}")
    print(f"  Wall time:      {wall_time:>12.0f}s")
    print(f"  Headway std:    {hw_std:>12.1f}s")
    print(f"  Deviation mean: {hw_dev.mean():>12.1f}s")
    print(f"  Bunching rate:  {bunching_rate*100:>11.1f}%  (<{args.bunching_threshold}s)")
    print(f"  Severe bunch:   {severe_bunching*100:>11.1f}%  (<{args.bunching_threshold*0.5}s)")
    print(f"  Hold mean:      {holds.mean():>12.1f}s (std={holds.std():.1f})")
    print("  Per-line:")
    for lid, s in sorted(per_line_summary.items()):
        print(f"    {lid:6s}: n={s['n']:5d} r={s['reward_sum']:>10,.0f} "
              f"hw_std={s['hw_std']:.0f} bunch={s['bunching_rate']*100:.0f}% "
              f"hold_mean={s['hold_mean']:.1f}s")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
