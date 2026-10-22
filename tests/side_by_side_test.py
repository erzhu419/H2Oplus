#!/usr/bin/env python
"""
side_by_side_test.py — SUMO ↔ Sim 1:1 对比测试
================================================

使用:
    cd H2Oplus/tests
    LIBSUMO_AS_TRACI=1 /home/erzhu419/anaconda3/envs/LSTM-RL/bin/python side_by_side_test.py
"""

import os
import sys
import json
import pickle
import time
import warnings
import traceback
from collections import defaultdict

import numpy as np

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H2O_ROOT   = os.path.dirname(SCRIPT_DIR)  # H2Oplus/
SIM_ROOT   = os.path.join(H2O_ROOT, "bus_h2o")
SUMO_ROOT  = os.path.normpath(os.path.join(H2O_ROOT, "..", "SUMO_ruiguang"))
BRIDGE_DIR = os.path.normpath(os.path.join(SUMO_ROOT, "online_control"))

sys.path.insert(0, SIM_ROOT)

# Match collect_worker.py path setup exactly:
# 1. SUMO_DIR = online_control  (for case modules)
SUMO_OC = os.path.join(SUMO_ROOT, "online_control")
sys.path.insert(0, SUMO_OC)
# 2. sumo_env/case
_CASE_DIR = os.path.join(SIM_ROOT, "sumo_env", "case")
if os.path.isdir(_CASE_DIR):
    sys.path.insert(0, _CASE_DIR)
sys.path.insert(0, os.path.join(H2O_ROOT, "collect_policy"))

LOG_DIR = os.path.join(SCRIPT_DIR, "side_by_side_logs")
os.makedirs(LOG_DIR, exist_ok=True)

LINE_ID = "7X"
SCHEDULE_XML = os.path.join(BRIDGE_DIR, "initialize_obj", "save_obj_bus.add.xml")
EDGE_XML = os.path.join(SUMO_ROOT, "online_control", "intersection_delay",
                        "a_sorted_busline_edge.xml")


def _build_indices():
    """Build stable bus/line indices from schedule XML."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(SCHEDULE_XML)
    line_deps = defaultdict(list)
    for elem in tree.getroot().findall(".//bus_obj"):
        lid = elem.get("belong_line_id_s")
        bid = elem.get("bus_id_s")
        st  = float(elem.get("start_time_n", "0"))
        if lid and bid:
            line_deps[lid].append((st, bid))
    for entries in line_deps.values():
        entries.sort()

    line_index = {lid: i for i, lid in enumerate(sorted(line_deps.keys()))}
    bus_index = {}
    counter = 0
    for lid, deps in line_deps.items():
        for _, bid in deps:
            if bid not in bus_index:
                bus_index[bid] = counter
                counter += 1

    times_7x = [t for t, _ in line_deps.get(LINE_ID, [])]
    diffs = [b - a for a, b in zip(times_7x[:-1], times_7x[1:]) if b > a]
    line_headway = float(np.median(diffs)) if diffs else 360.0

    return line_index, bus_index, line_headway


def _compute_reward(ev_raw, line_headway):
    """Compute reward matching rl_env + collect_worker."""
    fwd_hw = ev_raw["forward_headway"]
    bwd_hw = ev_raw["backward_headway"]
    t_f = ev_raw["target_forward_headway"]
    t_b = ev_raw["target_backward_headway"]
    fp  = ev_raw["forward_bus_present"]
    bp  = ev_raw["backward_bus_present"]

    def hr(hw, t): return -abs(hw - t)
    rf = hr(fwd_hw, t_f) if fp else None
    rb = hr(bwd_hw, t_b) if bp else None

    if rf is not None and rb is not None:
        fd = abs(fwd_hw - t_f); bd = abs(bwd_hw - t_b)
        w = fd / (fd + bd + 1e-6)
        R = t_f / max(t_b, 1e-6)
        sb = -abs(fwd_hw - R * bwd_hw) * 0.5 / ((1+R)/2)
        reward = rf * w + rb * (1-w) + sb
    elif rf is not None: reward = rf
    elif rb is not None: reward = rb
    else: return -50.0

    f_pen = (20.0*np.tanh((abs(fwd_hw-t_f) - 0.5*t_f)/30.0) if fp and t_f > 0 else 0.0)
    b_pen = (20.0*np.tanh((abs(bwd_hw-t_b) - 0.5*t_b)/30.0) if bp and t_b > 0 else 0.0)
    reward -= max(0.0, f_pen + b_pen)
    return reward


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: SUMO episode
# ═══════════════════════════════════════════════════════════════════════

def run_sumo_episode():
    print("\n" + "="*70)
    print(" PHASE 1: Running SUMO episode (zero-hold)")
    print("="*70, flush=True)

    try:
        from sumo_env.rl_bridge import SumoRLBridge
        from common.data_utils import build_edge_linear_map
        from sumo_env.sumo_snapshot import bridge_to_full_snapshot
    except Exception as ex:
        print(f"  Import error: {ex}", flush=True)
        import traceback; traceback.print_exc()
        return []

    line_index, bus_index, line_headway = _build_indices()
    edge_map = build_edge_linear_map(EDGE_XML, LINE_ID)

    try:
        # Clean stale libsumo state files from any directory
        import glob
        for sbx in glob.glob(os.path.join(SUMO_OC, "sumo_start_state.sbx")):
            os.remove(sbx)
        for sbx in glob.glob(os.path.join(SCRIPT_DIR, "sumo_start_state.sbx")):
            os.remove(sbx)
        
        bridge = SumoRLBridge(root_dir=SUMO_OC)
        bridge.first_run = True  # Force fresh SUMO start
        bridge.reset()
    except Exception as ex:
        print(f"  Bridge init error: {ex}", flush=True)
        import traceback; traceback.print_exc()
        return []

    sumo_log = []
    station_index = {}

    print(f"  line_headway={line_headway:.1f}s, buses={len(bus_index)}", flush=True)

    for fetch_iter in range(100000):
        events, done, departed = bridge.fetch_events()
        if done:
            break
        if not events:
            continue

        for ev in events:
            if ev.line_id != LINE_ID:
                bridge.apply_action(ev, 0.0)
                continue

            li = line_index.get(ev.line_id, 0)
            bi = bus_index.get(ev.bus_id, 0)
            station_idx = ev.stop_idx
            tp = int(ev.sim_time // 3600)
            direction = int(ev.direction)
            dyn_target = ev.target_forward_headway
            gap = (dyn_target - ev.forward_headway) if ev.forward_bus_present else 0.0

            obs = [
                float(li), float(bi), float(station_idx), float(tp),
                float(direction), float(ev.forward_headway),
                float(ev.backward_headway), float(ev.waiting_passengers),
                float(line_headway), float(ev.base_stop_duration),
                float(ev.sim_time), float(gap),
                float(ev.co_line_forward_headway),
                float(ev.co_line_backward_headway),
                float(ev.segment_mean_speed),
            ]

            ev_raw = {
                "forward_headway": ev.forward_headway,
                "backward_headway": ev.backward_headway,
                "forward_bus_present": ev.forward_bus_present,
                "backward_bus_present": ev.backward_bus_present,
                "target_forward_headway": ev.target_forward_headway,
                "target_backward_headway": ev.target_backward_headway,
                "waiting_passengers": ev.waiting_passengers,
                "base_stop_duration": ev.base_stop_duration,
                "co_line_forward_headway": ev.co_line_forward_headway,
                "co_line_backward_headway": ev.co_line_backward_headway,
                "segment_mean_speed": ev.segment_mean_speed,
            }
            reward = _compute_reward(ev_raw, line_headway)

            snap = {}
            try:
                snap = bridge_to_full_snapshot(bridge, edge_map, bus_index=bus_index)
                snap = snap.get(LINE_ID, {})
            except Exception as ex:
                if len(sumo_log) < 3:
                    print(f"  Snapshot error: {ex}", flush=True)

            sumo_log.append({
                "sim_time": ev.sim_time, "bus_id": ev.bus_id,
                "bus_idx": bi, "station_idx": station_idx,
                "obs": obs, "reward": reward, "snapshot": snap,
                "event_raw": ev_raw,
            })

            bridge.apply_action(ev, 0.0)

    try:
        bridge.close()
    except Exception:
        pass
    print(f"  SUMO: {len(sumo_log)} events for {LINE_ID}", flush=True)

    log_path = os.path.join(LOG_DIR, "sumo_log.pkl")
    with open(log_path, "wb") as f:
        pickle.dump(sumo_log, f, protocol=4)
    print(f"  Saved to {log_path}", flush=True)
    return sumo_log


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Sim episode
# ═══════════════════════════════════════════════════════════════════════

def run_sim_episode():
    print("\n" + "="*70)
    print(" PHASE 2: Running Sim episode (zero-hold)")
    print("="*70, flush=True)

    from sim_core.sim import env_bus
    env_bus._DATA_CACHE.clear()
    from envs.bus_sim_env import BusSimEnv

    env = BusSimEnv(path=SIM_ROOT)
    env.reset()

    sim_log = []
    for step_i in range(80000):
        state, reward, done, info = env.step(env.action_dict)
        if done:
            break
        for b in env.bus_all:
            if len(b.obs) > 0:
                snap = env.capture_full_system_snapshot()
                sim_log.append({
                    "sim_time": env.current_time,
                    "bus_id": b.bus_id,
                    "trip_id": b.trip_id,
                    "sumo_trip_index": getattr(b, 'sumo_trip_index', b.trip_id),
                    "station_name": b.last_station.station_name,
                    "obs": list(b.obs),
                    "reward": b.reward if b.reward is not None else 0.0,
                    "snapshot": snap,
                })

    print(f"  Sim: {len(sim_log)} events, final_time={env.current_time:.0f}s", flush=True)
    log_path = os.path.join(LOG_DIR, "sim_log.pkl")
    with open(log_path, "wb") as f:
        pickle.dump(sim_log, f, protocol=4)
    return sim_log


# ═══════════════════════════════════════════════════════════════════════
# Phase 3: Compare
# ═══════════════════════════════════════════════════════════════════════

OBS_NAMES = [
    "line_id", "bus_id", "station_id", "time_period", "direction",
    "fwd_headway", "bwd_headway", "waiting_pax",
    "target_hw", "base_stop_dur", "sim_time", "gap",
    "co_line_fwd", "co_line_bwd", "seg_speed",
]

CATEGORICAL = {0, 1, 2, 3, 4}
EXPECTED_GAP = {7, 9, 12, 13, 14}   # known sim-to-real gap dims


def compare_logs(sumo_log, sim_log):
    print("\n" + "="*70)
    print(" PHASE 3: Dimension-wise comparison")
    print("="*70, flush=True)

    print(f"\n  SUMO events: {len(sumo_log)}")
    print(f"  Sim  events: {len(sim_log)}")

    sumo_arr = np.array([e["obs"] for e in sumo_log])
    sim_arr  = np.array([e["obs"] for e in sim_log])

    header = f"{'Dim':>4s} {'Name':>15s} | {'SUMO mean':>10s} {'std':>8s} | {'Sim mean':>10s} {'std':>8s} | {'Status'}"
    print(f"\n{header}")
    print("-" * 85)

    for i in range(15):
        sm, ss = sumo_arr[:, i].mean(), sumo_arr[:, i].std()
        mm, ms = sim_arr[:, i].mean(), sim_arr[:, i].std()

        if i in CATEGORICAL:
            su = sorted(set(sumo_arr[:, i].astype(int)))
            mu = sorted(set(sim_arr[:, i].astype(int)))
            status = "✅ match" if su == mu else f"❌ S:{su[:5]} M:{mu[:5]}"
        elif i in EXPECTED_GAP:
            status = "🟢 expected gap"
        else:
            rd = abs(sm - mm) / (abs(sm) + 1e-6)
            status = "✅ close" if rd < 0.15 else ("🟡 moderate" if rd < 0.35 else f"⚠️ diff {rd:.0%}")

        print(f"[{i:>2d}] {OBS_NAMES[i]:>15s} | {sm:>10.2f} {ss:>8.2f} | {mm:>10.2f} {ms:>8.2f} | {status}")

    sr = np.array([e["reward"] for e in sumo_log])
    mr = np.array([e["reward"] for e in sim_log])
    print(f"\n  Reward: SUMO mean={sr.mean():.2f} std={sr.std():.2f}")
    print(f"          Sim  mean={mr.mean():.2f} std={mr.std():.2f}")

    # Bus overlap
    sumo_buses = defaultdict(int)
    sim_buses  = defaultdict(int)
    for e in sumo_log: sumo_buses[e["bus_idx"]] += 1
    for e in sim_log:  sim_buses[e.get("sumo_trip_index", e["trip_id"])] += 1
    common = set(sumo_buses) & set(sim_buses)
    print(f"\n  Bus overlap: {len(common)} common / SUMO {len(sumo_buses)} / Sim {len(sim_buses)}")
    diffs = {b: (sumo_buses[b], sim_buses[b]) for b in sorted(common)[:10]}
    print(f"  First 10 event counts (SUMO, Sim): {diffs}")

    # Time range
    st = [e["sim_time"] for e in sumo_log]
    mt = [e["sim_time"] for e in sim_log]
    print(f"\n  Time: SUMO [{min(st):.0f}..{max(st):.0f}], Sim [{min(mt):.0f}..{max(mt):.0f}]")

    # Snapshot bus counts
    sbc = [len(e["snapshot"].get("all_buses", [])) for e in sumo_log if e.get("snapshot")]
    mbc = [len(e["snapshot"].get("all_buses", [])) for e in sim_log if e.get("snapshot")]
    if sbc and mbc:
        print(f"  Snap bus count: SUMO mean={np.mean(sbc):.1f} max={max(sbc)}, "
              f"Sim mean={np.mean(mbc):.1f} max={max(mbc)}")


# ═══════════════════════════════════════════════════════════════════════
# Phase 4: Snapshot-reset test
# ═══════════════════════════════════════════════════════════════════════

def run_snapshot_reset_analysis(sumo_log):
    print("\n" + "="*70)
    print(" PHASE 4: Snapshot-Reset (5 checkpoints)")
    print("="*70, flush=True)

    from sim_core.sim import env_bus
    from envs.bus_sim_env import BusSimEnv

    valid = [e for e in sumo_log if e.get("snapshot") and e["snapshot"].get("all_buses")]
    if len(valid) < 5:
        print("  Not enough snapshots"); return []

    indices = np.linspace(0, len(valid)-1, 5, dtype=int)
    results = []

    for ti, idx in enumerate(indices):
        entry = valid[idx]
        snap = entry["snapshot"]
        t = entry["sim_time"]
        n_buses = len(snap.get("all_buses", []))
        print(f"\n  ── Test {ti+1}/5: t={t:.0f}s, {n_buses} buses ──", flush=True)

        try:
            env_bus._DATA_CACHE.clear()
            env = BusSimEnv(path=SIM_ROOT)
            env.reset()
            env.restore_full_system_snapshot(snap)

            active = sum(1 for b in env.bus_all if b.on_route)
            print(f"     Reset OK: sim_time={env.current_time:.0f}s, "
                  f"active={active}/{len(env.bus_all)}", flush=True)

            errors = []
            obs_count = 0
            for step in range(500):
                try:
                    state, reward, done, info = env.step(env.action_dict)
                    if done: break
                    for b in env.bus_all:
                        if len(b.obs) > 0:
                            obs_count += 1
                            o = b.obs
                            if not (0 <= o[0] <= 20):
                                errors.append(f"obs[0]={o[0]}")
                            if not (0 <= o[1] <= 500):
                                errors.append(f"obs[1]={o[1]}")
                            if o[5] < 0:
                                errors.append(f"obs[5] fwd_hw={o[5]:.1f}")
                except Exception as ex:
                    errors.append(f"step {step}: {ex}")
                    break

            status = "✅ OK" if not errors else f"❌ {len(errors)} errors"
            print(f"     500 steps: obs={obs_count}, {status}", flush=True)
            for err in errors[:3]:
                print(f"       {err}")
            results.append({"t": t, "status": status, "obs": obs_count, "errors": errors[:5]})

        except Exception as ex:
            print(f"     ❌ CRASH: {ex}")
            traceback.print_exc()
            results.append({"t": t, "status": f"CRASH: {ex}", "obs": 0, "errors": [str(ex)]})

    print("\n  ── Summary ──")
    for r in results:
        print(f"    t={r['t']:.0f}s → {r['status']} ({r['obs']} obs)")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    t0 = time.time()

    sumo_log = run_sumo_episode()
    sim_log  = run_sim_episode()

    if sumo_log and sim_log:
        compare_logs(sumo_log, sim_log)

    if sumo_log:
        run_snapshot_reset_analysis(sumo_log)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f" Total: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'='*70}")
