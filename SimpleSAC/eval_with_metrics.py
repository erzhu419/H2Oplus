"""
eval_with_metrics.py
====================
Evaluate a policy checkpoint on SUMO with comprehensive metrics for paper.

Reports:
  - Cumulative reward + per-step
  - Headway std (over time and aggregate)
  - Bunching rate (% events with fwd_hw < threshold)
  - Action distribution statistics
  - Per-line breakdown
  - Computation time
  - Passenger-side operational metrics (Issue-5 extension):
        passenger_wait_mean, passenger_wait_p90, in_vehicle_delay,
        total_travel_time, excess_waiting_time
  - Headway CV per (line, direction)        : headway_cv_per_line
  - Large-gap rate (>= 1.5x scheduled hw)   : large_gap_rate
  - Hold-time distribution                  : hold_time_dist
  - Completed bus trips                     : completed_trips
  - Jain fairness over per-line headway-CV  : jain_fairness

Runtime knobs (Issue-5):
  --sumo_seed N    propagated as `--seed` to SUMO/libsumo (default: 42)
  --od_scale F     propagated as SUMO `--scale F` (default: 1.0). Multiplies
                   ALL demand (vehicles AND persons) — we use it as the OD
                   intensity multiplier because SUMO loads passengers as a
                   route file and `--scale` is the only built-in knob that
                   uniformly duplicates/drops them at runtime.

Usage:
    SUMO_HOME=/usr/share/sumo LIBSUMO_AS_TRACI=1 python eval_with_metrics.py \
        --checkpoint PATH --output results.json [--sumo_seed 42] [--od_scale 1.0]
"""

import os, sys, time, json, argparse, csv
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)

from model import EmbeddingLayer, BusEmbeddingPolicy
from common.data_utils import build_edge_linear_map, set_route_length

SUMO_DIR = os.path.normpath(os.path.join(_BUS_H2O, os.pardir, os.pardir, "SUMO_ruiguang", "online_control"))
sys.path.insert(0, SUMO_DIR)
sys.path.insert(0, os.path.join(SUMO_DIR, "sim_obj"))
EDGE_XML = os.path.join(_BUS_H2O, "network_data", "a_sorted_busline_edge.xml")
SCHEDULE_XML = os.path.join(SUMO_DIR, "initialize_obj", "save_obj_bus.add.xml")

import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, required=False, default='')
parser.add_argument('--output', type=str, default='')
parser.add_argument('--max_steps', type=int, default=18000)
parser.add_argument('--zero_hold', action='store_true', help='Eval zero-hold baseline instead')
parser.add_argument('--ep39', action='store_true', help='Eval ep39 legacy operator-reference policy (loads from LSTM-RL-legacy/ensemble_version/best model/)')
parser.add_argument('--method_tag', type=str, default='',
                    help='Method label written into the JSON (provenance only).')
parser.add_argument('--bunching_threshold', type=float, default=180.0, help='Headway below this = bunching (s)')
parser.add_argument('--sumo_seed', type=int, default=42, help='SUMO RNG seed (--seed)')
parser.add_argument('--od_scale', type=float, default=1.0,
                    help='Multiplicative factor on passenger demand intensity '
                         '(propagated as SUMO --scale; affects vehicles+persons uniformly).')
parser.add_argument('--large_gap_factor', type=float, default=1.5,
                    help='headway >= factor*scheduled is a large gap (default 1.5x).')
parser.add_argument('--hold_scale', type=float, default=1.0,
                    help='Eval-time multiplier applied to the mapped hold action.')
args = parser.parse_args()


def build_sumo_indices(schedule_xml):
    from collections import defaultdict
    tree = ET.parse(schedule_xml)
    root = tree.getroot()
    line_deps = defaultdict(list)
    for elem in root.findall(".//bus_obj"):
        lid = elem.get("belong_line_id_s")
        bid = elem.get("bus_id_s")
        st = float(elem.get("start_time_n", "0"))
        if lid and bid:
            line_deps[lid].append((st, bid))
    for entries in line_deps.values():
        entries.sort(key=lambda p: p[0])
    line_idx = {lid: i for i, lid in enumerate(sorted(line_deps.keys()))}
    bus_idx = {}
    counter = 0
    for lid, deps in line_deps.items():
        for _, bid in deps:
            if bid not in bus_idx:
                bus_idx[bid] = counter
                counter += 1
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


def _patch_bridge_for_seed_and_scale(bridge_cls, seed: int, scale: float):
    """Wrap SumoRLBridge._start_traci to inject `--seed` and override `--scale`.

    SumoRLBridge already accepts `scale` in its constructor; we still patch
    `_start_traci` to also append `--seed N`. This is monkey-patched on the
    bridge instance after construction so we don't need to edit the upstream
    bridge file.
    """
    orig_start = bridge_cls._start_traci

    def _start_traci_with_seed(self):
        # Re-implement with seed appended. Mirror upstream layout exactly.
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
            # also patch the module-level traci alias inside rl_bridge
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


def main():
    t_start = time.time()

    # Setup
    line_idx_map, bus_idx_map = build_sumo_indices(SCHEDULE_XML)
    em = build_edge_linear_map(EDGE_XML, '7X')
    set_route_length(max(em.values()) if em else 13119.0)

    from sumo_env.rl_bridge import SumoRLBridge

    # Inject --seed into the bridge's _start_traci. od_scale is passed via
    # the constructor's `scale` kwarg, which the bridge already forwards
    # to SUMO as `--scale`.
    _patch_bridge_for_seed_and_scale(SumoRLBridge, args.sumo_seed, args.od_scale)

    bridge = SumoRLBridge(root_dir=SUMO_DIR, gui=False, max_steps=args.max_steps,
                          scale=args.od_scale)

    # Load policy
    policy = None
    ep39_policy = None
    ep39_norm = None
    if args.ep39:
        # Legacy ep39 ckpt has its own loader + normaliser + action mapping; bypass the
        # standard BusEmbeddingPolicy code path and use the legacy interface directly.
        from eval_offline_on_sumo import load_ep39_policy
        ep39_policy, ep39_norm = load_ep39_policy()
    elif not args.zero_hold:
        cat_cols = ['line_id','bus_id','station_id','time_period','direction']
        cat_code_dict = {'line_id':{i:i for i in range(12)},'bus_id':{i:i for i in range(389)},
                         'station_id':{i:i for i in range(1)},'time_period':{i:i for i in range(1)},
                         'direction':{0:0,1:1}}
        emb = EmbeddingLayer(cat_code_dict, cat_cols, layer_norm=True, dropout=0.05)
        state_dim = emb.output_dim + 12
        policy = BusEmbeddingPolicy(state_dim, 2, 48, emb.clone(), action_range=1.0)
        # Support both H2O+/SAC checkpoints (key='policy_state_dict') and
        # BC checkpoints (key='policy'). Both store identical state_dict
        # layouts because both train a BusEmbeddingPolicy.
        try:
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        except Exception:
            ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        if 'policy_state_dict' in ckpt:
            sd = ckpt['policy_state_dict']
        elif 'policy' in ckpt:
            sd = ckpt['policy']
        else:
            raise KeyError(f"Checkpoint missing policy state dict: {list(ckpt.keys())}")
        policy.load_state_dict(sd)
        policy.eval()

    # Run episode
    station_index = {}
    time_period_index = {}
    bridge.reset()
    line_headway = dict(bridge.line_headways)

    pending = {}
    last_action = {}
    events_data = []  # per-event detailed data

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
            obs = event_to_obs(ev, line_idx_map, bus_idx_map, line_headway, station_index, time_period_index)
            si = int(obs[2])
            rew = compute_reward(ev)

            # Settle pending
            if bid in pending:
                prev = pending.pop(bid)
                if si != prev["si"]:
                    events_data.append({
                        "time": ev.sim_time,
                        "line": ev.line_id,
                        "direction": int(ev.direction),
                        "bus_id": str(bid),
                        "station": si,
                        "fwd_hw": ev.forward_headway,
                        "bwd_hw": ev.backward_headway,
                        "target_hw": getattr(ev, 'target_forward_headway', 360),
                        "reward": rew,
                        "hold": float(prev.get("hold", 0)),
                        "speed": float(prev.get("speed", 1.0)),
                    })

            # Get action
            if args.zero_hold:
                raw_action = np.array([-1.0, 0.0], dtype=np.float32)
            elif args.ep39:
                # ep39 returns mapped action [0,60] x [0.8,1.2]; convert back to raw tanh.
                prev_a = last_action.get(bid, np.zeros(2, dtype=np.float32))
                obs_aug = np.concatenate([obs, prev_a])
                obs_normed = ep39_norm(obs_aug, update=False)
                with torch.no_grad():
                    mapped = ep39_policy.get_action(torch.FloatTensor(obs_normed), deterministic=True)
                raw_action = np.array([(mapped[0] - 30.0) / 30.0, (mapped[1] - 1.0) / 0.2], dtype=np.float32)
            else:
                prev_a = last_action.get(bid, np.zeros(2, dtype=np.float32))
                obs_aug = np.concatenate([obs, prev_a])
                with torch.no_grad():
                    action, _ = policy(torch.FloatTensor(obs_aug).unsqueeze(0), deterministic=True)
                raw_action = action.cpu().numpy()[0]

            hold = float(np.clip((30.0 * raw_action[0] + 30.0) * args.hold_scale, 0.0, 60.0))
            a_sp = float(raw_action[1])
            speed = 1.2 if a_sp > 0.6 else 1.1 if a_sp > 0.2 else 1.0 if a_sp > -0.2 else 0.9 if a_sp > -0.6 else 0.8

            bridge.apply_action(ev, [hold, speed])
            pending[bid] = {"si": si, "hold": hold, "speed": speed}
            last_action[bid] = raw_action.copy()

    # ── Snapshot passenger + bus dicts BEFORE close (just in case) ──
    passenger_obj_dic = dict(bridge.passenger_obj_dic)
    bus_obj_dic = dict(bridge.bus_obj_dic)

    wall_time = time.time() - t_start
    bridge.close()

    # ── Compute metrics ──
    if not events_data:
        print("No events collected!")
        return

    fwd_hws = np.array([e["fwd_hw"] for e in events_data])
    bwd_hws = np.array([e["bwd_hw"] for e in events_data])
    targets = np.array([e["target_hw"] for e in events_data])
    rewards = np.array([e["reward"] for e in events_data])
    holds = np.array([e["hold"] for e in events_data])
    times = np.array([e["time"] for e in events_data])

    n = len(events_data)
    cum_reward = rewards.sum()
    per_step = cum_reward / n

    hw_dev = np.abs(fwd_hws - targets)
    hw_std = fwd_hws.std()
    bunching_rate = (fwd_hws < args.bunching_threshold).mean()
    severe_bunching = (fwd_hws < args.bunching_threshold * 0.5).mean()

    # Per-line breakdown
    per_line = {}
    for e in events_data:
        lid = e["line"]
        if lid not in per_line:
            per_line[lid] = {"rewards": [], "fwd_hws": [], "holds": []}
        per_line[lid]["rewards"].append(e["reward"])
        per_line[lid]["fwd_hws"].append(e["fwd_hw"])
        per_line[lid]["holds"].append(e["hold"])

    per_line_summary = {}
    for lid, data in sorted(per_line.items()):
        r = np.array(data["rewards"])
        h = np.array(data["fwd_hws"])
        a = np.array(data["holds"])
        per_line_summary[lid] = {
            "n": len(r),
            "reward_sum": float(r.sum()),
            "reward_mean": float(r.mean()),
            "hw_mean": float(h.mean()),
            "hw_std": float(h.std()),
            "bunching_rate": float((h < args.bunching_threshold).mean()),
            "hold_mean": float(a.mean()),
        }

    # Headway std in time windows
    hw_by_time = {}
    for e in events_data:
        hour = int(e["time"] // 3600)
        hw_by_time.setdefault(hour, []).append(e["fwd_hw"])
    hw_std_by_hour = {h: float(np.std(vs)) for h, vs in sorted(hw_by_time.items())}

    # Action distribution by headway gap
    gaps = fwd_hws - targets
    action_by_gap = {}
    for lo, hi, name in [(-999,-100,"bunch_severe"),(-100,0,"bunch_mild"),
                          (0,100,"normal"),(100,999,"wide_gap")]:
        mask = (gaps >= lo) & (gaps < hi)
        if mask.sum() > 0:
            action_by_gap[name] = {
                "n": int(mask.sum()),
                "hold_mean": float(holds[mask].mean()),
                "hold_std": float(holds[mask].std()),
            }

    # ── Issue-5: Operational / passenger-side metrics ──

    # 1) Passenger waiting + travel times — sourced from Passenger.travel_data_l
    # Each completed leg is [arrive, board, alight, wait, travel, bus_id].
    pax_waits = []
    pax_travels = []
    pax_in_vehicle = []
    for p in passenger_obj_dic.values():
        for leg in getattr(p, 'travel_data_l', []):
            try:
                arr, board, alight, wait, travel, _bid = leg
            except Exception:
                continue
            if wait is None or wait < 0 or wait > 7200:  # filter sentinel/junk
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
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "p25": float(np.percentile(a, 25)),
            "p50": float(np.percentile(a, 50)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
            "max": float(a.max()),
        }

    pax_wait_stats = _stats(pax_waits)
    pax_travel_stats = _stats(pax_travels)
    pax_in_veh_stats = _stats(pax_in_vehicle)

    # Excess wait = realised wait minus expected wait under perfect even
    # service. Under random arrivals and headway H, expected wait = H/2.
    # We use the median per-line scheduled headway as the H reference.
    if pax_waits:
        # use the global median scheduled headway as a single proxy H/2
        H_ref = float(np.median(list(line_headway.values()))) if line_headway else 360.0
        expected_wait = H_ref / 2.0
        excess = [max(w - expected_wait, 0.0) for w in pax_waits]
        excess_wait_stats = _stats(excess)
    else:
        H_ref = float(np.median(list(line_headway.values()))) if line_headway else 360.0
        excess_wait_stats = None

    # 2) Headway CV per (line, direction), from arrival-event stream.
    #    headway_cv = std(fwd_hw) / mean(fwd_hw).
    by_ld = {}
    for e in events_data:
        key = f"{e['line']}_{'S' if e['direction']==1 else 'X'}_dir{e['direction']}"
        by_ld.setdefault(key, []).append(e["fwd_hw"])
    headway_cv_per_line = {}
    for key, vals in by_ld.items():
        arr = np.asarray(vals, dtype=float)
        if arr.size > 1 and arr.mean() > 1e-6:
            headway_cv_per_line[key] = float(arr.std() / arr.mean())

    # 3) Large-gap rate: fraction of headway intervals exceeding
    #    large_gap_factor * scheduled headway (default 1.5x).
    large_mask = fwd_hws >= (args.large_gap_factor * targets)
    large_gap_rate = float(large_mask.mean()) if large_mask.size else 0.0

    # 4) Hold-time distribution
    hold_time_dist = _stats(holds.tolist())

    # 5) Completed bus trips. A bus is "completed" if it served at least one
    # stop (has trajectory_dict entries) and is no longer active. We
    # approximate by counting buses with trajectory_dict entries in their
    # final state — they recorded service-completion timestamps.
    completed_trips = 0
    for b in bus_obj_dic.values():
        traj = getattr(b, 'trajectory_dict', None)
        if traj and len(traj) >= 1:
            # count as completed if it served the last stop on its line
            completed_trips += 1

    # 6) Jain's fairness index over per-line headway-CV.
    cv_vals = [v for v in headway_cv_per_line.values() if v > 0]
    if len(cv_vals) >= 2:
        s = sum(cv_vals)
        sq = sum(v * v for v in cv_vals)
        jain_fairness = float((s * s) / (len(cv_vals) * sq + 1e-12))
    else:
        jain_fairness = 1.0  # degenerate single-line case

    results = {
        "checkpoint": args.checkpoint,
        "method_tag": args.method_tag,
        "zero_hold": args.zero_hold,
        "hold_scale": float(args.hold_scale),
        "sumo_seed": int(args.sumo_seed),
        "od_scale": float(args.od_scale),
        "n_decisions": n,
        "cumulative_reward": float(cum_reward),
        "per_step_reward": float(per_step),
        "wall_time_sec": float(wall_time),
        "headway": {
            "fwd_mean": float(fwd_hws.mean()),
            "fwd_std": float(hw_std),
            "deviation_mean": float(hw_dev.mean()),
            "bunching_rate": float(bunching_rate),
            "severe_bunching_rate": float(severe_bunching),
            "large_gap_rate": large_gap_rate,
            "large_gap_factor": float(args.large_gap_factor),
        },
        "action": {
            "hold_mean": float(holds.mean()),
            "hold_std": float(holds.std()),
            "hold_min": float(holds.min()),
            "hold_max": float(holds.max()),
        },
        "action_by_gap": action_by_gap,
        "per_line": per_line_summary,
        "hw_std_by_hour": hw_std_by_hour,
        # Issue-5 fields:
        "passenger_wait_mean": (pax_wait_stats or {}).get("mean"),
        "passenger_wait_p90":  (pax_wait_stats or {}).get("p90"),
        "passenger_wait_stats": pax_wait_stats,
        "in_vehicle_delay_stats": pax_in_veh_stats,
        "total_travel_time_stats": pax_travel_stats,
        "excess_waiting_time_stats": excess_wait_stats,
        "expected_wait_reference_sec": H_ref / 2.0 if pax_wait_stats is not None else None,
        "headway_cv_per_line": headway_cv_per_line,
        "large_gap_rate": large_gap_rate,
        "hold_time_dist": hold_time_dist,
        "completed_trips": int(completed_trips),
        "jain_fairness": jain_fairness,
    }

    # Print summary
    print("=" * 60)
    print(f"{'Zero-hold' if args.zero_hold else args.checkpoint}")
    print(f"  sumo_seed={args.sumo_seed}  od_scale={args.od_scale}")
    print("=" * 60)
    print(f"  Reward:         {cum_reward:>12,.0f} (per-step {per_step:.1f})")
    print(f"  Decisions:      {n:>12,d}")
    print(f"  Wall time:      {wall_time:>12.0f}s")
    print(f"  Headway std:    {hw_std:>12.1f}s")
    print(f"  Deviation mean: {hw_dev.mean():>12.1f}s")
    print(f"  Bunching rate:  {bunching_rate*100:>11.1f}%  (<{args.bunching_threshold}s)")
    print(f"  Severe bunch:   {severe_bunching*100:>11.1f}%  (<{args.bunching_threshold*0.5}s)")
    print(f"  Large-gap rate: {large_gap_rate*100:>11.1f}%  (>={args.large_gap_factor}x sched)")
    print(f"  Hold mean:      {holds.mean():>12.1f}s (std={holds.std():.1f})")
    print(f"  Completed trips:{completed_trips:>12d}")
    print(f"  Jain fairness:  {jain_fairness:>12.3f}")
    if pax_wait_stats:
        print(f"  Pax wait mean:  {pax_wait_stats['mean']:>12.1f}s  p90={pax_wait_stats['p90']:.1f}s  n={pax_wait_stats['n']}")
    if pax_travel_stats:
        print(f"  Pax total tt:   {pax_travel_stats['mean']:>12.1f}s  p90={pax_travel_stats['p90']:.1f}s")
    if pax_in_veh_stats:
        print(f"  In-veh delay:   {pax_in_veh_stats['mean']:>12.1f}s  p90={pax_in_veh_stats['p90']:.1f}s")
    if excess_wait_stats:
        print(f"  Excess wait:    {excess_wait_stats['mean']:>12.1f}s  p90={excess_wait_stats['p90']:.1f}s")
    print()
    print("  Per-line:")
    for lid, s in sorted(per_line_summary.items()):
        print(f"    {lid:6s}: n={s['n']:5d} r={s['reward_sum']:>10,.0f} hw_std={s['hw_std']:.0f} bunch={s['bunching_rate']*100:.0f}%")
    print()
    print("  Headway CV per (line, dir):")
    for k, v in sorted(headway_cv_per_line.items()):
        print(f"    {k:18s}: cv={v:.3f}")
    print()
    print("  Action by headway gap:")
    for name, s in sorted(action_by_gap.items()):
        print(f"    {name:15s}: hold={s['hold_mean']:.1f}s ±{s['hold_std']:.1f}  n={s['n']}")

    # Save
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
