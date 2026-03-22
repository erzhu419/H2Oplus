"""
verify_phases.py
================
Phase 0-3 verification script for H2O+ sim-to-real framework.
Uses LSTM-RL-legacy single-line env_bus as SimpleSim.

Run:
    cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o
    python verify_phases.py --skip_sumo        # sim-only mode
    python verify_phases.py                    # with SUMO (if available)
"""

import sys, os, time, csv, argparse, cProfile, pstats, io
from collections import defaultdict
import numpy as np

# ────────────────────────── Paths ──────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LSTM_RL_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../../LSTM-RL-legacy'))
sys.path.insert(0, LSTM_RL_DIR)

OUT_DIR = os.path.join(SCRIPT_DIR, 'verify_results')
os.makedirs(OUT_DIR, exist_ok=True)

# ────────────────────────── CLI ──────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--skip_sumo', action='store_true', help='Skip SUMO-dependent tests')
parser.add_argument('--snapshot_time', type=int, default=2000, help='Sim time to capture snapshot (s)')
parser.add_argument('--post_reset_steps', type=int, default=3000, help='Steps to run after snapshot reset')
parser.add_argument('--num_runs', type=int, default=3, help='Number of episode runs for V2/V4')
args = parser.parse_args()


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════

def make_env():
    """Create a fresh LSTM-RL-legacy single-line env_bus."""
    from env.sim import env_bus
    env_path = os.path.join(LSTM_RL_DIR, 'env')
    env = env_bus(env_path, debug=False)
    env.enable_plot = False
    return env


def capture_snapshot(env):
    """Capture a lightweight snapshot of env state for z_t computation."""
    buses = []
    for b in env.bus_all:
        if b.on_route:
            buses.append({
                'pos': b.absolute_distance,
                'speed': b.current_speed,
                'load': len(b.passengers),
                'last_stop_index': b.last_station.station_id,
            })
    stations = []
    for s in env.stations:
        stations.append({
            'pos': s.station_id * 500.0,  # approximate position
            'waiting_count': len(s.waiting_passengers) if hasattr(s, 'waiting_passengers') else 0,
        })
    return {'all_buses': buses, 'all_stations': stations, 'global_time': env.current_time}


def extract_context(snapshot, num_segments=10, route_length=11500.0):
    """Simplified version of extract_structured_context from H2O+.md."""
    seg_len = route_length / num_segments
    seg_speeds = [[] for _ in range(num_segments)]
    seg_counts = [0] * num_segments
    seg_waiting = [0] * num_segments

    for bus in snapshot['all_buses']:
        idx = min(int(bus['pos'] / seg_len), num_segments - 1)
        idx = max(idx, 0)
        seg_speeds[idx].append(bus['speed'])
        seg_counts[idx] += 1

    for st in snapshot['all_stations']:
        idx = min(int(st['pos'] / seg_len), num_segments - 1)
        idx = max(idx, 0)
        seg_waiting[idx] += st['waiting_count']

    vec_speed = np.array([np.mean(s) if s else 5.0 for s in seg_speeds]) / 10.0
    vec_density = np.array(seg_counts) / 5.0
    vec_waiting = np.array(seg_waiting) / 20.0
    return np.concatenate([vec_speed, vec_density, vec_waiting]).astype(np.float32)


def cosine_sim(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return dot / (na * nb)


def run_sim_episode(env, action_value=0.0):
    """Run one complete episode with constant action, return event log."""
    env.reset()
    actions = {k: action_value for k in range(env.max_agent_num)}

    # Warmup until first obs
    while sum(1 for v in env.state.values() if v) == 0:
        env.step(actions)

    events = []  # (bus_id, station_id, time, fwd_hw, bwd_hw, pax_onboard, pax_waiting)
    while not env.done:
        state, reward, done = env.step(actions)
        for bus in env.state_bus_list:
            obs = bus.obs
            if obs:
                events.append({
                    'bus_id': bus.bus_id,
                    'station_id': bus.last_station.station_id,
                    'station_name': bus.last_station.station_name,
                    'time': env.current_time,
                    'fwd_hw': bus.forward_headway,
                    'bwd_hw': bus.backward_headway,
                    'pax_onboard': len(bus.passengers),
                    'pax_waiting': len(bus.next_station.waiting_passengers)
                                  if hasattr(bus.next_station, 'waiting_passengers') else 0,
                })
    return events


# ═══════════════════════════════════════════════════════════
#  V1: w-Decay after Snapshot Reset
# ═══════════════════════════════════════════════════════════

def test_v1_w_decay():
    print("\n" + "=" * 70)
    print("  V1: w-Decay after Snapshot Reset")
    print("=" * 70)

    env = make_env()
    env.reset()
    actions = {k: 0.0 for k in range(env.max_agent_num)}

    # Run until snapshot_time
    print(f"  Running sim until t={args.snapshot_time}s...")
    while env.current_time < args.snapshot_time and not env.done:
        env.step(actions)

    if env.done:
        print(f"  ⚠️ Sim ended at t={env.current_time} before snapshot_time={args.snapshot_time}")
        return

    snap = capture_snapshot(env)
    z_snapshot = extract_context(snap)
    print(f"  Captured snapshot at t={env.current_time}: {len(snap['all_buses'])} buses, "
          f"z_snapshot norm={np.linalg.norm(z_snapshot):.3f}")

    # Continue running from snapshot point and track z_t divergence
    decay_log = []
    steps_after = 0
    while not env.done and steps_after < args.post_reset_steps:
        env.step(actions)
        steps_after += 1
        if steps_after % 50 == 0 or steps_after <= 10:
            snap_t = capture_snapshot(env)
            z_t = extract_context(snap_t)
            cs = cosine_sim(z_snapshot, z_t)
            l2 = np.linalg.norm(z_t - z_snapshot)
            decay_log.append({
                'steps_after_snap': steps_after,
                'sim_time': env.current_time,
                'cosine_sim': cs,
                'l2_dist': l2,
                'num_buses': len(snap_t['all_buses']),
            })

    # Write CSV
    out_path = os.path.join(OUT_DIR, 'v1_w_decay.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=decay_log[0].keys())
        w.writeheader()
        w.writerows(decay_log)

    # Summary
    if len(decay_log) >= 2:
        cs_start = decay_log[0]['cosine_sim']
        cs_end = decay_log[-1]['cosine_sim']
        l2_start = decay_log[0]['l2_dist']
        l2_end = decay_log[-1]['l2_dist']
        print(f"\n  ── w-Decay Results ──")
        print(f"  Cosine sim: {cs_start:.4f} → {cs_end:.4f}  (Δ={cs_end - cs_start:+.4f})")
        print(f"  L2 distance: {l2_start:.3f} → {l2_end:.3f}  (Δ={l2_end - l2_start:+.3f})")
        if cs_end < cs_start:
            print(f"  ✅ z_t diverges from z_snapshot over time (w should decrease)")
        else:
            print(f"  ⚠️ z_t is NOT diverging — check context extraction or sim dynamics")
    print(f"  Wrote: {out_path}")


# ═══════════════════════════════════════════════════════════
#  V2: Passenger Count Sanity
# ═══════════════════════════════════════════════════════════

def test_v2_passengers():
    print("\n" + "=" * 70)
    print("  V2: Passenger Count Sanity")
    print("=" * 70)

    all_station_pax = defaultdict(list)  # station_name -> [pax_onboard per event]

    for run_idx in range(1, args.num_runs + 1):
        print(f"  Run {run_idx}/{args.num_runs}...", end=" ", flush=True)
        env = make_env()
        t0 = time.time()
        events = run_sim_episode(env, action_value=0.0)
        elapsed = time.time() - t0
        print(f"{len(events)} events, {elapsed:.1f}s")

        for ev in events:
            all_station_pax[ev['station_name']].append(ev['pax_onboard'])

    # Write CSV
    out_path = os.path.join(OUT_DIR, 'v2_passengers.csv')
    rows = []
    for sname in sorted(all_station_pax.keys()):
        vals = all_station_pax[sname]
        rows.append({
            'station': sname,
            'count': len(vals),
            'mean_pax': np.mean(vals),
            'std_pax': np.std(vals),
            'cv': np.std(vals) / (np.mean(vals) + 1e-6),
            'max_pax': max(vals),
        })
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n  ── Passenger Stats ──")
    for r in rows:
        flag = "✓" if r['cv'] < 0.5 else "⚠️ high CV"
        print(f"  {r['station']:15s}: mean={r['mean_pax']:5.1f}  std={r['std_pax']:5.1f}  CV={r['cv']:.2f}  {flag}")
    print(f"  Wrote: {out_path}")


# ═══════════════════════════════════════════════════════════
#  V3: Per-Station Travel Times
# ═══════════════════════════════════════════════════════════

def test_v3_travel_times():
    print("\n" + "=" * 70)
    print("  V3: Per-Station Travel Times")
    print("=" * 70)

    env = make_env()
    env.reset()
    actions = {k: 0.0 for k in range(env.max_agent_num)}

    # Track arrival times per bus
    arrival_log = defaultdict(list)  # bus_id -> [(station_id, station_name, time)]

    while sum(1 for v in env.state.values() if v) == 0:
        env.step(actions)

    while not env.done:
        env.step(actions)
        for bus in env.state_bus_list:
            if bus.obs:
                arrival_log[bus.bus_id].append((
                    bus.last_station.station_id,
                    bus.last_station.station_name,
                    env.current_time
                ))

    # Compute segment travel times
    segment_times = defaultdict(list)  # (from_station, to_station) -> [travel_time_seconds]
    for bus_id, arrivals in arrival_log.items():
        for i in range(1, len(arrivals)):
            prev_sid, prev_name, prev_t = arrivals[i - 1]
            curr_sid, curr_name, curr_t = arrivals[i]
            if curr_sid != prev_sid:  # actual station change
                dt = curr_t - prev_t
                if 0 < dt < 3600:  # sanity bound
                    key = f"{prev_name}→{curr_name}"
                    segment_times[key].append(dt)

    # Write CSV
    out_path = os.path.join(OUT_DIR, 'v3_travel_times.csv')
    rows = []
    for seg in sorted(segment_times.keys()):
        vals = segment_times[seg]
        rows.append({
            'segment': seg,
            'count': len(vals),
            'mean_sec': np.mean(vals),
            'std_sec': np.std(vals),
            'min_sec': min(vals),
            'max_sec': max(vals),
        })

    if rows:
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"\n  ── Segment Travel Times ──")
        for r in rows:
            print(f"  {r['segment']:25s}: mean={r['mean_sec']:6.1f}s  std={r['std_sec']:5.1f}  "
                  f"[{r['min_sec']:.0f}, {r['max_sec']:.0f}]  n={r['count']}")
        print(f"  Wrote: {out_path}")
    else:
        print("  ⚠️ No segment travel times recorded!")


# ═══════════════════════════════════════════════════════════
#  V4: Performance Benchmark
# ═══════════════════════════════════════════════════════════

def test_v4_benchmark():
    print("\n" + "=" * 70)
    print("  V4: Performance Benchmark (SimpleSim)")
    print("=" * 70)

    timings = []
    for run_idx in range(1, args.num_runs + 1):
        env = make_env()
        env.reset()
        actions = {k: 0.0 for k in range(env.max_agent_num)}

        t0 = time.time()
        step_count = 0
        while not env.done:
            env.step(actions)
            step_count += 1
        elapsed = time.time() - t0

        timings.append({
            'run': run_idx,
            'steps': step_count,
            'wall_sec': elapsed,
            'sim_time_sec': env.current_time,
            'speedup': env.current_time / elapsed if elapsed > 0 else 0,
        })
        print(f"  Run {run_idx}: {step_count:,} steps, sim_time={env.current_time}s, "
              f"wall={elapsed:.2f}s, speedup={timings[-1]['speedup']:.0f}x")

    out_path = os.path.join(OUT_DIR, 'v4_benchmark.csv')
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=timings[0].keys())
        w.writeheader()
        w.writerows(timings)

    avg_wall = np.mean([t['wall_sec'] for t in timings])
    avg_speedup = np.mean([t['speedup'] for t in timings])
    print(f"\n  Average: wall={avg_wall:.2f}s, speedup={avg_speedup:.0f}x realtime")
    print(f"  Wrote: {out_path}")


# ═══════════════════════════════════════════════════════════
#  V5: cProfile Analysis
# ═══════════════════════════════════════════════════════════

def test_v5_cprofile():
    print("\n" + "=" * 70)
    print("  V5: cProfile Analysis")
    print("=" * 70)

    env = make_env()
    env.reset()
    actions = {k: 0.0 for k in range(env.max_agent_num)}

    profiler = cProfile.Profile()
    profiler.enable()

    while not env.done:
        env.step(actions)

    profiler.disable()

    # Write top 30 to file
    out_path = os.path.join(OUT_DIR, 'v5_profile.txt')
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    profile_text = s.getvalue()

    with open(out_path, 'w') as f:
        f.write(profile_text)

    # Print top 15 to console
    s2 = io.StringIO()
    ps2 = pstats.Stats(profiler, stream=s2).sort_stats('cumulative')
    ps2.print_stats(15)
    print(s2.getvalue())
    print(f"  Full profile: {out_path}")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 70)
    print("  Phase 0-3 Verification Script")
    print(f"  LSTM-RL env: {os.path.join(LSTM_RL_DIR, 'env')}")
    print(f"  Output dir:  {OUT_DIR}")
    print(f"  skip_sumo:   {args.skip_sumo}")
    print("=" * 70)

    test_v1_w_decay()
    test_v2_passengers()
    test_v3_travel_times()
    test_v4_benchmark()
    test_v5_cprofile()

    print("\n" + "=" * 70)
    print("  ALL MODULES COMPLETE")
    print("=" * 70)
