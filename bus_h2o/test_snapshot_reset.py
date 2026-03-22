"""
test_snapshot_reset.py  (Approach 2)
====================================
SUMO snapshot at time T → reset MultiLineSimEnv → run sim forward →
measure z divergence and w decay over time.

Design
------
1.  Run SUMO to time T (e.g. after 100 decision events)
2.  Capture full-fidelity per-line snapshots via bridge_to_full_snapshot()
3.  Also capture z_SUMO at T for reference
4.  Continue running SUMO beyond T for K more events → collect z_SUMO[k]
5.  Reset MultiLineSimEnv from SUMO snapshot → run forward K steps → collect z_sim[k]
6.  Compare z_SUMO[k] vs z_sim[k] at matched time points
7.  Train ZOnlyDiscriminator, compute w

Expected:
    - z_sim[0] ≈ z_SUMO[T] (snapshot fidelity check)
    - z_sim diverges from z_SUMO over time → L2 grows, cos decays
    - w for sim transitions is lower than w for SUMO transitions

Usage:
    cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o
    python test_snapshot_reset.py [--T_events 80] [--K_events 100] [--plot]
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

SUMO_DIR = os.path.normpath(os.path.join(
    _HERE, os.pardir, os.pardir, "SUMO_ruiguang", "online_control"))
sys.path.insert(0, SUMO_DIR)
sys.path.insert(0, os.path.join(SUMO_DIR, "sim_obj"))
_CASE_DIR = os.path.join(_HERE, "sumo_env", "case")
if os.path.isdir(_CASE_DIR):
    sys.path.insert(0, _CASE_DIR)

from sumo_env.rl_bridge    import SumoRLBridge                     # noqa
from sumo_env.sumo_snapshot import (bridge_to_snapshot,             # noqa
                                    bridge_to_full_snapshot)
from envs.bus_sim_env import MultiLineSimEnv, BusSimEnv             # noqa
from common.data_utils import (build_edge_linear_map,               # noqa
                               extract_structured_context,
                               set_route_length,
                               ZOnlyDiscriminator,
                               compute_z_importance_weight)

EDGE_XML   = os.path.join(_HERE, "network_data", "a_sorted_busline_edge.xml")
LINE_ID    = "7X"
CALIB_PATH = os.path.join(_HERE, "calibrated_env")


# ── helpers ───────────────────────────────────────────────────────────────

def make_snapshot_sim(env: MultiLineSimEnv) -> dict:
    """Build z-compatible snapshot dict from sim."""
    all_buses, all_stations = [], []
    for lid, le in env.line_map.items():
        cum = 0.0; route_cum = [0.0]
        for r in le.routes:
            cum += r.distance; route_cum.append(cum)
        for bus in le.bus_all:
            if getattr(bus, "on_route", False):
                all_buses.append({
                    "pos": bus.absolute_distance,
                    "speed": getattr(bus, "current_speed", 5.0),
                    "load": len(getattr(bus, "passengers", [])),
                })
        for i, st in enumerate(le.stations):
            idx = min(i, len(route_cum) - 1)
            wp = getattr(st, "waiting_passengers", np.array([]))
            all_stations.append({
                "pos": route_cum[idx],
                "waiting_count": int(len(wp)) if hasattr(wp, "__len__") else 0,
            })
    return {"sim_time": env.current_time, "all_buses": all_buses,
            "all_stations": all_stations}


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 1e-8 and nb > 1e-8 else 1.0


# ══════════════════════════════════════════════════════════════════════════

def main(args):
    print("=" * 65)
    print("  Approach 2: SUMO Snapshot → Reset Sim → Measure Divergence")
    print("=" * 65, flush=True)

    # ── Setup ─────────────────────────────────────────────────────────
    edge_map = build_edge_linear_map(EDGE_XML, LINE_ID) if os.path.exists(EDGE_XML) else {}
    route_len = max(edge_map.values()) if edge_map else 13119.0
    set_route_length(route_len)
    print(f"\n[0] Route length: {route_len:.0f} m", flush=True)

    # ── Phase 1: Run SUMO to T, capture full snapshot, then K more ────
    print(f"\n[1] Running SUMO: warm-up {args.T_events} events, "
          f"then {args.K_events} post-T events ...", flush=True)

    bridge = SumoRLBridge(root_dir=SUMO_DIR, gui=False, max_steps=25000)
    bridge.reset()

    # Warm-up to T
    event_count = 0
    for _ in range(args.T_events * 10):  # max iterations safety
        events, done, _ = bridge.fetch_events()
        if done:
            break
        if not events:
            continue
        for ev in events:
            bridge.apply_action(ev, 0.0)
        event_count += 1
        if event_count >= args.T_events:
            break

    T = bridge.current_time
    print(f"    Warm-up done: t={T:.0f}s, {event_count} events", flush=True)

    # Capture FULL snapshot at T (per-line)
    full_snap = bridge_to_full_snapshot(bridge, edge_map)
    print(f"    Full snapshot captured: {len(full_snap)} lines: "
          f"{list(full_snap.keys())}", flush=True)

    # Also capture z at T
    lite_snap_T = bridge_to_snapshot(bridge, edge_map)
    z_sumo_T = extract_structured_context(lite_snap_T)
    print(f"    z_SUMO at T: norm={np.linalg.norm(z_sumo_T):.3f}", flush=True)

    # Continue SUMO for K more events, collecting z
    z_sumo_post = [(T, z_sumo_T.copy())]
    for _ in range(args.K_events * 10):
        events, done, _ = bridge.fetch_events()
        if done:
            break
        if not events:
            continue
        for ev in events:
            bridge.apply_action(ev, 0.0)
        snap = bridge_to_snapshot(bridge, edge_map)
        z = extract_structured_context(snap)
        z_sumo_post.append((snap["sim_time"], z.copy()))
        if len(z_sumo_post) - 1 >= args.K_events:
            break

    bridge.close()
    print(f"    SUMO post-T: {len(z_sumo_post)} z-samples "
          f"(t={z_sumo_post[0][0]:.0f}→{z_sumo_post[-1][0]:.0f})", flush=True)

    # ── Phase 2: Reset sim from SUMO snapshot ─────────────────────────
    print(f"\n[2] Resetting MultiLineSimEnv from SUMO snapshot at T={T:.0f}s ...",
          flush=True)

    sim_env = MultiLineSimEnv(CALIB_PATH)
    sim_env.reset()

    # Apply per-line snapshots
    restored_lines = 0
    for line_id, le in sim_env.line_map.items():
        if line_id in full_snap:
            try:
                le.restore_full_system_snapshot(full_snap[line_id])
                restored_lines += 1
            except Exception as ex:
                print(f"    ⚠️  Failed to restore {line_id}: {ex}", flush=True)
    print(f"    Restored {restored_lines}/{len(sim_env.line_map)} lines", flush=True)

    # Verify z immediately after reset
    snap_sim_0 = make_snapshot_sim(sim_env)
    z_sim_0 = extract_structured_context(snap_sim_0)
    l2_reset = float(np.linalg.norm(z_sumo_T - z_sim_0))
    cos_reset = cosine_sim(z_sumo_T, z_sim_0)
    print(f"    z fidelity after reset: L2={l2_reset:.4f}, cos={cos_reset:.4f}", flush=True)

    # ── Phase 3: Run sim forward, collect z at matching times ─────────
    print(f"\n[3] Running sim forward from T={T:.0f}s ...", flush=True)

    actions = {lid: {k: 0.0 for k in range(le.max_agent_num)}
               for lid, le in sim_env.line_map.items()}

    sumo_post_times = [t for t, _ in z_sumo_post[1:]]  # exclude T itself
    z_sim_post = [(float(sim_env.current_time), z_sim_0.copy())]
    t_idx = 0
    done = False

    while not done and t_idx < len(sumo_post_times):
        state, reward, done = sim_env.step(actions)
        while t_idx < len(sumo_post_times) and sim_env.current_time >= sumo_post_times[t_idx]:
            snap = make_snapshot_sim(sim_env)
            z = extract_structured_context(snap)
            z_sim_post.append((float(sim_env.current_time), z.copy()))
            t_idx += 1

    print(f"    Sim post-T: {len(z_sim_post)} z-samples", flush=True)

    # ── Phase 4: Compare and train discriminator ──────────────────────
    N = min(len(z_sumo_post), len(z_sim_post))
    if N < 4:
        print(f"\n⚠️  Only {N} samples — not enough.", flush=True)
        return
    print(f"\n[4] Comparing {N} matched z-samples ...", flush=True)

    l2_list, cos_list, t_list = [], [], []
    for i in range(N):
        t_s, z_s = z_sumo_post[i]
        t_m, z_m = z_sim_post[i]
        l2 = float(np.linalg.norm(z_s - z_m))
        cos = cosine_sim(z_s, z_m)
        l2_list.append(l2)
        cos_list.append(cos)
        t_list.append(t_s)
        if i < 5 or i % 30 == 0 or i == N - 1:
            print(f"    [{i:4d}] t_sumo={t_s:8.0f}  t_sim={t_m:8.0f}  "
                  f"L2={l2:.4f}  cos={cos:.4f}", flush=True)

    # Transition pairs
    z_sumo_pairs = [(z_sumo_post[i][1], z_sumo_post[i+1][1]) for i in range(N-1)]
    z_sim_pairs  = [(z_sim_post[i][1],  z_sim_post[i+1][1])  for i in range(N-1)]

    if len(z_sumo_pairs) < 4:
        print("Not enough pairs for discriminator.", flush=True)
        return

    # Train discriminator
    print(f"\n[5] Training ZOnlyDiscriminator on {len(z_sumo_pairs)} pairs ...", flush=True)
    D   = ZOnlyDiscriminator(context_dim=30)
    opt = torch.optim.Adam(D.parameters(), lr=3e-4)
    crit = torch.nn.BCEWithLogitsLoss()

    zt_r  = torch.tensor(np.stack([p[0] for p in z_sumo_pairs]), dtype=torch.float32)
    zt1_r = torch.tensor(np.stack([p[1] for p in z_sumo_pairs]), dtype=torch.float32)
    zt_s  = torch.tensor(np.stack([p[0] for p in z_sim_pairs]),  dtype=torch.float32)
    zt1_s = torch.tensor(np.stack([p[1] for p in z_sim_pairs]),  dtype=torch.float32)

    for ep in range(200):
        loss = (crit(D(zt_r, zt1_r), torch.full((len(z_sumo_pairs), 1), 0.9)) +
                crit(D(zt_s, zt1_s), torch.full((len(z_sim_pairs),  1), 0.1)))
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 50 == 0:
            print(f"    ep {ep:3d}  loss={loss.item():.4f}", flush=True)

    # Compute w
    print(f"\n[6] Computing importance weights w ...", flush=True)
    w_sumo, w_sim = [], []
    for i in range(len(z_sumo_pairs)):
        zt  = torch.tensor(z_sumo_pairs[i][0], dtype=torch.float32).unsqueeze(0)
        zt1 = torch.tensor(z_sumo_pairs[i][1], dtype=torch.float32).unsqueeze(0)
        w_sumo.append(float(compute_z_importance_weight(D, zt, zt1).squeeze()))

        zt  = torch.tensor(z_sim_pairs[i][0], dtype=torch.float32).unsqueeze(0)
        zt1 = torch.tensor(z_sim_pairs[i][1], dtype=torch.float32).unsqueeze(0)
        w_sim.append(float(compute_z_importance_weight(D, zt, zt1).squeeze()))

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Reset at T               : {T:.0f}s")
    print(f"  z fidelity at reset      : L2={l2_reset:.4f}  cos={cos_reset:.4f}")
    print(f"  Matched z-samples        : {N}")
    print(f"  L2 [at reset]            : {l2_list[0]:.4f}")
    print(f"  L2 [end]                 : {np.mean(l2_list[-5:]):.4f}")
    print(f"  cos [at reset]           : {cos_list[0]:.4f}")
    print(f"  cos [end]                : {np.mean(cos_list[-5:]):.4f}")
    print(f"  w_SUMO [mean]            : {np.mean(w_sumo):.4f}")
    print(f"  w_sim  [mean]            : {np.mean(w_sim):.4f}")

    ok_l2  = np.mean(l2_list[-5:]) > l2_list[0]
    ok_cos = np.mean(cos_list[-5:]) <= cos_list[0]
    ok_w   = np.mean(w_sumo) > np.mean(w_sim)

    print(f"\n  L2 grows from reset?     : {'PASS' if ok_l2 else 'FAIL'}")
    print(f"  cos decays from reset?   : {'PASS' if ok_cos else 'FAIL'}")
    print(f"  w_SUMO > w_sim?          : {'PASS' if ok_w else 'FAIL'}")

    if ok_w:
        print("\n✅  Snapshot-reset divergence test PASSED")
    else:
        print("\n⚠️  Check results above")

    # ── Plot ──────────────────────────────────────────────────────────
    if args.plot:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        axes[0].axvline(T, color='red', ls='--', lw=1, label=f'Reset T={T:.0f}s')
        axes[0].plot(t_list, l2_list, "b-", lw=1.5, alpha=0.7)
        axes[0].set_ylabel("L2 ||z_SUMO - z_sim||")
        axes[0].set_title("z Divergence after snapshot reset")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].axvline(T, color='red', ls='--', lw=1, label=f'Reset T={T:.0f}s')
        axes[1].plot(t_list, cos_list, "g-", lw=1.5, alpha=0.7)
        axes[1].set_ylabel("Cosine similarity")
        axes[1].set_ylim([0, 1.05]); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        pair_t = t_list[:-1]
        axes[2].plot(pair_t, w_sumo, "r-", lw=1.5, alpha=0.7, label="w (SUMO)")
        axes[2].plot(pair_t, w_sim,  "b-", lw=1.5, alpha=0.7, label="w (sim)")
        axes[2].set_ylabel("Importance weight w")
        axes[2].set_xlabel("Sim time (s)")
        axes[2].set_title("w after snapshot reset at T")
        axes[2].legend(); axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(_HERE, "snapshot_reset_divergence.png")
        plt.savefig(out, dpi=120)
        print(f"\n  Plot → {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Approach 2: SUMO snapshot → reset sim → divergence test")
    parser.add_argument("--T_events", type=int, default=80,
                        help="SUMO events before snapshot (default: 80)")
    parser.add_argument("--K_events", type=int, default=100,
                        help="Events after snapshot for comparison (default: 100)")
    parser.add_argument("--plot", action="store_true",
                        help="Save snapshot_reset_divergence.png")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"\n!!! FATAL: {e}", flush=True)
        import traceback; traceback.print_exc()
