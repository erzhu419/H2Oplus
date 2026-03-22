"""
test_snapshot_headtohead.py
===========================
Head-to-head comparison: SUMO (real) vs MultiLineSimEnv (sim).

Runs BOTH environments with zero-hold policy, extracts 30-dim z_t
(spatial fingerprint) at matched sim-time intervals, then:

  1. Measures L2 divergence and cosine similarity of z_SUMO vs z_sim
  2. Trains ZOnlyDiscriminator on paired (z_t, z_t+1) from both sources
  3. Computes importance weight w — expects w to be high initially (both
     generate similar transitions) and decrease over time as sim diverges

Expected result:
  - z_SUMO and z_sim start similar (same timetable, same routes)
  - They diverge because sim uses calibrated speeds while SUMO has traffic
  - Discriminator learns to distinguish → w decreases

Usage:
    cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o
    python test_snapshot_headtohead.py [--max_events 200] [--plot]
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

# ── SUMO env ──────────────────────────────────────────────────────────────
SUMO_DIR = os.path.normpath(os.path.join(
    _HERE, os.pardir, os.pardir, "SUMO_ruiguang", "online_control"))

# rl_bridge transitively imports sim_obj.passenger via case/f_8_create_obj
# — these modules need SUMO_DIR and its case/ subdirectory on sys.path.
sys.path.insert(0, SUMO_DIR)
sys.path.insert(0, os.path.join(SUMO_DIR, "sim_obj"))
_CASE_DIR = os.path.join(_HERE, "sumo_env", "case")
if os.path.isdir(_CASE_DIR):
    sys.path.insert(0, _CASE_DIR)

from sumo_env.rl_bridge    import SumoRLBridge        # noqa: E402
from sumo_env.sumo_snapshot import bridge_to_snapshot  # noqa: E402

from envs.bus_sim_env import MultiLineSimEnv           # noqa: E402
from common.data_utils import (                        # noqa: E402
    build_edge_linear_map,
    extract_structured_context,
    set_route_length,
    ZOnlyDiscriminator,
    compute_z_importance_weight,
)

# ── constants ─────────────────────────────────────────────────────────────
EDGE_XML  = os.path.join(_HERE, "network_data", "a_sorted_busline_edge.xml")
LINE_ID   = "7X"
CALIB_PATH = os.path.join(_HERE, "calibrated_env")

# ── helpers ───────────────────────────────────────────────────────────────

def make_snapshot_sim(env: MultiLineSimEnv) -> dict:
    """Build a snapshot dict from MultiLineSimEnv (same as test_divergence.py)."""
    all_buses = []
    all_stations = []
    for lid, le in env.line_map.items():
        cum_dist = 0.0
        route_cum = [0.0]
        for r in le.routes:
            cum_dist += r.distance
            route_cum.append(cum_dist)

        for bus in le.bus_all:
            if getattr(bus, "on_route", False):
                all_buses.append({
                    "pos":   bus.absolute_distance,
                    "speed": getattr(bus, "current_speed",
                                     le.route_state[0] if le.route_state else 5.0),
                    "load":  len(getattr(bus, "passengers", [])),
                })

        for i, st in enumerate(le.stations):
            idx = min(i, len(route_cum) - 1)
            pos = route_cum[idx]
            wp = getattr(st, "waiting_passengers", np.array([]))
            all_stations.append({
                "pos":           pos,
                "waiting_count": int(len(wp)) if hasattr(wp, "__len__") else 0,
            })

    return {
        "sim_time":     env.current_time,
        "all_buses":    all_buses,
        "all_stations": all_stations,
    }


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


# ══════════════════════════════════════════════════════════════════════════
#  Phase A: collect z from SUMO
# ══════════════════════════════════════════════════════════════════════════

def collect_sumo_z(edge_map: dict, max_events: int) -> list:
    """Run SUMO episode with zero-hold via bridge directly, return [(sim_time, z_30)]."""
    print("\n[A] Launching SUMO ...", flush=True)
    bridge = SumoRLBridge(root_dir=SUMO_DIR, gui=False, max_steps=25000)
    bridge.reset()
    print(f"    Bridge reset OK. t={bridge.current_time}", flush=True)

    results = []
    t0 = time.time()

    for k in range(max_events):
        try:
            events, done, _ = bridge.fetch_events()
        except Exception as ex:
            print(f"    fetch_events error at k={k}: {ex}", flush=True)
            import traceback; traceback.print_exc()
            break
        if done:
            print(f"    SUMO done at event {k}", flush=True)
            break
        if not events:
            continue

        # Apply zero hold to all pending events
        for ev in events:
            bridge.apply_action(ev, 0.0)

        # Capture snapshot → z
        try:
            snap = bridge_to_snapshot(bridge, edge_map)
            z    = extract_structured_context(snap)
            results.append((snap["sim_time"], z.copy()))
        except Exception as ex:
            print(f"    snapshot error at k={k}: {ex}", flush=True)
            import traceback; traceback.print_exc()
            break

        if k % 50 == 0:
            print(f"    [{k:4d}] t={snap['sim_time']:8.0f}  "
                  f"z_norm={np.linalg.norm(z):.3f}  "
                  f"buses={len(snap['all_buses'])}", flush=True)

    bridge.close()
    wall = time.time() - t0
    print(f"    SUMO collected {len(results)} z-samples in {wall:.1f}s", flush=True)
    return results


# ══════════════════════════════════════════════════════════════════════════
#  Phase B: collect z from Sim at matching times
# ══════════════════════════════════════════════════════════════════════════

def collect_sim_z(sumo_times: list, route_len: float) -> list:
    """Run sim, sample z at the closest times matching SUMO events."""
    print("\n[B] Running MultiLineSimEnv ...")
    set_route_length(route_len)
    env = MultiLineSimEnv(CALIB_PATH)
    env.reset()
    actions = {lid: {k: 0.0 for k in range(le.max_agent_num)}
               for lid, le in env.line_map.items()}

    # Target times we want to sample z at
    target_times = sorted(sumo_times)
    t_idx = 0
    results = []
    t0 = time.time()
    done = False

    while not done and t_idx < len(target_times):
        state, reward, done = env.step(actions)
        # Check if we passed a target time
        while t_idx < len(target_times) and env.current_time >= target_times[t_idx]:
            snap = make_snapshot_sim(env)
            z    = extract_structured_context(snap)
            results.append((float(env.current_time), z.copy()))
            t_idx += 1

    wall = time.time() - t0
    print(f"    Sim collected {len(results)} z-samples in {wall:.1f}s")
    return results


# ══════════════════════════════════════════════════════════════════════════
#  Phase C: compare z, train discriminator, compute w
# ══════════════════════════════════════════════════════════════════════════

def compare_and_train(z_sumo: list, z_sim: list, plot: bool):
    """Compare z trajectories, train discriminator, compute w."""
    N = min(len(z_sumo), len(z_sim))
    if N < 4:
        print(f"\n⚠️  Only {N} matched z-samples — not enough for comparison.")
        return

    print(f"\n[C] Comparing {N} matched z-samples ...")

    # ── 1. Raw z comparison ───────────────────────────────────────────────
    l2_list  = []
    cos_list = []
    t_list   = []
    for i in range(N):
        t_sumo, z_s = z_sumo[i]
        t_sim,  z_m = z_sim[i]
        l2  = float(np.linalg.norm(z_s - z_m))
        cos = cosine_sim(z_s, z_m)
        l2_list.append(l2)
        cos_list.append(cos)
        t_list.append(t_sumo)
        if i % 50 == 0 or i < 5 or i == N - 1:
            print(f"    [{i:4d}] t_sumo={t_sumo:8.0f}  t_sim={t_sim:8.0f}  "
                  f"L2={l2:.4f}  cos={cos:.4f}")

    # ── 2. Build transition pairs for discriminator ───────────────────────
    z_sumo_pairs = []  # (z_t, z_{t+1})
    z_sim_pairs  = []
    for i in range(N - 1):
        z_sumo_pairs.append((z_sumo[i][1], z_sumo[i + 1][1]))
        z_sim_pairs.append((z_sim[i][1],  z_sim[i + 1][1]))

    if len(z_sumo_pairs) < 4:
        print("    Not enough pairs for discriminator training.")
        return

    # ── 3. Train ZOnlyDiscriminator ───────────────────────────────────────
    print(f"\n[D] Training ZOnlyDiscriminator on {len(z_sumo_pairs)} pairs ...")
    D   = ZOnlyDiscriminator(context_dim=30)
    opt = torch.optim.Adam(D.parameters(), lr=3e-4)
    crit = torch.nn.BCEWithLogitsLoss()

    zt_r  = torch.tensor(np.stack([p[0] for p in z_sumo_pairs]), dtype=torch.float32)
    zt1_r = torch.tensor(np.stack([p[1] for p in z_sumo_pairs]), dtype=torch.float32)
    zt_s  = torch.tensor(np.stack([p[0] for p in z_sim_pairs]),  dtype=torch.float32)
    zt1_s = torch.tensor(np.stack([p[1] for p in z_sim_pairs]),  dtype=torch.float32)

    for ep in range(200):
        # Mini-batch (full batch if small enough)
        loss = (crit(D(zt_r, zt1_r), torch.full((len(z_sumo_pairs), 1), 0.9)) +
                crit(D(zt_s, zt1_s), torch.full((len(z_sim_pairs),  1), 0.1)))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep % 50 == 0:
            print(f"    ep {ep:3d}  loss={loss.item():.4f}")

    # ── 4. Compute w per transition pair ──────────────────────────────────
    print(f"\n[E] Computing importance weights w ...")
    w_sumo = []  # w for SUMO transitions (should be high)
    w_sim  = []  # w for sim transitions (should be lower)
    for i in range(len(z_sumo_pairs)):
        zt  = torch.tensor(z_sumo_pairs[i][0], dtype=torch.float32).unsqueeze(0)
        zt1 = torch.tensor(z_sumo_pairs[i][1], dtype=torch.float32).unsqueeze(0)
        w = compute_z_importance_weight(D, zt, zt1)
        w_sumo.append(float(w.squeeze()))

        zt  = torch.tensor(z_sim_pairs[i][0], dtype=torch.float32).unsqueeze(0)
        zt1 = torch.tensor(z_sim_pairs[i][1], dtype=torch.float32).unsqueeze(0)
        w = compute_z_importance_weight(D, zt, zt1)
        w_sim.append(float(w.squeeze()))

    # ── 5. Report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Matched z-samples       : {N}")
    print(f"  L2 [start]              : {np.mean(l2_list[:5]):.4f}")
    print(f"  L2 [end]                : {np.mean(l2_list[-5:]):.4f}")
    print(f"  cos [start]             : {np.mean(cos_list[:5]):.4f}")
    print(f"  cos [end]               : {np.mean(cos_list[-5:]):.4f}")
    print(f"  w_SUMO [mean]           : {np.mean(w_sumo):.4f}")
    print(f"  w_sim  [mean]           : {np.mean(w_sim):.4f}")
    print(f"  w_SUMO [first 10]       : {np.mean(w_sumo[:10]):.4f}")
    print(f"  w_sim  [first 10]       : {np.mean(w_sim[:10]):.4f}")
    if len(w_sim) > 20:
        print(f"  w_sim  [last 10]        : {np.mean(w_sim[-10:]):.4f}")

    # Assertions
    ok_l2 = np.mean(l2_list[-5:]) > np.mean(l2_list[:5])
    ok_cos = np.mean(cos_list[-5:]) <= np.mean(cos_list[:5])
    ok_w = np.mean(w_sumo) > np.mean(w_sim)
    print(f"\n  L2 grows?               : {'PASS' if ok_l2 else 'FAIL'}")
    print(f"  cos decays?             : {'PASS' if ok_cos else 'FAIL (may be OK if both similar)'}")
    print(f"  w_SUMO > w_sim?         : {'PASS' if ok_w else 'FAIL'}")

    if ok_w:
        print("\n✅  Head-to-head test PASSED: discriminator distinguishes SUMO from sim")
    else:
        print("\n⚠️  Check results above")

    # ── 6. Plot ───────────────────────────────────────────────────────────
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))

        # L2 divergence
        axes[0].plot(t_list, l2_list, "b-", lw=1.5, alpha=0.7)
        axes[0].set_ylabel("L2 ||z_SUMO - z_sim||")
        axes[0].set_xlabel("Sim time (s)")
        axes[0].set_title("z Divergence: SUMO vs Sim over episode")
        axes[0].grid(True, alpha=0.3)

        # Cosine similarity
        axes[1].plot(t_list, cos_list, "g-", lw=1.5, alpha=0.7)
        axes[1].set_ylabel("Cosine similarity")
        axes[1].set_xlabel("Sim time (s)")
        axes[1].set_title("Cosine similarity of z_SUMO vs z_sim")
        axes[1].set_ylim([0, 1.05])
        axes[1].grid(True, alpha=0.3)

        # Importance weights
        pair_t = t_list[:-1]
        axes[2].plot(pair_t, w_sumo, "r-", lw=1.5, alpha=0.7, label="w (SUMO trans)")
        axes[2].plot(pair_t, w_sim,  "b-", lw=1.5, alpha=0.7, label="w (sim trans)")
        axes[2].set_ylabel("Importance weight w")
        axes[2].set_xlabel("Sim time (s)")
        axes[2].set_title("Importance weights from trained ZOnlyDiscriminator")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        out = os.path.join(_HERE, "snapshot_headtohead.png")
        plt.savefig(out, dpi=120)
        print(f"\n  Plot → {out}")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

def main(args):
    print("=" * 65)
    print("  SUMO ↔ Sim Head-to-Head z-Comparison")
    print("=" * 65)

    # ── Setup ─────────────────────────────────────────────────────────────
    print(f"\n[0] Building edge map ...")
    if os.path.exists(EDGE_XML):
        edge_map = build_edge_linear_map(EDGE_XML, LINE_ID)
        route_len = max(edge_map.values()) if edge_map else 13119.0
    else:
        print(f"    ⚠️  {EDGE_XML} not found, using fallback route_len=14000")
        edge_map = {}
        route_len = 14000.0
    set_route_length(route_len)
    print(f"    Route length: {route_len:.0f} m")

    # ── Phase A: SUMO ─────────────────────────────────────────────────────
    z_sumo = collect_sumo_z(edge_map, args.max_events)
    if len(z_sumo) < 4:
        print("Not enough SUMO samples. Check SUMO_HOME and config.")
        return

    # ── Phase B: Sim ──────────────────────────────────────────────────────
    sumo_times = [t for t, _ in z_sumo]
    z_sim = collect_sim_z(sumo_times, route_len)

    # ── Phase C: Compare ──────────────────────────────────────────────────
    compare_and_train(z_sumo, z_sim, args.plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SUMO↔Sim head-to-head z-comparison test")
    parser.add_argument("--max_events", type=int, default=200,
                        help="Max SUMO events to collect (default: 200)")
    parser.add_argument("--plot", action="store_true",
                        help="Save snapshot_headtohead.png")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"\n\n!!! FATAL: {e}", flush=True)
        import traceback; traceback.print_exc()
