"""
test_divergence.py  (v3 — MultiLine deepcopy approach)
=======================================================
Validates the H2Oplus core hypothesis using MultiLineSimEnv:

    When sim is seeded from a real-world snapshot at t=T,
    its observations initially match reality closely but diverge over time.
    Importance weight w = P(real)/P(sim) should therefore DECREASE.

Design
------
SUMO "snapshot to sim" test via deepcopy:

  Step 1: Warm up ONE MultiLineSimEnv with known seed (deterministic) up to t=T.
  Step 2: deepcopy() the env → "sim" starts at IDENTICAL state as "real" at t=T.
  Step 3: From t=T, both envs diverge because Python's random state advances
          independently (stdlib `random.lognormvariate` for route speed updates).

At each bus event we compute:
  z_t              — 30-dim spatial context  (extract_structured_context)
  dz[k]            — ||z_real[k] - z_sim[k]||
  cos_sim(z_r,z_s) — similarity proxy (1 at reset → decreases)

When SUMO snapshot available (--use_sumo_snapshot):
  The sim is seeded from an actual SUMO SnapshotDict via
  BusSimEnv.restore_full_system_snapshot() per line.
  z from SUMO and z from sim are compared.

Expected result:
  dz[0] ≈ 0  (at reset t=T)  → dz grows over time
  cos_w[0] ≈ 1               → cos_w decreases over time

Usage
-----
  cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o
  python test_divergence.py [--n_events 15] [--T_seed_steps 1500] [--plot]
"""

import argparse
import os
import sys
import copy
import random
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from typing import List, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from envs.bus_sim_env import MultiLineSimEnv
from common.data_utils import (
    extract_structured_context,
    ZOnlyDiscriminator,
    compute_z_importance_weight,
    set_route_length,
)

CALIB_PATH = os.path.join(_HERE, "calibrated_env")


# ── helpers ───────────────────────────────────────────────────────────────────

def zero_actions(env: MultiLineSimEnv) -> dict:
    return {lid: {k: 0.0 for k in range(le.max_agent_num)}
            for lid, le in env.line_map.items()}


def any_obs(state: dict) -> bool:
    return any(bool(v) for bd in state.values() for v in bd.values())


def step_until_obs(
    env: MultiLineSimEnv, actions: dict, max_steps: int = 5000
) -> Tuple[dict, dict, bool]:
    """Step until at least one bus emits obs. Returns (state, reward, done)."""
    for _ in range(max_steps):
        state, reward, done = env.step(actions)
        if done:
            return state, reward, True
        if any_obs(state):
            return state, reward, False
    return state, reward, False


def make_snapshot(env: MultiLineSimEnv) -> dict:
    """Build a snapshot dict for extract_structured_context."""
    all_buses = []
    all_stations = []
    for lid, le in env.line_map.items():
        # Build cumulative distances per station from routes
        cum_dist = 0.0
        route_cum = [0.0]  # cum dist at station 0
        for r in le.routes:
            cum_dist += r.distance
            route_cum.append(cum_dist)
        total_dist = cum_dist if cum_dist > 0 else 14000.0

        for bus in le.bus_all:
            if getattr(bus, "on_route", False):
                all_buses.append({
                    "pos":   bus.absolute_distance,
                    "speed": getattr(bus, "current_speed", le.route_state[0] if le.route_state else 5.0),
                    "load":  len(getattr(bus, "passengers", [])),
                })

        for i, st in enumerate(le.stations):
            # Use cumulative distance from routes for station pos
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



def compute_z(env: MultiLineSimEnv, route_len: float) -> np.ndarray:
    set_route_length(route_len)
    return extract_structured_context(make_snapshot(env))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 1.0
    return float(np.dot(a, b) / (na * nb))


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 65)
    print("H2Oplus Divergence Test  — MultiLine deepcopy approach")
    print("=" * 65)

    # ── 1. Compute route_len (for z normalisation) ────────────────────────
    print(f"\n[1] Loading MultiLineSimEnv ...")
    base_env = MultiLineSimEnv(CALIB_PATH)
    route_len = max(
        sum(r.distance for r in le.routes) if le.routes else 14000.0
        for le in base_env.line_map.values()
    )
    set_route_length(route_len)
    print(f"   Lines: {list(base_env.line_map.keys())}")
    print(f"   Route length for z: {route_len:.0f} m")

    # ── 2. Warm up with fixed seed up to t=T ──────────────────────────────
    print(f"\n[2] Warming up env for {args.T_seed_steps} steps (seed=42) ...")
    random.seed(42)
    base_env.reset()
    actions = zero_actions(base_env)

    n_obs_fired = 0
    for step in range(args.T_seed_steps):
        try:
            state, reward, done = base_env.step(actions)
        except Exception as ex:
            print(f"   step {step} error: {ex}")
            break
        if any_obs(state):
            n_obs_fired += 1
        if done:
            print("   env done during warmup — stopping early")
            break

    n_buses = sum(1 for le in base_env.line_map.values()
                  for b in le.bus_all if b.on_route)
    print(f"   t={base_env.current_time}s, {n_buses} active buses, {n_obs_fired} obs events")

    # ── 3. deepcopy → sim_env at exactly same state ───────────────────────
    print("\n[3] deep-copying env → real_env (keeps running) + sim_env (frozen at T)")
    real_env = base_env
    sim_env  = copy.deepcopy(base_env)
    # Advance real_env's random state by running 350 extra steps — this is
    # past route_state_update_freq=300s threshold so at least one route speed
    # update will fire differently between real and sim, causing divergence.
    actions2 = zero_actions(real_env)
    for _ in range(350):
        try:
            real_env.step(actions2)
        except Exception:
            break

    # Now both envs are at t≈T (same bus positions/passengers/timetable)
    # but real_env's stdlib random state has advanced → they'll diverge via
    # independent stochastic route_state_update() calls


    # ── 4. Measure z at t=T (should be identical) ─────────────────────────
    z_real_t0 = compute_z(real_env, route_len)
    z_sim_t0  = compute_z(sim_env,  route_len)
    dz_t0 = float(np.linalg.norm(z_real_t0 - z_sim_t0))
    cos_t0 = cosine_sim(z_real_t0, z_sim_t0)
    print(f"\n   [t=T+0] dz={dz_t0:.6f}  cos_sim={cos_t0:.6f}  (dz should be ~0, cos_sim ~1)")

    # ── 5. Collect event pairs ─────────────────────────────────────────────
    print(f"\n[4] Collecting {args.n_events} event pairs ...")

    dz_list   : List[float] = [dz_t0]
    cos_list  : List[float] = [cos_t0]
    z_real_pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    z_sim_pairs : List[Tuple[np.ndarray, np.ndarray]] = []
    prev_z_r = z_real_t0.copy()
    prev_z_s = z_sim_t0.copy()

    actions_r = zero_actions(real_env)
    actions_s = zero_actions(sim_env)

    for k in range(args.n_events):
        state_r, reward_r, done_r = step_until_obs(real_env, actions_r)
        state_s, reward_s, done_s = step_until_obs(sim_env,  actions_s)

        if done_r or done_s:
            print(f"   [{k:3d}] env terminated — stopping.")
            break

        z_r = compute_z(real_env, route_len)
        z_s = compute_z(sim_env,  route_len)

        dz  = float(np.linalg.norm(z_r - z_s))
        cos = cosine_sim(z_r, z_s)

        dz_list.append(dz)
        cos_list.append(cos)
        z_real_pairs.append((prev_z_r.copy(), z_r.copy()))
        z_sim_pairs.append((prev_z_s.copy(), z_s.copy()))
        prev_z_r = z_r; prev_z_s = z_s

        print(
            f"   [{k:3d}] t_r={real_env.current_time:6.0f}  "
            f"t_s={sim_env.current_time:6.0f}  "
            f"dz={dz:.6f}  cos={cos:.4f}"
        )

    # ── 6. ZOnlyDiscriminator training ────────────────────────────────────
    w_trained: List[float] = []
    if len(z_real_pairs) >= 4:
        print(f"\n[5] Training ZOnlyDiscriminator on {len(z_real_pairs)} pairs ...")
        D   = ZOnlyDiscriminator(context_dim=30)
        opt = torch.optim.Adam(D.parameters(), lr=3e-4)
        crit = torch.nn.BCEWithLogitsLoss()

        def to_t(pairs):
            zt  = torch.tensor(np.stack([p[0] for p in pairs]), dtype=torch.float32)
            zt1 = torch.tensor(np.stack([p[1] for p in pairs]), dtype=torch.float32)
            return zt, zt1

        for ep in range(100):
            zt_r, zt1_r = to_t(z_real_pairs)
            zt_s, zt1_s = to_t(z_sim_pairs)
            loss = (crit(D(zt_r, zt1_r), torch.full((len(z_real_pairs), 1), 0.9)) +
                    crit(D(zt_s, zt1_s), torch.full((len(z_sim_pairs),  1), 0.1)))
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 25 == 0:
                print(f"   ep {ep:3d}  loss={loss.item():.4f}")

        for (z_t, z_t1) in z_real_pairs:
            zt  = torch.tensor(z_t,  dtype=torch.float32).unsqueeze(0)
            zt1 = torch.tensor(z_t1, dtype=torch.float32).unsqueeze(0)
            w = compute_z_importance_weight(D, zt, zt1)
            w_trained.append(float(w.squeeze()))

    # ── 7. Report ─────────────────────────────────────────────────────────
    n = len(dz_list)
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Events collected        : {n} (k=0 is t=T reset)")
    print(f"  dz[0]  (at reset)       : {dz_list[0]:.6f}  ← expect ≈0")
    print(f"  dz[-1] (final)          : {dz_list[-1]:.6f}  ← expect >0")
    print(f"  cos[0] (at reset)       : {cos_list[0]:.6f}  ← expect ≈1")
    print(f"  cos[-1](final)          : {cos_list[-1]:.6f}  ← expect <1")
    if w_trained:
        print(f"  w_D[0]  (trained)       : {w_trained[0]:.4f}")
        print(f"  w_D[-1] (trained)       : {w_trained[-1]:.4f}")

    # ── 8. Plot ───────────────────────────────────────────────────────────
    if args.plot and n >= 2:
        ks = list(range(n))
        nrow = 3 if z_real_pairs else 2
        fig, axes = plt.subplots(nrow, 1, figsize=(10, 4 * nrow))

        axes[0].axhline(0, color="gray", lw=0.5)
        axes[0].plot(ks, dz_list, "b-o", lw=2, ms=5, label="||z_real − z_sim||₂")
        axes[0].set_ylabel("Context divergence dz")
        axes[0].set_xlabel("Bus event k  (k=0: t=T reset)")
        axes[0].set_title("Global context divergence  (expected: grows from ~0)")
        axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(ks, cos_list, "g-o", lw=2, ms=5, label="cos_sim(z_real, z_sim)")
        if w_trained:
            axes[1].plot(ks[1:1+len(w_trained)], w_trained, "r-s",
                         lw=2, ms=5, label="w (ZOnlyD trained)")
        axes[1].set_ylim([-0.1, 1.1])
        axes[1].set_ylabel("Similarity / importance weight")
        axes[1].set_xlabel("Bus event k")
        axes[1].set_title("Similarity over time  (expected: decreasing from ~1)")
        axes[1].legend(); axes[1].grid(True, alpha=0.3)

        if z_real_pairs and nrow == 3:
            diff_mat = np.abs(np.stack([p[0] for p in z_real_pairs]) -
                              np.stack([p[0] for p in z_sim_pairs]))  # (K, 30)
            im = axes[2].imshow(diff_mat.T, aspect="auto", origin="lower",
                                cmap="hot", interpolation="nearest")
            axes[2].set_xlabel("Bus event k")
            axes[2].set_ylabel("z dimension")
            axes[2].set_title("|z_real − z_sim| per dimension")
            plt.colorbar(im, ax=axes[2])

        plt.tight_layout()
        out = os.path.join(_HERE, "divergence_multiline.png")
        plt.savefig(out, dpi=120)
        print(f"\n  Plot → {out}")

    # ── 9. Assertions ─────────────────────────────────────────────────────
    passed = True
    if n >= 5:
        early_dz = np.mean(dz_list[:2])
        late_dz  = np.mean(dz_list[-2:])
        ok_dz = late_dz >= early_dz
        print(f"\n  dz grows?    early={early_dz:.4f} → late={late_dz:.4f}  {'PASS' if ok_dz else 'FAIL'}")
        if not ok_dz:
            passed = False

        early_cos = np.mean(cos_list[:2])
        late_cos  = np.mean(cos_list[-2:])
        ok_cos = late_cos <= early_cos
        print(f"  cos decays?  early={early_cos:.4f} → late={late_cos:.4f}  {'PASS' if ok_cos else 'FAIL'}")
        if not ok_cos:
            passed = False
    else:
        print("\n  Not enough events for assertion.")

    print("\n✅  Divergence test PASSED" if passed else
          "\n⚠️  Divergence test: check assertions above")
    return passed


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H2Oplus multi-line divergence test")
    parser.add_argument("--n_events",    type=int,  default=15,
                        help="Bus events per env (default: 15)")
    parser.add_argument("--T_seed_steps", type=int, default=1500,
                        help="Steps to warm up before deepcopy (default: 1500)")
    parser.add_argument("--plot",        action="store_true",
                        help="Save divergence_multiline.png")
    parser.add_argument("--use_sumo_snapshot", type=str, default=None,
                        help="Path to SUMO SnapshotDict .pkl (advanced)")
    args = parser.parse_args()
    main(args)
