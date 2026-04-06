"""
eval_data_policies_on_sumo.py
=============================
Evaluate the 5 data-collection policies on SUMO to verify their actual reward
and compare with the reward stored in the offline dataset.

Policies: zero, random, heuristic_best, heuristic_weak, sac (ep39)
"""

import os, sys, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_H2O_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Reuse eval_sumo_checkpoint.py infrastructure
sys.path.insert(0, os.path.join(_H2O_ROOT, "collect_policy"))
sys.path.insert(0, os.path.join(_H2O_ROOT, "bus_h2o"))

SUMO_DIR = os.path.normpath(os.path.join(
    _H2O_ROOT, "bus_h2o", os.pardir, os.pardir, "SUMO_ruiguang", "online_control"))
sys.path.insert(0, SUMO_DIR)
sys.path.insert(0, os.path.join(SUMO_DIR, "sim_obj"))

from collect_worker import (
    make_policy_fn, event_to_obs, compute_reward,
    _reset_indices, _line_headway, _map_raw_to_env,
    load_sac_policy, _SUMO_LINE_INDEX, _SUMO_BUS_INDEX,
)
from sumo_env.rl_bridge import SumoRLBridge
from common.data_utils import build_edge_linear_map, set_route_length
import xml.etree.ElementTree as ET

EDGE_XML = os.path.join(_H2O_ROOT, "bus_h2o", "network_data", "a_sorted_busline_edge.xml")


def run_episode(bridge, policy_fn, needs_obs):
    _reset_indices()
    bridge.reset()
    _line_headway.update(bridge.line_headways)

    pending = {}
    last_action_for_obs = {}  # for obs augmentation in sac policy
    cum_reward = 0.0
    n_dec = 0
    t0 = time.time()

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
            obs = event_to_obs(ev) if needs_obs else None
            si = int(event_to_obs(ev)[2]) if not needs_obs else int(obs[2])
            rew = compute_reward(ev)

            if bid in pending:
                prev = pending.pop(bid)
                if si != prev["si"]:
                    cum_reward += rew
                    n_dec += 1

            raw_action = policy_fn(ev, obs)

            hold, speed = _map_raw_to_env(raw_action)
            bridge.apply_action(ev, [hold, speed])

            pending[bid] = {"si": si}

    return cum_reward, n_dec, time.time() - t0


def main():
    print("=" * 70)
    print("SUMO Evaluation: All 5 Data-Collection Policies")
    print("=" * 70)

    if os.path.exists(EDGE_XML):
        em = build_edge_linear_map(EDGE_XML, "7X")
        set_route_length(max(em.values()) if em else 13119.0)

    bridge = SumoRLBridge(root_dir=SUMO_DIR, gui=False, max_steps=18000)

    rng = np.random.RandomState(42)
    results = {}

    for policy_name in ["zero", "heuristic_best", "heuristic_weak", "random", "sac"]:
        print(f"\nEvaluating: {policy_name}...")
        policy_fn, needs_obs = make_policy_fn(policy_name, rng)
        r, n, t = run_episode(bridge, policy_fn, needs_obs)
        per_step = r / max(n, 1)
        print(f"  {policy_name}: reward={r:,.0f}, decisions={n}, per_step={per_step:.1f}, time={t:.0f}s")
        results[policy_name] = (r, n, per_step)

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Policy':>18s}  {'Total Reward':>14s}  {'Decisions':>10s}  {'Per-Step':>10s}")
    for name, (r, n, ps) in sorted(results.items(), key=lambda x: x[1][0], reverse=True):
        marker = " ← best" if r == max(v[0] for v in results.values()) else ""
        print(f"{name:>18s}  {r:>14,.0f}  {n:>10d}  {ps:>10.1f}{marker}")
    print(f"{'='*70}")

    bridge.close()


if __name__ == "__main__":
    main()
