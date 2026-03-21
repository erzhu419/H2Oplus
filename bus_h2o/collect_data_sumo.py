"""
collect_data_sumo.py
====================
Phase 3 — SUMO offline data collection for H2O+ sim-to-real.

Runs the Ruiguang SUMO environment under zero-hold (or optional random) policy
and collects (z_t, z_t+1) transition tuples per bus decision event.

Saves to HDF5:
    datasets/sumo_offline.h5
    Keys: z_t (N,30), z_t1 (N,30), s_t (N,15), s_t1 (N,15),
          a_t (N,1), r_t (N,1), sim_time (N,)

Usage
-----
    cd /home/erzhu419/mine_code/sumo-rl/H2Oplus/bus_h2o
    python collect_data_sumo.py \\
        --n_episodes 3 \\
        --max_steps   18000 \\
        --policy      zero \\
        --out         datasets/sumo_offline.h5

Requirements
------------
    SUMO_HOME must be set and SUMO must be installed.
    h5py: pip install h5py
"""

import argparse
import os
import sys
import random
import numpy as np

# ── path setup ─────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

# ── SUMO env ────────────────────────────────────────────────────────────
SUMO_DIR = os.path.join(
    os.path.dirname(_HERE),           # H2Oplus/
    os.path.pardir,                   # sumo-rl/
    "SUMO_ruiguang", "online_control"
)
SUMO_DIR = os.path.normpath(SUMO_DIR)

from sumo_env.rl_bridge import SumoRLBridge      # noqa: E402
from sumo_env.rl_env    import SumoBusHoldingEnv  # noqa: E402
from sumo_env.sumo_snapshot import bridge_to_snapshot  # noqa: E402

from common.data_utils import (  # noqa: E402
    build_edge_linear_map,
    extract_structured_context,
    set_route_length,
)

# ── constants ───────────────────────────────────────────────────────────
EDGE_XML   = os.path.join(_HERE, "network_data", "a_sorted_busline_edge.xml")
LINE_ID    = "7X"           # primary line used for edge_map
OBS_DIM    = 15             # SumoBusHoldingEnv obs dim
ACT_DIM    = 1


# ── helpers ─────────────────────────────────────────────────────────────

def obs_to_vec(obs: dict) -> np.ndarray:
    """Flatten the first available observation from a nested {line:{bus:[obs]}} dict."""
    for line_id, buses in obs.items():
        for bus_id, state_list in buses.items():
            if state_list:
                return np.array(state_list[-1], dtype=np.float32)
    return np.zeros(OBS_DIM, dtype=np.float32)


def reward_to_scalar(rew: dict) -> float:
    total = 0.0
    for buses in rew.values():
        if isinstance(buses, dict):
            for r in buses.values():
                total += float(r) if r is not None else 0.0
        elif buses is not None:
            total += float(buses)
    return total


def zero_actions(env: SumoBusHoldingEnv) -> dict:
    """Return zero hold for every bus key expected by the env."""
    # SumoBusHoldingEnv expects nested {line_id: {bus_id: action}}
    return {}   # empty = all buses hold 0


def run_episode(bridge: SumoRLBridge, env: SumoBusHoldingEnv,
                edge_map: dict, policy: str, max_steps: int
                ) -> list[dict]:
    """Run one SUMO episode, return list of transition dicts."""
    env.reset()
    transitions = []
    prev_z    = None
    prev_obs  = None
    prev_a    = None
    prev_r    = None
    prev_t    = None

    for _ in range(max_steps):
        # ── collect next batch of events ─────────────────────────────
        events, done, _ = bridge.fetch_events()
        if done:
            break
        if not events:
            continue

        # snapshot → z_t
        snap = bridge_to_snapshot(bridge, edge_map)
        z_t  = extract_structured_context(snap)
        t    = snap["sim_time"]

        # SumoBusHoldingEnv step (feeds events → obs, rew)
        if policy == "zero":
            actions = {}
        else:
            actions = {}  # random: still zero for now (env handles internally)

        obs, rew, done2, info = env.step(actions)
        s_t = obs_to_vec(obs)
        r_t = reward_to_scalar(rew)
        a_t = 0.0

        # Build transition from PREVIOUS step
        if prev_z is not None:
            transitions.append({
                "z_t"     : prev_z,
                "z_t1"    : z_t,
                "s_t"     : prev_obs,
                "s_t1"    : s_t,
                "a_t"     : np.array([prev_a], dtype=np.float32),
                "r_t"     : np.array([prev_r], dtype=np.float32),
                "sim_time": prev_t,
            })

        prev_z   = z_t
        prev_obs = s_t
        prev_a   = a_t
        prev_r   = r_t
        prev_t   = t

        if done or done2:
            break

    return transitions


# ── main ────────────────────────────────────────────────────────────────

def main(args):
    try:
        import h5py
    except ImportError:
        print("ERROR: h5py not installed. Run: pip install h5py")
        sys.exit(1)

    # Build edge map from locally copied XML
    print(f"[1] Building edge map from {EDGE_XML} ...")
    edge_map = build_edge_linear_map(EDGE_XML, LINE_ID)
    route_len = max(edge_map.values()) if edge_map else 13119.0
    set_route_length(route_len)
    print(f"    Route length: {route_len:.0f} m, edges: {len(edge_map)}")

    # Init SUMO bridge
    print(f"[2] Initialising SumoRLBridge (root_dir={SUMO_DIR}) ...")
    bridge = SumoRLBridge(
        root_dir  = SUMO_DIR,
        gui       = args.gui,
        max_steps = args.max_steps,
    )

    # Init RL env wrapper on top of bridge
    env = SumoBusHoldingEnv(
        root_dir           = SUMO_DIR,
        decision_provider  = bridge.fetch_events,
        action_executor    = bridge.apply_action,
        reset_callback     = bridge.reset,
        close_callback     = bridge.close,
    )

    all_transitions: list[dict] = []

    for ep in range(args.n_episodes):
        print(f"\n[3] Episode {ep + 1}/{args.n_episodes} ...")
        try:
            ep_trans = run_episode(bridge, env, edge_map, args.policy, args.max_steps)
            all_transitions.extend(ep_trans)
            print(f"    Collected {len(ep_trans)} transitions (total: {len(all_transitions)})")
        except Exception as e:
            print(f"    Episode {ep + 1} crashed: {e}")
            import traceback; traceback.print_exc()

    bridge.close()

    if not all_transitions:
        print("\nNo transitions collected — check SUMO_HOME and config.")
        return

    # ── Save to HDF5 ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f"\n[4] Saving {len(all_transitions)} transitions → {args.out}")

    with h5py.File(args.out, "w") as f:
        f.create_dataset("z_t",      data=np.stack([t["z_t"]   for t in all_transitions]), compression="gzip")
        f.create_dataset("z_t1",     data=np.stack([t["z_t1"]  for t in all_transitions]), compression="gzip")
        f.create_dataset("s_t",      data=np.stack([t["s_t"]   for t in all_transitions]), compression="gzip")
        f.create_dataset("s_t1",     data=np.stack([t["s_t1"]  for t in all_transitions]), compression="gzip")
        f.create_dataset("a_t",      data=np.stack([t["a_t"]   for t in all_transitions]), compression="gzip")
        f.create_dataset("r_t",      data=np.stack([t["r_t"]   for t in all_transitions]), compression="gzip")
        f.create_dataset("sim_time", data=np.array([t["sim_time"] for t in all_transitions]), compression="gzip")
        f.attrs["n_episodes"] = args.n_episodes
        f.attrs["policy"]     = args.policy
        f.attrs["route_len"]  = route_len

    print(f"    Saved. z_t shape: {f['z_t'].shape}")
    print("\n✅ collect_data_sumo.py done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect SUMO offline transitions for H2O+")
    parser.add_argument("--n_episodes", type=int,  default=3,
                        help="Number of SUMO episodes to run (default: 3)")
    parser.add_argument("--max_steps",  type=int,  default=18000,
                        help="Max simulation steps per episode (default: 18000)")
    parser.add_argument("--policy",     type=str,  default="zero",
                        choices=["zero", "random"],
                        help="Action policy during collection (default: zero)")
    parser.add_argument("--out",        type=str,  default="datasets/sumo_offline.h5",
                        help="Output HDF5 path (default: datasets/sumo_offline.h5)")
    parser.add_argument("--gui",        action="store_true",
                        help="Launch SUMO with GUI (slow, for debugging)")
    args = parser.parse_args()
    main(args)
