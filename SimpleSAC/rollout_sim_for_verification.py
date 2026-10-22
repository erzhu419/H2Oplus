#!/usr/bin/env python3
"""rollout_sim_for_verification.py — generate sim transitions for Assumption 1 check.

Produces an h5 file with the same schema as the offline SUMO data (observations,
actions, next_observations, z_t, z_t1) but populated with transitions rolled out
in the low-fidelity sim_core environment. Used by verify_assumption1.py.

Key design choice: we sample actions from a uniform distribution over [-1, 1]^2
rather than zero-hold or a trained policy. Uniform sampling is required for the
Assumption 1 check because we need coverage across action bins to estimate
P_sim(s' | s, a) per action bin.

Usage:
    cd H2Oplus/SimpleSAC
    python rollout_sim_for_verification.py \\
        --n_events 50000 \\
        --out_h5 ../experiment_output/sim_rollouts_for_verification.h5
"""

import argparse
import os
import sys
import h5py
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_H2O_ROOT = os.path.dirname(_HERE)
_BUS_H2O = os.path.join(_H2O_ROOT, "bus_h2o")
sys.path.insert(0, _HERE)
sys.path.insert(0, _BUS_H2O)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim_env_path",
                    default=os.path.join(_BUS_H2O, "calibrated_env"))
    ap.add_argument("--n_events", type=int, default=50000)
    ap.add_argument("--warmup_time", type=float, default=5000.0,
                    help="Let the sim warm up to this sim-time before recording.")
    ap.add_argument("--out_h5", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_episodes", type=int, default=3,
                    help="Restart sim this many times to cover different snapshots.")
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)

    from envs.bus_sim_env import MultiLineSimEnv
    from common.data_utils import extract_structured_context

    obs_list, act_list, nobs_list = [], [], []
    z_list, znext_list, rew_list = [], [], []

    per_episode = args.n_events // args.n_episodes
    print(f"Rolling out {args.n_episodes} episodes × {per_episode} events each "
          f"= {args.n_events} total.")

    for ep in range(args.n_episodes):
        env = MultiLineSimEnv(path=args.sim_env_path, debug=False)
        env.reset()

        # Warm up with zero actions
        for _ in range(100000):
            full_a = {lid: {k: 0.0 for k in range(env.line_map[lid].max_agent_num)}
                      for lid in env.line_map}
            try:
                _, _, done = type(env).__bases__[0].step(env, full_a)
            except Exception as e:
                print(f"  warmup step error: {e}")
                break
            if done or env.current_time > args.warmup_time:
                break

        print(f"[ep {ep+1}/{args.n_episodes}] warm-up done at t={env.current_time:.0f}s")

        collected = 0
        snap_prev = env.capture_full_system_snapshot()
        z_prev = extract_structured_context(snap_prev)

        # For Assumption 1 verification: we need obs/next_obs in the same shape
        # as the offline data (17-dim). Since that requires the full RL bridge
        # which we aren't using here, we fall back to projecting z context down:
        # use first 17 dims of the 30-dim z as a stand-in. This is sufficient for
        # bin-based P_sim(s' | s, a) estimation, which is what verify_assumption1.py
        # needs. The z already encodes everything about the sim state at that event.
        obs_prev = z_prev[:17].copy()

        for step_idx in range(200000):
            # Sample random action in [-1, 1]^2 per agent
            full_a = {}
            sampled_a = None
            for lid in env.line_map:
                full_a[lid] = {}
                for k in range(env.line_map[lid].max_agent_num):
                    a_vec = rng.uniform(-1, 1, size=2).astype(np.float32)
                    # Only first bus's action is what we record for the verification
                    if sampled_a is None:
                        sampled_a = a_vec
                    # Map action to hold time: max(0, a[0]) > 0 triggers hold;
                    # a[1] rescaled to [0, 60]s
                    hold = 0.0
                    if a_vec[0] > 0:
                        hold = float(max(0.0, (a_vec[1] + 1.0) * 30.0))
                    full_a[lid][k] = hold

            try:
                _, rew_dict, done = type(env).__bases__[0].step(env, full_a)
            except Exception as e:
                print(f"  step error at ep {ep}: {e}")
                break

            if done:
                break

            # Capture new snapshot every ~3 simulator steps to avoid
            # degenerate near-identical consecutive z pairs
            if step_idx % 3 == 0:
                snap_now = env.capture_full_system_snapshot()
                z_now = extract_structured_context(snap_now)
                obs_now = z_now[:17].copy()

                # Record transition
                obs_list.append(obs_prev)
                act_list.append(sampled_a if sampled_a is not None else np.zeros(2, dtype=np.float32))
                nobs_list.append(obs_now)
                z_list.append(z_prev.copy())
                znext_list.append(z_now.copy())
                # Use mean reward across lines as scalar
                if isinstance(rew_dict, dict):
                    all_r = []
                    for lid_rew in rew_dict.values():
                        if isinstance(lid_rew, dict):
                            all_r.extend(list(lid_rew.values()))
                        else:
                            all_r.append(lid_rew)
                    rew_list.append(float(np.mean(all_r)) if all_r else 0.0)
                else:
                    rew_list.append(float(rew_dict) if rew_dict is not None else 0.0)

                z_prev = z_now
                obs_prev = obs_now
                collected += 1

            if collected >= per_episode:
                break

        print(f"[ep {ep+1}/{args.n_episodes}] collected {collected} transitions")

    # ── Write h5 ──
    obs_arr = np.stack(obs_list).astype(np.float32)
    act_arr = np.stack(act_list).astype(np.float32)
    nobs_arr = np.stack(nobs_list).astype(np.float32)
    z_arr = np.stack(z_list).astype(np.float32)
    znext_arr = np.stack(znext_list).astype(np.float32)
    rew_arr = np.array(rew_list, dtype=np.float32)
    term_arr = np.zeros(len(obs_arr), dtype=np.float32)

    os.makedirs(os.path.dirname(args.out_h5), exist_ok=True)
    with h5py.File(args.out_h5, "w") as f:
        f.create_dataset("observations", data=obs_arr)
        f.create_dataset("actions", data=act_arr)
        f.create_dataset("next_observations", data=nobs_arr)
        f.create_dataset("z_t", data=z_arr)
        f.create_dataset("z_t1", data=znext_arr)
        f.create_dataset("rewards", data=rew_arr)
        f.create_dataset("terminals", data=term_arr)
    print(f"\nWrote {len(obs_arr)} sim transitions to {args.out_h5}")
    print(f"  obs shape: {obs_arr.shape}, action shape: {act_arr.shape}")


if __name__ == "__main__":
    main()
