"""
augment_for_2d.py
=================
Converts merged_all.h5 → merged_all_v2.h5 with two key augmentations:

1. Actions padded from (N, 1) → (N, 2) with speed_ratio=0 (since speed was
   never applied during collection).

2. Observations augmented from (N, 15) → (N, 17) by appending per-bus
   last_action (2D) to each observation vector.  This matches the legacy SAC
   policy architecture which takes obs(15) + prev_action(2) = 17 inputs.

   last_action reconstruction:
     - Group transitions by (seed, bus_id) to identify per-bus sequences.
     - Within each group, sort by sim_time (obs[10]).
     - For the first transition of each bus, last_action = [0, 0].
     - For subsequent transitions, last_action = action from previous step.

Output file has the same fields as input plus:
   observations:      (N, 17)   ← was (N, 15)
   next_observations: (N, 17)   ← was (N, 15)
   actions:           (N, 2)    ← was (N, 1)
"""

import os
import sys
import numpy as np
import h5py
from tqdm import tqdm

INPUT = "merged_all.h5"
OUTPUT = "merged_all_v2.h5"


def main():
    print(f"[augment_for_2d] Reading {INPUT} ...")
    with h5py.File(INPUT, "r") as f:
        obs = np.array(f["observations"], dtype=np.float32)           # (N, 15)
        next_obs = np.array(f["next_observations"], dtype=np.float32) # (N, 15)
        actions = np.array(f["actions"], dtype=np.float32).reshape(-1, 1)  # (N, 1)
        rewards = np.array(f["rewards"], dtype=np.float32)
        terminals = np.array(f["terminals"], dtype=np.float32)
        timeouts = np.array(f["timeouts"], dtype=np.float32)
        z_t = np.array(f["z_t"], dtype=np.float32)
        z_t1 = np.array(f["z_t1"], dtype=np.float32)
        policy_id = np.array(f["policy_id"])
        seed = np.array(f["seed"])
        sim_time = np.array(f["sim_time"], dtype=np.float64)

    N = obs.shape[0]
    print(f"[augment_for_2d] Loaded {N:,} transitions")

    # ── 1. Pad actions to 2D ────────────────────────────────────────────
    print("[augment_for_2d] Padding actions from (N,1) → (N,2) ...")
    actions_2d = np.zeros((N, 2), dtype=np.float32)
    actions_2d[:, 0] = actions[:, 0]  # hold_time
    # actions_2d[:, 1] = 0.0           # speed_ratio (never applied)

    # ── 2. Reconstruct last_action per bus ──────────────────────────────
    print("[augment_for_2d] Reconstructing last_action per bus ...")

    # obs[:,1] = bus_id, obs[:,10] = sim_time within episode
    bus_ids = obs[:, 1].astype(np.int32)
    obs_sim_time = obs[:, 10]

    # Group by (seed, bus_id)
    last_action_arr = np.zeros((N, 2), dtype=np.float32)

    # For seed=0 the sim_time field in the HDF5 is 0, use obs sim_time
    effective_time = np.where(sim_time > 0, sim_time, obs_sim_time)

    unique_seeds = np.unique(seed)
    for s in tqdm(unique_seeds, desc="Seeds"):
        seed_mask = seed == s
        seed_indices = np.where(seed_mask)[0]
        seed_bus_ids = bus_ids[seed_indices]

        unique_buses = np.unique(seed_bus_ids)
        for bid in unique_buses:
            bus_mask = seed_bus_ids == bid
            bus_indices = seed_indices[bus_mask]

            # Sort by sim_time within this bus
            bus_times = effective_time[bus_indices]
            sort_order = np.argsort(bus_times, kind='stable')
            sorted_indices = bus_indices[sort_order]

            # First transition: last_action = [0, 0]
            # Subsequent transitions: last_action = previous action
            for i, idx in enumerate(sorted_indices):
                if i == 0:
                    last_action_arr[idx] = [0.0, 0.0]
                else:
                    prev_idx = sorted_indices[i - 1]
                    last_action_arr[idx] = actions_2d[prev_idx]

    # ── 3. Augment observations: append last_action ─────────────────────
    print("[augment_for_2d] Augmenting observations (15→17) ...")
    obs_aug = np.concatenate([obs, last_action_arr], axis=1)  # (N, 17)

    # For next_observations, we need last_action for the NEXT state.
    # The "last_action" for next_obs[i] is the action[i] (the action that
    # was taken to transition from obs[i] to next_obs[i]).
    next_last_action = actions_2d.copy()
    next_obs_aug = np.concatenate([next_obs, next_last_action], axis=1)  # (N, 17)

    # ── 4. Write output ────────────────────────────────────────────────
    print(f"[augment_for_2d] Writing {OUTPUT} ...")
    with h5py.File(OUTPUT, "w") as f:
        f.create_dataset("observations", data=obs_aug, dtype=np.float32,
                          compression="gzip", compression_opts=4)
        f.create_dataset("next_observations", data=next_obs_aug, dtype=np.float32,
                          compression="gzip", compression_opts=4)
        f.create_dataset("actions", data=actions_2d, dtype=np.float32,
                          compression="gzip", compression_opts=4)
        f.create_dataset("rewards", data=rewards, dtype=np.float32,
                          compression="gzip", compression_opts=4)
        f.create_dataset("terminals", data=terminals, dtype=np.float32,
                          compression="gzip", compression_opts=4)
        f.create_dataset("timeouts", data=timeouts, dtype=np.float32,
                          compression="gzip", compression_opts=4)
        f.create_dataset("z_t", data=z_t, dtype=np.float32,
                          compression="gzip", compression_opts=4)
        f.create_dataset("z_t1", data=z_t1, dtype=np.float32,
                          compression="gzip", compression_opts=4)
        f.create_dataset("policy_id", data=policy_id)
        f.create_dataset("seed", data=seed)
        f.create_dataset("sim_time", data=sim_time, dtype=np.float64,
                          compression="gzip", compression_opts=4)

    # ── 5. Verify ──────────────────────────────────────────────────────
    print(f"\n[augment_for_2d] Verification:")
    with h5py.File(OUTPUT, "r") as f:
        for k in sorted(f.keys()):
            print(f"  {k}: {f[k].shape}, {f[k].dtype}")

    out_size = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"\n[augment_for_2d] Output size: {out_size:.1f} MB")
    print("[augment_for_2d] Done!")


if __name__ == "__main__":
    main()
