#!/usr/bin/env python3
"""merge_v2_lazy.py — Merge 25 HDF5 → compact merged_all_v2.h5 with lazy snapshot index.

Combines merge_and_compress.py + augment_for_2d.py into a single streaming pass.

Output merged_all_v2.h5 contains:
    observations      [N, 17]   float32   (15 obs + 2 last_action)
    next_observations  [N, 17]   float32
    actions            [N, 2]    float32   (hold_time + speed_ratio)
    rewards            [N]       float32
    terminals          [N]       float32
    timeouts           [N]       float32
    z_t                [N, 30]   float32
    z_t1               [N, 30]   float32
    sim_time           [N]       float64
    policy_id          [N]       int8
    seed               [N]       int32
    snap_file_id       [N]       uint8     ← NEW: index into file_manifest
    snap_row_id        [N]       uint32    ← NEW: row in original HDF5

Snapshots are NOT copied — they stay in the original files.
SnapshotStore uses (snap_file_id, snap_row_id) for lazy on-demand loading.

Usage:
    cd H2Oplus/collect_policy
    python merge_v2_lazy.py --input_dir ../bus_h2o/datasets_v2
"""

import argparse
import glob
import json
import os
import sys
import time

import h5py
import numpy as np
from tqdm import tqdm


def parse_filename(path):
    """sumo_sac_seed42.h5 → ('sac', 42)"""
    base = os.path.basename(path).replace(".h5", "")
    parts = base.split("_")
    seed_str = [p for p in parts if p.startswith("seed")][0]
    seed = int(seed_str.replace("seed", ""))
    policy = "_".join(parts[1:-1])
    return policy, seed


def main():
    parser = argparse.ArgumentParser(
        description="Merge 25 HDF5 files with lazy snapshot index"
    )
    parser.add_argument("--input_dir", default="../bus_h2o/datasets_v2")
    parser.add_argument("--output", default="../bus_h2o/datasets_v2/merged_all_v2.h5")
    parser.add_argument("--manifest", default="../bus_h2o/datasets_v2/file_manifest.json",
                        help="Output JSON manifest for SnapshotStore")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "sumo_*.h5")))
    files = [f for f in files if "merged" not in f]
    print(f"Found {len(files)} source files")

    if not files:
        print("ERROR: No source files found!")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════════════
    # Pass 1: Count total transitions, build file manifest
    # ═══════════════════════════════════════════════════════════════════
    total_n = 0
    file_info = []       # [(fpath, policy, seed, n_rows), ...]
    file_manifest = []   # [(basename, n_rows), ...] — for SnapshotStore

    for fpath in files:
        policy, seed = parse_filename(fpath)
        with h5py.File(fpath, "r") as f:
            n = f["rewards"].shape[0]
        file_info.append((fpath, policy, seed, n))
        file_manifest.append((os.path.basename(fpath), n))
        total_n += n
        print(f"  [{len(file_info):2d}] {os.path.basename(fpath):45s}  "
              f"policy={policy:20s}  seed={seed:5d}  n={n:,}")

    print(f"\nTotal transitions: {total_n:,}")

    # Policy label encoding
    all_policies = sorted(set(p for _, p, _, _ in file_info))
    policy_map = {p: i for i, p in enumerate(all_policies)}
    print(f"Policy encoding: {policy_map}\n")

    # ═══════════════════════════════════════════════════════════════════
    # Pass 2: Create output file and copy data (streaming, per-file)
    # ═══════════════════════════════════════════════════════════════════
    if os.path.exists(args.output):
        os.remove(args.output)

    OBS_DIM_RAW = 15    # raw observation dimension (before augment)
    OBS_DIM_AUG = 17    # augmented: 15 + 2 last_action
    ACT_DIM = 2         # 2D action (hold + speed)
    Z_DIM = 30
    CHUNK = min(4096, total_n)

    t0 = time.time()

    with h5py.File(args.output, "w") as out:
        # Pre-allocate all datasets
        kw = dict(compression="gzip", compression_opts=6)

        ds_obs      = out.create_dataset("observations",      shape=(total_n, OBS_DIM_AUG), dtype=np.float32, chunks=(CHUNK, OBS_DIM_AUG), **kw)
        ds_next_obs = out.create_dataset("next_observations", shape=(total_n, OBS_DIM_AUG), dtype=np.float32, chunks=(CHUNK, OBS_DIM_AUG), **kw)
        ds_act      = out.create_dataset("actions",           shape=(total_n, ACT_DIM),     dtype=np.float32, chunks=(CHUNK, ACT_DIM),     **kw)
        ds_rew      = out.create_dataset("rewards",           shape=(total_n,),             dtype=np.float32, chunks=(CHUNK,),             **kw)
        ds_term     = out.create_dataset("terminals",         shape=(total_n,),             dtype=np.float32, chunks=(CHUNK,),             **kw)
        ds_timeout  = out.create_dataset("timeouts",          shape=(total_n,),             dtype=np.float32, chunks=(CHUNK,),             **kw)
        ds_zt       = out.create_dataset("z_t",               shape=(total_n, Z_DIM),       dtype=np.float32, chunks=(CHUNK, Z_DIM),       **kw)
        ds_zt1      = out.create_dataset("z_t1",              shape=(total_n, Z_DIM),       dtype=np.float32, chunks=(CHUNK, Z_DIM),       **kw)
        ds_simtime  = out.create_dataset("sim_time",          shape=(total_n,),             dtype=np.float64, chunks=(CHUNK,),             **kw)
        ds_pid      = out.create_dataset("policy_id",         shape=(total_n,),             dtype=np.int8,    chunks=(CHUNK,),             **kw)
        ds_seed     = out.create_dataset("seed",              shape=(total_n,),             dtype=np.int32,   chunks=(CHUNK,),             **kw)

        # Lazy snapshot index fields
        ds_sfid     = out.create_dataset("snap_file_id",      shape=(total_n,),             dtype=np.uint8,   chunks=(CHUNK,),             **kw)
        ds_srid     = out.create_dataset("snap_row_id",       shape=(total_n,),             dtype=np.uint32,  chunks=(CHUNK,),             **kw)

        offset = 0
        for file_id, (fpath, policy, seed_val, n) in enumerate(file_info):
            print(f"  [{file_id+1}/{len(file_info)}] Processing {os.path.basename(fpath)} "
                  f"({n:,} transitions)...", end="", flush=True)

            with h5py.File(fpath, "r") as src:
                # Read raw data
                obs_raw     = np.array(src["observations"],      dtype=np.float32)  # (n, 15 or 17)
                next_raw    = np.array(src["next_observations"],  dtype=np.float32)
                actions_raw = np.array(src["actions"],            dtype=np.float32)
                rewards     = np.array(src["rewards"],            dtype=np.float32).ravel()
                terminals   = np.array(src["terminals"],          dtype=np.float32).ravel()
                zt          = np.array(src["z_t"],                dtype=np.float32)
                zt1         = np.array(src["z_t1"],               dtype=np.float32)
                sim_time    = np.array(src["sim_time"],           dtype=np.float64).ravel()

                # Handle timeouts (may not exist)
                if "timeouts" in src:
                    timeouts = np.array(src["timeouts"], dtype=np.float32).ravel()
                else:
                    timeouts = np.zeros(n, dtype=np.float32)

            # ── Augment obs & actions ──────────────────────────────────
            obs_dim = obs_raw.shape[1]
            act_dim = actions_raw.shape[1] if actions_raw.ndim > 1 else 1

            # Ensure actions are 2D
            if act_dim == 1:
                actions_2d = np.zeros((n, 2), dtype=np.float32)
                actions_2d[:, 0] = actions_raw.ravel()
            else:
                actions_2d = actions_raw.reshape(n, -1)[:, :2]

            # Augment observations to 17-dim if needed
            if obs_dim == OBS_DIM_RAW:
                # Reconstruct last_action per bus within this file
                bus_ids = obs_raw[:, 1].astype(np.int32)
                obs_sim_time = obs_raw[:, 10]

                last_action_arr = np.zeros((n, 2), dtype=np.float32)

                # Group by bus_id, sort by sim_time, chain last_action
                unique_buses = np.unique(bus_ids)
                for bid in unique_buses:
                    bus_mask = bus_ids == bid
                    bus_indices = np.where(bus_mask)[0]
                    bus_times = obs_sim_time[bus_indices]
                    sort_order = np.argsort(bus_times, kind='stable')
                    sorted_indices = bus_indices[sort_order]

                    for i, idx in enumerate(sorted_indices):
                        if i == 0:
                            last_action_arr[idx] = [0.0, 0.0]
                        else:
                            prev_idx = sorted_indices[i - 1]
                            last_action_arr[idx] = actions_2d[prev_idx]

                obs_aug = np.concatenate([obs_raw, last_action_arr], axis=1)
                next_last_action = actions_2d.copy()
                next_obs_aug = np.concatenate([next_raw, next_last_action], axis=1)
            elif obs_dim == OBS_DIM_AUG:
                # Already 17-dim
                obs_aug = obs_raw
                next_obs_aug = next_raw
            else:
                raise ValueError(f"Unexpected obs dim {obs_dim} in {fpath}")

            # ── Write to output ────────────────────────────────────────
            sl = slice(offset, offset + n)
            ds_obs[sl]      = obs_aug
            ds_next_obs[sl] = next_obs_aug
            ds_act[sl]      = actions_2d
            ds_rew[sl]      = rewards
            ds_term[sl]     = terminals
            ds_timeout[sl]  = timeouts
            ds_zt[sl]       = zt
            ds_zt1[sl]      = zt1
            ds_simtime[sl]  = sim_time
            ds_pid[sl]      = policy_map[policy]
            ds_seed[sl]     = seed_val

            # Lazy snapshot index
            ds_sfid[sl] = file_id
            ds_srid[sl] = np.arange(n, dtype=np.uint32)

            offset += n
            print(f" done ({time.time()-t0:.0f}s elapsed)")

        # Store metadata
        out.attrs["policy_names"] = json.dumps(policy_map)
        out.attrs["n_files"] = len(file_info)
        out.attrs["n_transitions"] = total_n
        out.attrs["obs_dim"] = OBS_DIM_AUG
        out.attrs["action_dim"] = ACT_DIM
        out.attrs["z_dim"] = Z_DIM
        out.attrs["has_lazy_snapshot_index"] = True

    # ═══════════════════════════════════════════════════════════════════
    # Write file manifest JSON (for SnapshotStore)
    # ═══════════════════════════════════════════════════════════════════
    manifest_out = args.manifest
    with open(manifest_out, "w") as f:
        json.dump(file_manifest, f, indent=2)
    print(f"\nFile manifest written to: {manifest_out}")

    # ═══════════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════════
    final_size = os.path.getsize(args.output) / 1024**2
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Merged file: {args.output}")
    print(f"Total transitions: {total_n:,}")
    print(f"File size: {final_size:.1f} MB")
    print(f"Time: {elapsed:.0f}s")
    print(f"Datasets: observations({OBS_DIM_AUG}), actions({ACT_DIM}), "
          f"snap_file_id, snap_row_id + standard fields")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
