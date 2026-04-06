#!/usr/bin/env python3
"""merge_and_compress.py — Strip snapshots, merge 25 HDF5 → 1 compact file.

Drops: raw_snapshot, snapshot_T1 (serialized objects, ~95% of disk usage)
Keeps: observations, next_observations, actions, rewards, terminals, timeouts,
       sim_time, z_t, z_t1  (all gzip-compressed)
Adds:  policy (string label), seed (int) as per-transition metadata
"""
import argparse, glob, os, sys
import h5py
import numpy as np

KEEP_KEYS = [
    "observations", "next_observations", "actions",
    "rewards", "terminals", "timeouts",
    "sim_time", "z_t", "z_t1",
]

def parse_filename(path):
    """sumo_sac_seed42.h5 → ('sac', 42)"""
    base = os.path.basename(path).replace(".h5", "")
    parts = base.split("_")
    # sumo_{policy}_seed{seed}
    seed_str = [p for p in parts if p.startswith("seed")][0]
    seed = int(seed_str.replace("seed", ""))
    # policy is everything between 'sumo_' and '_seedXXX'
    policy = "_".join(parts[1:-1])
    return policy, seed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="../bus_h2o/datasets_v2")
    parser.add_argument("--output", default="../bus_h2o/datasets_v2/merged_all.h5")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "sumo_*.h5")))
    files = [f for f in files if "merged" not in f]
    print(f"Found {len(files)} source files")

    # ---- Pass 1: count total transitions ----
    total_n = 0
    file_info = []
    for fpath in files:
        policy, seed = parse_filename(fpath)
        with h5py.File(fpath, "r") as f:
            n = f["rewards"].shape[0]
        file_info.append((fpath, policy, seed, n))
        total_n += n
        print(f"  {os.path.basename(fpath):45s}  policy={policy:20s}  seed={seed:5d}  n={n}")
    print(f"\nTotal transitions: {total_n:,}")

    # ---- Pass 2: create output and copy ----
    if os.path.exists(args.output):
        os.remove(args.output)

    with h5py.File(args.output, "w") as out:
        # Pre-allocate datasets
        # Probe shapes & dtypes from first file
        with h5py.File(files[0], "r") as probe:
            for key in KEEP_KEYS:
                ds = probe[key]
                shape = list(ds.shape)
                shape[0] = total_n
                out.create_dataset(
                    key,
                    shape=tuple(shape),
                    dtype=ds.dtype,
                    chunks=(min(4096, total_n),) + tuple(shape[1:]),
                    compression="gzip",
                    compression_opts=6,
                )

        # Policy label encoded as int for compactness
        all_policies = sorted(set(p for _, p, _, _ in file_info))
        policy_map = {p: i for i, p in enumerate(all_policies)}
        print(f"\nPolicy encoding: {policy_map}")

        out.create_dataset("policy_id", shape=(total_n,), dtype=np.int8,
                           chunks=(min(4096, total_n),),
                           compression="gzip", compression_opts=6)
        out.create_dataset("seed", shape=(total_n,), dtype=np.int32,
                           chunks=(min(4096, total_n),),
                           compression="gzip", compression_opts=6)
        # Store the mapping as an attribute
        out.attrs["policy_names"] = str(policy_map)

        # Copy data
        offset = 0
        for i, (fpath, policy, seed, n) in enumerate(file_info):
            print(f"  [{i+1}/{len(file_info)}] Copying {os.path.basename(fpath)}  "
                  f"({n:,} transitions, offset={offset:,})...", end="", flush=True)
            with h5py.File(fpath, "r") as src:
                for key in KEEP_KEYS:
                    out[key][offset:offset+n] = src[key][:]
            out["policy_id"][offset:offset+n] = policy_map[policy]
            out["seed"][offset:offset+n] = seed
            offset += n
            print(" done")

        print(f"\nMerged file: {args.output}")
        print(f"Total transitions: {total_n:,}")
        print(f"Datasets: {list(out.keys())}")

    final_size = os.path.getsize(args.output)
    print(f"Final size: {final_size / 1024**3:.2f} GB")

if __name__ == "__main__":
    main()
