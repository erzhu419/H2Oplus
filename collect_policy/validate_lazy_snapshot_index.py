#!/usr/bin/env python3
"""Validate merged_all_v2 lazy snapshot index.

Checks that a merged HDF5 file can map resettable rows back to serialized
snapshots in the original per-policy HDF5 archives.
"""

import argparse
import json
import os
import sys

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--archive_dir", required=True)
    parser.add_argument("--snapshot_key", default="snapshot_T1")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo_root, "SimpleSAC"))
    from snapshot_store import SnapshotStore

    required = {"observations", "actions", "snap_file_id", "snap_row_id", "snap_valid"}
    with h5py.File(args.merged, "r") as f:
        missing = sorted(required - set(f.keys()))
        if missing:
            raise RuntimeError(f"Missing merged datasets: {missing}")

        n_rows = int(f["observations"].shape[0])
        snap_valid = np.array(f["snap_valid"], dtype=bool)
        if snap_valid.shape[0] != n_rows:
            raise RuntimeError(
                f"snap_valid length {snap_valid.shape[0]} != observations {n_rows}"
            )
        valid_indices = np.flatnonzero(snap_valid)
        if valid_indices.size == 0:
            raise RuntimeError("No resettable rows in snap_valid")

        snap_file_id = np.array(f["snap_file_id"], dtype=np.uint8)
        snap_row_id = np.array(f["snap_row_id"], dtype=np.uint32)

    with open(args.manifest) as mf:
        manifest = json.load(mf)
    if not manifest:
        raise RuntimeError("Empty file manifest")

    max_file_id = int(snap_file_id[valid_indices].max())
    if max_file_id >= len(manifest):
        raise RuntimeError(
            f"snap_file_id max {max_file_id} exceeds manifest size {len(manifest)}"
        )

    if args.samples >= valid_indices.size:
        sample_indices = valid_indices
    else:
        positions = np.linspace(0, valid_indices.size - 1, args.samples, dtype=int)
        sample_indices = valid_indices[positions]

    store = SnapshotStore(
        archive_dir=args.archive_dir,
        file_manifest=manifest,
        cache_size=max(args.samples, 1),
        snapshot_key=args.snapshot_key,
    )
    try:
        for idx in sample_indices:
            snap = store.get_by_buffer_idx(snap_file_id, snap_row_id, int(idx))
            if not isinstance(snap, dict):
                raise RuntimeError(f"Snapshot at row {idx} is not a dict")
            for key in ("current_time", "all_buses", "all_stations"):
                if key not in snap:
                    raise RuntimeError(f"Snapshot at row {idx} missing {key!r}")
    finally:
        store.close()

    print(
        "lazy snapshot index OK: "
        f"rows={n_rows:,}, resettable={valid_indices.size:,}, "
        f"manifest_files={len(manifest)}, samples={len(sample_indices)}"
    )


if __name__ == "__main__":
    main()
