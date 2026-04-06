"""
snapshot_store.py
=================
Two-tier lazy-loading snapshot access for H2O+ training.

Design:
    - Training uses merged_all_v2.h5 (numerical data only, ~73MB in RAM)
    - Snapshots stay on disk in original HDF5 files (~4GB each)
    - Each merged transition has (snap_file_id, snap_row_id) pointing into originals
    - SnapshotStore lazy-opens file handles and uses LRU cache for hot snapshots

Latency: ~20KB random read per snapshot, <1ms on SSD.
RAM overhead: cache_size × ~20KB = ~5MB default.
"""

import os
import pickle
import logging
from collections import OrderedDict

import h5py
import numpy as np


class SnapshotStore:
    """Lazy-load snapshots from on-disk HDF5 archive with LRU caching.

    Usage:
        store = SnapshotStore(
            archive_dir="/path/to/datasets_v2",
            file_manifest=[("sumo_zero_seed42.h5", 13500), ...]
        )
        snap_bytes = store.get(file_id=0, row_id=42)
        snap_dict  = pickle.loads(snap_bytes)
    """

    def __init__(self, archive_dir, file_manifest, cache_size=256,
                 snapshot_key="raw_snapshot"):
        """
        Args:
            archive_dir:    Directory containing original HDF5 files.
            file_manifest:  List of (filename, n_rows) tuples.
                            Index in list = file_id.
            cache_size:     Max snapshots to keep in LRU cache.
            snapshot_key:   HDF5 dataset name for snapshot blobs.
        """
        self.archive_dir = archive_dir
        self.file_manifest = file_manifest
        self.snapshot_key = snapshot_key
        self._file_handles = {}       # file_id → h5py.File (lazy-opened)
        self._cache = OrderedDict()   # (file_id, row_id) → snapshot_bytes
        self._cache_size = cache_size
        self._miss_count = 0
        self._hit_count = 0

    def get(self, file_id: int, row_id: int) -> bytes:
        """Lazy-load a single snapshot (~20KB I/O, <1ms on SSD).

        Returns raw pickle bytes. Caller does pickle.loads() if needed.
        """
        key = (file_id, row_id)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hit_count += 1
            return self._cache[key]

        self._miss_count += 1

        # Open file handle (pooled, never closed until __del__)
        if file_id not in self._file_handles:
            fname = self.file_manifest[file_id][0]
            fpath = os.path.join(self.archive_dir, fname)
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Snapshot archive not found: {fpath}"
                )
            self._file_handles[file_id] = h5py.File(fpath, "r")
            logging.debug(
                f"[SnapshotStore] Opened {fname} "
                f"({self.file_manifest[file_id][1]} rows)"
            )

        h5 = self._file_handles[file_id]
        if self.snapshot_key not in h5:
            raise KeyError(
                f"Dataset '{self.snapshot_key}' not found in "
                f"{self.file_manifest[file_id][0]}"
            )

        blob = h5[self.snapshot_key][row_id]
        snap_bytes = bytes(blob)

        # LRU eviction
        self._cache[key] = snap_bytes
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

        return snap_bytes

    def get_by_buffer_idx(self, snap_file_id_arr, snap_row_id_arr,
                          idx: int) -> dict:
        """Convenience: look up by merged buffer index, return unpickled dict.

        Args:
            snap_file_id_arr: numpy array of file IDs (from buffer).
            snap_row_id_arr:  numpy array of row IDs (from buffer).
            idx:              Index into the merged buffer.

        Returns:
            Unpickled snapshot dict.
        """
        file_id = int(snap_file_id_arr[idx])
        row_id = int(snap_row_id_arr[idx])
        snap_bytes = self.get(file_id, row_id)
        return pickle.loads(snap_bytes)

    @property
    def cache_stats(self):
        total = self._hit_count + self._miss_count
        return {
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": self._hit_count / max(total, 1),
            "cache_size": len(self._cache),
            "open_files": len(self._file_handles),
        }

    def close(self):
        """Close all open HDF5 file handles."""
        for fid, h5 in self._file_handles.items():
            try:
                h5.close()
            except Exception:
                pass
        self._file_handles.clear()
        self._cache.clear()

    def __del__(self):
        self.close()
