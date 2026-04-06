"""
restore_z_from_backups.py
=========================
Restore original z_t/z_t1 from the .z_backup files created by rebuild_z_from_snapshots.py.

The rebuild was incorrect — SUMO's all-line z was the correct data, not a bug.
"""

import glob
import os
import h5py
import numpy as np

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")

for backup_path in sorted(glob.glob(os.path.join(DATASET_DIR, "*.h5.z_backup"))):
    h5_path = backup_path.replace(".z_backup", "")
    basename = os.path.basename(h5_path)

    with h5py.File(backup_path, "r") as fb:
        old_zt = np.array(fb["z_t_old"])
        old_zt1 = np.array(fb["z_t1_old"])

    with h5py.File(h5_path, "a") as f:
        del f["z_t"]
        del f["z_t1"]
        f.create_dataset("z_t", data=old_zt, compression="gzip")
        f.create_dataset("z_t1", data=old_zt1, compression="gzip")

    print(f"✅ Restored {basename}: z_t density[10:20] mean={old_zt[:, 10:20].mean():.4f}")

    # Remove backup
    os.remove(backup_path)

print(f"\nDone! Restored {len(glob.glob(os.path.join(DATASET_DIR, '*.h5.z_backup')))} remaining backups (should be 0).")
