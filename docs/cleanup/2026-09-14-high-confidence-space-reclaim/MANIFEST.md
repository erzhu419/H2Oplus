# Frozen WSL cleanup manifest

- Batch: `2026-09-14-high-confidence-space-reclaim`
- Project root: `/home/erzhu419/mine_code/sumo-rl/H2Oplus`
- Exact records: **71** (69 deployment + 2 extra workspace paths)
- Record / backup / deletion status: `completed` / `verified` / `deleted`
- Backup root: `jtl311linux:/home/erzhu419/wsl_cleanup_backup_before_delete_20260914` (original absolute paths preserved)
- Individual allocated sum: **57222549504 B**
- Unique allocated bytes across allowlist: **14423457792 B**
- Predicted actually reclaimable allocated bytes: **12795625472**

Only the 71 paths in `items/*.json` are allowlisted. Backup uses one `rsync -aHzR` stream with `-H` across all sources. Verification is per-workspace `rsync -ani --delete` with no `--size-only`. Deletion is exact-path only and requires 71 successful verification receipts plus a fresh scheduler/process/fd gate.

The deployment parent directories and all outside `results`, `closeout`, `manifest`, and `remote_results` artifacts are excluded from deletion.
