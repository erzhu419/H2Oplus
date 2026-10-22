# Exact H2Oplus 71-workspace execution protocol

This batch contains exactly 69 `reproducibility/deployments/*/workspace` paths plus the two explicitly recorded diagnostic/incident workspaces.

1. `python3 safe_backup_runner.py prepare --all --execute --confirm PREPARE_EXACT_71`
2. `python3 safe_backup_runner.py preflight --all`
3. `python3 safe_backup_runner.py gate --all`
4. `python3 safe_backup_runner.py backup --all --execute --confirm BACKUP_EXACT_71`
5. `python3 safe_backup_runner.py verify --all --execute --confirm VERIFY_EXACT_71`
6. Only after 71/71 empty strict dry-runs: `python3 safe_backup_runner.py delete --all --execute --confirm DELETE_EXACT_71`

Remote root: `jtl311linux:/home/erzhu419/wsl_cleanup_backup_before_delete_20260914`. The single backup stream uses `rsync -aHzR --partial --compress-level=1`; each verification uses `rsync -ani --delete` and forbids `--size-only`.

Deletion is limited to the 71 exact `source_path` directories. Deployment parents and outside results/closeout/manifest/remote_results artifacts are preserved.
