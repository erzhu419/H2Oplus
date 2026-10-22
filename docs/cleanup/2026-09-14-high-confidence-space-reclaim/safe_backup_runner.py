#!/usr/bin/env python3
"""Exact 71-workspace H2Oplus backup, verification, and deletion runner.

The allowlist is the existing project-root manifest.  Backup is one rsync
archive stream so hardlinks shared by different workspaces remain shared.
Deletion is unavailable until every workspace has an empty archive-semantic
rsync dry-run against the unified remote backup root.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
from collections import defaultdict


BATCH_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = BATCH_DIR / "manifest.json"
PROJECT_ROOT = Path("/home/erzhu419/mine_code/sumo-rl/H2Oplus")
QUEUE_PATH = Path("/home/erzhu419/.claude/scheduler/queue.json")
HOST = "jtl311linux"
REMOTE_ROOT = Path("/home/erzhu419/wsl_cleanup_backup_before_delete_20260914")
EXPECTED_EXTRAS = {
    PROJECT_ROOT
    / "reproducibility/diagnostics/server_trainer_gate_v4_wrong_builder_runtime/workspace",
    PROJECT_ROOT
    / "reproducibility/h2oplus_v12_staged_release_results/evaluation_incidents/pre_submission_source_contract_v1/deployment_v1/workspace",
}
TERMINAL_SCHEDULER_STATES = {"done", "failed", "cancelled", "canceled"}


def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def log_event(event: str, **fields: object) -> None:
    record = {"at": now(), "event": event, **fields}
    with (BATCH_DIR / "execution-log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_batch() -> tuple[dict, list[dict], list[Path]]:
    manifest = read_json(MANIFEST_PATH)
    item_paths = [BATCH_DIR / rel for rel in manifest["item_files"]]
    items = [read_json(path) for path in item_paths]
    if manifest.get("item_count") != 71 or len(items) != 71:
        raise RuntimeError("the frozen manifest must contain exactly 71 items")
    if len({item["item_id"] for item in items}) != 71:
        raise RuntimeError("duplicate item_id in frozen manifest")
    sources = [Path(item["source_path"]) for item in items]
    if len(set(sources)) != 71:
        raise RuntimeError("duplicate source_path in frozen manifest")
    deployment = {
        path for path in sources if PROJECT_ROOT / "reproducibility/deployments" in path.parents
    }
    extras = set(sources) - deployment
    if len(deployment) != 69 or extras != EXPECTED_EXTRAS:
        raise RuntimeError(
            f"allowlist shape changed: deployment={len(deployment)}, extras={sorted(map(str, extras))}"
        )
    for source in sources:
        if source.name != "workspace" or PROJECT_ROOT not in source.parents:
            raise RuntimeError(f"out-of-scope source path: {source}")
    for left in sources:
        for right in sources:
            if left != right and left in right.parents:
                raise RuntimeError(f"nested allowlist paths: {left} and {right}")
    return manifest, items, item_paths


def du_bytes(path: Path, apparent: bool = False) -> int:
    command = ["du", "-s", "-B1"]
    if apparent:
        command.append("--apparent-size")
    command.extend(["--", os.fspath(path)])
    output = subprocess.check_output(command, text=True)
    return int(output.split(None, 1)[0])


def entry_counts(path: Path) -> dict[str, int]:
    counts = {"directories": 0, "regular_files": 0, "symlinks": 0, "other_entries": 0}

    def add(candidate: Path) -> None:
        mode = candidate.lstat().st_mode
        if stat.S_ISDIR(mode):
            counts["directories"] += 1
        elif stat.S_ISREG(mode):
            counts["regular_files"] += 1
        elif stat.S_ISLNK(mode):
            counts["symlinks"] += 1
        else:
            counts["other_entries"] += 1

    add(path)
    for root, directories, files in os.walk(path, followlinks=False):
        root_path = Path(root)
        for name in directories:
            add(root_path / name)
        for name in files:
            add(root_path / name)
    return counts


def preflight_item(item: dict, *, allow_deleted: bool = False) -> None:
    source = Path(item["source_path"])
    if item.get("deletion_status") == "deleted" and allow_deleted:
        if source.exists() or source.is_symlink():
            raise RuntimeError(f"{item['item_id']}: marked deleted but source exists")
        return
    if not source.exists() or source.is_symlink() or not source.is_dir():
        raise RuntimeError(f"{item['item_id']}: source is missing, symlinked, or not a directory")
    frozen = item["frozen_metadata"]
    actual_allocated = du_bytes(source)
    actual_apparent = du_bytes(source, apparent=True)
    actual_counts = entry_counts(source)
    mismatches = []
    if actual_allocated != frozen["allocated_bytes"]:
        mismatches.append(f"allocated {actual_allocated} != {frozen['allocated_bytes']}")
    if actual_apparent != frozen["apparent_bytes"]:
        mismatches.append(f"apparent {actual_apparent} != {frozen['apparent_bytes']}")
    if actual_counts != frozen["entry_counts"]:
        mismatches.append(f"entries {actual_counts!r} != {frozen['entry_counts']!r}")
    if mismatches:
        raise RuntimeError(f"{item['item_id']}: source changed: " + "; ".join(mismatches))


def remote_item_path(source: Path) -> Path:
    return REMOTE_ROOT / source.relative_to("/")


def render_records(manifest: dict, items: list[dict], item_paths: list[Path]) -> None:
    for item, path in zip(items, item_paths):
        atomic_json(path, item)
    atomic_json(MANIFEST_PATH, manifest)

    with (BATCH_DIR / "backup-plan.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["item_id", "source_kind", "allocated_bytes", "apparent_bytes", "source_path",
             "remote_item_path", "record_status", "backup_status", "deletion_status"]
        )
        for item in items:
            writer.writerow(
                [item["item_id"], item["source_kind"], item["frozen_metadata"]["allocated_bytes"],
                 item["frozen_metadata"]["apparent_bytes"], item["source_path"],
                 item["backup"]["remote_item_path"], item["record_status"],
                 item["backup_status"], item["deletion_status"]]
            )

    (BATCH_DIR / "MANIFEST.md").write_text(
        "\n".join(
            [
                "# Frozen WSL cleanup manifest",
                "",
                f"- Batch: `{manifest['batch_id']}`",
                f"- Project root: `{manifest['project_root']}`",
                f"- Exact records: **{manifest['item_count']}** (69 deployment + 2 extra workspace paths)",
                f"- Record / backup / deletion status: `{manifest['status']}` / `{manifest['backup_status']}` / `{manifest['deletion_status']}`",
                f"- Backup root: `{HOST}:{REMOTE_ROOT}` (original absolute paths preserved)",
                f"- Individual allocated sum: **{manifest['allocated_bytes_sum_individual']} B**",
                f"- Unique allocated bytes across allowlist: **{manifest['allocated_bytes_unique_across_allowlist']} B**",
                f"- Predicted actually reclaimable allocated bytes: **{manifest.get('predicted_reclaim_allocated_bytes', 'pending')}**",
                "",
                "Only the 71 paths in `items/*.json` are allowlisted. Backup uses one `rsync -aHzR` stream with `-H` across all sources. Verification is per-workspace `rsync -ani --delete` with no `--size-only`. Deletion is exact-path only and requires 71 successful verification receipts plus a fresh scheduler/process/fd gate.",
                "",
                "The deployment parent directories and all outside `results`, `closeout`, `manifest`, and `remote_results` artifacts are excluded from deletion.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def formal_snapshot(sources: list[Path]) -> dict:
    source_set = set(sources)
    reproducibility = PROJECT_ROOT / "reproducibility"
    outside_named: list[str] = []
    closeout_count = 0
    manifest_count = 0
    remote_result_dirs: list[str] = []
    for root, directories, files in os.walk(reproducibility, followlinks=False):
        root_path = Path(root)
        directories[:] = [name for name in directories if root_path / name not in source_set]
        for name in directories + files:
            path = root_path / name
            lower = name.lower()
            if "closeout" in lower:
                closeout_count += 1
            if "manifest" in lower:
                manifest_count += 1
            if path.is_dir() and "remote_results" in lower:
                remote_result_dirs.append(os.fspath(path))
    sibling_result_dirs = sorted(
        {
            os.fspath(path)
            for source in sources
            for path in source.parent.iterdir()
            if path != source and path.is_dir() and "result" in path.name.lower()
        }
    )
    top_level_result_dirs = sorted(
        os.fspath(path)
        for path in reproducibility.iterdir()
        if path.is_dir() and "result" in path.name.lower()
    )
    outside_named.extend(top_level_result_dirs)
    outside_named.extend(sibling_result_dirs)
    outside_named.extend(remote_result_dirs)
    return {
        "checked_at": now(),
        "protected_outside_paths": sorted(set(outside_named)),
        "top_level_result_directory_count": len(top_level_result_dirs),
        "workspace_parent_sibling_result_directory_count": len(sibling_result_dirs),
        "remote_result_directory_count": len(set(remote_result_dirs)),
        "outside_closeout_named_entry_count": closeout_count,
        "outside_manifest_named_entry_count": manifest_count,
    }


def assert_formal_snapshot(current: dict, frozen: dict) -> None:
    missing = [path for path in frozen["protected_outside_paths"] if not Path(path).exists()]
    if missing:
        raise RuntimeError(f"protected outside result paths disappeared: {missing[:5]}")
    for field in (
        "top_level_result_directory_count",
        "workspace_parent_sibling_result_directory_count",
        "remote_result_directory_count",
        "outside_closeout_named_entry_count",
        "outside_manifest_named_entry_count",
    ):
        if current[field] < frozen[field]:
            raise RuntimeError(f"formal artifact gate regressed: {field} {current[field]} < {frozen[field]}")


def activity_gate(sources: list[Path]) -> dict:
    source_strings = [os.fspath(path) for path in sources]
    queue = read_json(QUEUE_PATH)
    queue_hits = []
    nonterminal = 0
    for task in queue.get("tasks", []):
        status_value = str(task.get("status", "")).lower()
        if status_value in TERMINAL_SCHEDULER_STATES:
            continue
        nonterminal += 1
        encoded = json.dumps(task, ensure_ascii=False)
        matches = [path for path in source_strings if path in encoded]
        if matches:
            queue_hits.append({"id": task.get("id"), "status": status_value, "paths": matches})

    process_hits = []
    proc_count = 0
    fd_count = 0
    self_pid = os.getpid()
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == self_pid:
            continue
        try:
            cmdline = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            proc_count += 1
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        matches = {path for path in source_strings if path in cmdline}
        for link_name in ("cwd",):
            try:
                target = os.readlink(proc / link_name)
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            matches.update(path for path in source_strings if target == path or target.startswith(path + "/"))
        try:
            descriptors = list((proc / "fd").iterdir())
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            descriptors = []
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
                fd_count += 1
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            matches.update(path for path in source_strings if target == path or target.startswith(path + "/"))
        if matches:
            process_hits.append({"pid": int(proc.name), "cmdline": cmdline[:500], "paths": sorted(matches)})
    result = {
        "checked_at": now(),
        "scheduler_source": os.fspath(QUEUE_PATH),
        "nonterminal_scheduler_tasks_checked": nonterminal,
        "nonterminal_scheduler_exact_candidate_path_hits": queue_hits,
        "processes_checked": proc_count,
        "fds_checked": fd_count,
        "process_argv_cwd_fd_exact_candidate_path_hits": process_hits,
    }
    if queue_hits or process_hits:
        raise RuntimeError(f"activity gate failed: queue={len(queue_hits)}, process/fd={len(process_hits)}")
    return result


def predicted_reclaim(sources: list[Path]) -> dict:
    directory_inodes: set[tuple[int, int]] = set()
    directory_blocks = 0
    occurrences: defaultdict[tuple[int, int], int] = defaultdict(int)
    blocks: dict[tuple[int, int], int] = {}
    links: dict[tuple[int, int], int] = {}
    for source in sources:
        for root, directories, files in os.walk(source, followlinks=False):
            root_path = Path(root)
            root_stat = root_path.lstat()
            root_key = (root_stat.st_dev, root_stat.st_ino)
            if root_key not in directory_inodes:
                directory_inodes.add(root_key)
                directory_blocks += root_stat.st_blocks * 512
            for name in directories + files:
                entry = root_path / name
                entry_stat = entry.lstat()
                if stat.S_ISDIR(entry_stat.st_mode) and not stat.S_ISLNK(entry_stat.st_mode):
                    continue
                key = (entry_stat.st_dev, entry_stat.st_ino)
                occurrences[key] += 1
                blocks[key] = entry_stat.st_blocks * 512
                links[key] = entry_stat.st_nlink
    exclusive = {key for key, count in occurrences.items() if count == links[key]}
    shared_outside = set(occurrences) - exclusive
    return {
        "calculated_at": now(),
        "directory_unique_inode_count": len(directory_inodes),
        "directory_allocated_bytes": directory_blocks,
        "nondirectory_unique_inode_count": len(occurrences),
        "exclusive_nondirectory_inode_count": len(exclusive),
        "outside_linked_nondirectory_inode_count": len(shared_outside),
        "exclusive_nondirectory_allocated_bytes": sum(blocks[key] for key in exclusive),
        "retained_via_outside_hardlinks_allocated_bytes": sum(blocks[key] for key in shared_outside),
        "predicted_reclaim_allocated_bytes": directory_blocks + sum(blocks[key] for key in exclusive),
    }


def df_available(path: Path) -> int:
    info = os.statvfs(path)
    return info.f_bavail * info.f_frsize


def backup_command(items: list[dict]) -> list[str]:
    return [
        "rsync", "-aHzR", "--partial", "--compress-level=1", "--info=progress2", "-s",
        *[item["source_path"] for item in items], f"{HOST}:{REMOTE_ROOT}/",
    ]


def verify_command(item: dict) -> list[str]:
    source = item["source_path"].rstrip("/") + "/"
    remote = item["backup"]["remote_item_path"].rstrip("/") + "/"
    return ["rsync", "-ani", "--delete", "-s", source, f"{HOST}:{remote}"]


def remote_all_exist(items: list[dict]) -> None:
    lines = ["set -e"]
    lines.extend(f"test -d {shlex.quote(item['backup']['remote_item_path'])}" for item in items)
    subprocess.run(["ssh", HOST, "bash", "-s"], input="\n".join(lines) + "\n", text=True, check=True)


def cmd_prepare(execute: bool, confirm: str | None) -> None:
    manifest, items, item_paths = load_batch()
    if not execute or confirm != "PREPARE_EXACT_71":
        raise RuntimeError("prepare requires --execute --confirm PREPARE_EXACT_71")
    if manifest.get("status") not in {"planned", "execution_prepared"}:
        raise RuntimeError(f"cannot prepare manifest in status {manifest.get('status')}")
    sources = [Path(item["source_path"]) for item in items]
    snapshot = formal_snapshot(sources)
    gate = activity_gate(sources)
    reclaim = predicted_reclaim(sources)
    for item in items:
        preflight_item(item)
        source = Path(item["source_path"])
        item["record_status"] = "execution_prepared"
        item["backup_status"] = "pending"
        item["deletion_status"] = "pending"
        item["backup"] = {
            "host": HOST,
            "remote_backup_root": os.fspath(REMOTE_ROOT),
            "remote_project_snapshot_root": os.fspath(remote_item_path(PROJECT_ROOT)),
            "remote_item_path": os.fspath(remote_item_path(source)),
            "original_absolute_path_preserved": True,
            "transfer": "single batch rsync -aHzR --partial --compress-level=1",
            "verification": "per-workspace rsync -ani --delete; stdout must be empty; archive metadata semantics; no size-only",
            "rsync_compress_level": 1,
        }
    manifest.update(
        status="execution_prepared",
        backup_status="pending",
        deletion_status="pending",
        remote_backup_root=os.fspath(REMOTE_ROOT),
        remote_host=HOST,
        original_absolute_paths_preserved=True,
        predicted_reclaim_allocated_bytes=reclaim["predicted_reclaim_allocated_bytes"],
        retained_via_outside_hardlinks_allocated_bytes=reclaim["retained_via_outside_hardlinks_allocated_bytes"],
        activity_check=gate,
        formal_artifact_gate=snapshot,
        runner_capabilities=["plan", "prepare", "preflight_all", "backup_all", "verify_all", "delete_exact_all"],
        runner_explicitly_unsupported=["delete_parent_directories", "move", "rename", "size_only_verification"],
    )
    atomic_json(BATCH_DIR / "formal-artifact-snapshot.json", snapshot)
    atomic_json(BATCH_DIR / "predicted-reclaim.json", reclaim)
    render_records(manifest, items, item_paths)
    log_event("prepared", item_count=71, remote_root=os.fspath(REMOTE_ROOT), **reclaim)
    print(f"PREPARED\t71\t{REMOTE_ROOT}\tpredicted_reclaim={reclaim['predicted_reclaim_allocated_bytes']}")


def cmd_preflight() -> None:
    manifest, items, _ = load_batch()
    if manifest.get("status") not in {"execution_prepared", "transfer_complete_unverified", "backup_verified"}:
        raise RuntimeError(f"manifest status is not preflightable: {manifest.get('status')}")
    for item in items:
        preflight_item(item)
        print(f"PREFLIGHT_OK\t{item['item_id']}\t{item['source_path']}")
    print("PREFLIGHT_COUNT\t71")


def cmd_gate() -> None:
    manifest, items, _ = load_batch()
    sources = [Path(item["source_path"]) for item in items]
    gate = activity_gate(sources)
    current = formal_snapshot(sources)
    frozen = read_json(BATCH_DIR / "formal-artifact-snapshot.json")
    assert_formal_snapshot(current, frozen)
    atomic_json(BATCH_DIR / "gate-latest.json", {"activity": gate, "formal_artifacts": current})
    print(f"GATE_OK\tqueue_hits=0\tprocess_fd_hits=0\toutside_paths={len(current['protected_outside_paths'])}")


def cmd_backup(execute: bool, confirm: str | None) -> None:
    manifest, items, item_paths = load_batch()
    if not execute or confirm != "BACKUP_EXACT_71":
        raise RuntimeError("backup requires --execute --confirm BACKUP_EXACT_71")
    if manifest.get("status") not in {"execution_prepared", "transfer_complete_unverified"}:
        raise RuntimeError(f"cannot back up manifest in status {manifest.get('status')}")
    for item in items:
        preflight_item(item)
    cmd_gate()
    subprocess.run(["ssh", HOST, "mkdir", "-p", "--", os.fspath(REMOTE_ROOT)], check=True)
    command = backup_command(items)
    print("BACKUP_COMMAND\t" + shlex.join(command), flush=True)
    log_event("backup_started", item_count=71, command=shlex.join(command))
    subprocess.run(command, check=True)
    completed = now()
    for item in items:
        item["record_status"] = "transfer_complete_unverified"
        item["backup_status"] = "transferred_unverified"
        item["backup"]["transfer_completed_at"] = completed
    manifest.update(
        status="transfer_complete_unverified",
        backup_status="transferred_unverified",
        transfer_completed_at=completed,
    )
    render_records(manifest, items, item_paths)
    log_event("backup_exit_0", item_count=71)
    print("BACKUP_EXIT_0\t71\tverification_required", flush=True)


def cmd_verify(execute: bool, confirm: str | None) -> None:
    manifest, items, item_paths = load_batch()
    if not execute or confirm != "VERIFY_EXACT_71":
        raise RuntimeError("verify requires --execute --confirm VERIFY_EXACT_71")
    if manifest.get("status") != "transfer_complete_unverified":
        raise RuntimeError(f"cannot verify manifest in status {manifest.get('status')}")
    for item in items:
        preflight_item(item)
    verified: list[tuple[dict, list[str]]] = []
    for index, item in enumerate(items, 1):
        command = verify_command(item)
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            sys.stderr.write(result.stderr)
            raise RuntimeError(f"{item['item_id']}: strict rsync dry-run failed rc={result.returncode}")
        if result.stdout != "":
            sys.stderr.write(result.stdout)
            raise RuntimeError(f"{item['item_id']}: strict rsync dry-run was not empty")
        verified.append((item, command))
        print(f"VERIFY_EMPTY\t{index}/71\t{item['item_id']}", flush=True)
    verified_at = now()
    receipt_dir = BATCH_DIR / "receipts"
    receipt_dir.mkdir(exist_ok=True)
    for item, command in verified:
        receipt = {
            "schema_version": "cleanup-backup-verification-v2",
            "item_id": item["item_id"],
            "source_path": item["source_path"],
            "remote_item_path": item["backup"]["remote_item_path"],
            "status": "backup_verified_archive_tree",
            "verification_command": shlex.join(command),
            "verification_stdout_bytes": 0,
            "size_only_used": False,
            "verified_at": verified_at,
            "deletion_status": "pending",
        }
        atomic_json(receipt_dir / f"{item['item_id']}.backup_verified.json", receipt)
        item["record_status"] = "backup_verified"
        item["backup_status"] = "verified"
        item["backup"]["verified_at"] = verified_at
        item["backup"]["verification_receipt"] = os.fspath(
            receipt_dir / f"{item['item_id']}.backup_verified.json"
        )
    manifest.update(
        status="backup_verified",
        backup_status="verified",
        deletion_status="pending",
        verified_at=verified_at,
        verified_item_count=71,
        verification="71 per-workspace rsync -ani --delete checks returned rc=0 and exactly zero stdout bytes; no size-only",
    )
    render_records(manifest, items, item_paths)
    log_event("backup_verified", item_count=71, empty_dry_runs=71, size_only_used=False)
    print("BACKUP_VERIFIED\t71/71\tstrict_empty_archive_dry_runs", flush=True)


def cmd_delete(execute: bool, confirm: str | None) -> None:
    manifest, items, item_paths = load_batch()
    if not execute or confirm != "DELETE_EXACT_71":
        raise RuntimeError("delete requires --execute --confirm DELETE_EXACT_71")
    if manifest.get("status") not in {"backup_verified", "deleting"}:
        raise RuntimeError(f"cannot delete manifest in status {manifest.get('status')}")
    if manifest.get("backup_status") != "verified" or manifest.get("verified_item_count") != 71:
        raise RuntimeError("all 71 backups are not recorded as verified")
    receipt_dir = BATCH_DIR / "receipts"
    for item in items:
        receipt = read_json(receipt_dir / f"{item['item_id']}.backup_verified.json")
        if receipt.get("status") != "backup_verified_archive_tree" or receipt.get("verification_stdout_bytes") != 0:
            raise RuntimeError(f"{item['item_id']}: invalid verification receipt")
        if receipt.get("size_only_used") is not False:
            raise RuntimeError(f"{item['item_id']}: size-only receipt is forbidden")
        preflight_item(item, allow_deleted=True)
    cmd_gate()
    remote_all_exist(items)
    remaining_sources = [
        Path(item["source_path"]) for item in items if item.get("deletion_status") != "deleted"
    ]
    reclaim = predicted_reclaim(remaining_sources) if remaining_sources else {
        "predicted_reclaim_allocated_bytes": 0,
        "retained_via_outside_hardlinks_allocated_bytes": 0,
    }
    expected_full = manifest.get("predicted_reclaim_allocated_bytes")
    if manifest.get("status") == "backup_verified" and reclaim["predicted_reclaim_allocated_bytes"] != expected_full:
        raise RuntimeError(
            f"predicted reclaim changed: {reclaim['predicted_reclaim_allocated_bytes']} != {expected_full}"
        )
    before = df_available(PROJECT_ROOT)
    manifest.update(
        status="deleting",
        deletion_status="in_progress",
        deletion_started_at=now(),
        local_available_bytes_before_delete=before,
    )
    render_records(manifest, items, item_paths)
    deleted_count = 0
    for item, item_path in zip(items, item_paths):
        source = Path(item["source_path"])
        if item.get("deletion_status") == "deleted":
            deleted_count += 1
            continue
        if source.name != "workspace" or PROJECT_ROOT not in source.parents or source.is_symlink():
            raise RuntimeError(f"delete scope check failed: {source}")
        shutil.rmtree(source)
        if source.exists() or source.is_symlink():
            raise RuntimeError(f"exact workspace still exists after delete: {source}")
        deleted_at = now()
        item["record_status"] = "completed"
        item["deletion_status"] = "deleted"
        item["deleted_at"] = deleted_at
        atomic_json(item_path, item)
        receipt_path = receipt_dir / f"{item['item_id']}.backup_verified.json"
        receipt = read_json(receipt_path)
        receipt["deletion_status"] = "deleted"
        receipt["deleted_at"] = deleted_at
        atomic_json(receipt_path, receipt)
        deleted_count += 1
        print(f"DELETED_EXACT\t{deleted_count}/71\t{source}", flush=True)
    if any(Path(item["source_path"]).exists() or Path(item["source_path"]).is_symlink() for item in items):
        raise RuntimeError("postcheck found a remaining allowlisted workspace")
    remote_all_exist(items)
    current_formal = formal_snapshot([Path(item["source_path"]) for item in items])
    assert_formal_snapshot(current_formal, read_json(BATCH_DIR / "formal-artifact-snapshot.json"))
    after = df_available(PROJECT_ROOT)
    completed_at = now()
    manifest.update(
        status="completed",
        backup_status="verified",
        deletion_status="deleted",
        completed_at=completed_at,
        deleted_item_count=71,
        remote_surviving_item_count=71,
        local_remaining_item_count=0,
        local_available_bytes_after_delete=after,
        observed_available_bytes_delta=after - before,
    )
    render_records(manifest, items, item_paths)
    log_event(
        "completed", item_count=71,
        predicted_reclaim_allocated_bytes=expected_full,
        observed_available_bytes_delta=after - before,
        remote_surviving_item_count=71,
    )
    print(
        f"DELETE_COMPLETE\t71/71\tpredicted={expected_full}\tobserved_df_delta={after - before}",
        flush=True,
    )


def cmd_plan() -> None:
    manifest, items, _ = load_batch()
    print(f"STATUS\t{manifest.get('status')}\t{manifest.get('backup_status')}\t{manifest.get('deletion_status')}")
    print("BACKUP\t" + shlex.join(backup_command(items)))
    print("VERIFY\t71 commands: rsync -ani --delete; stdout must be empty")
    print("DELETE\texactly the 71 manifest source_path directories; basename workspace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "preflight", "gate"):
        command = subparsers.add_parser(name)
        command.add_argument("--all", action="store_true", required=True)
    for name, token in (
        ("prepare", "PREPARE_EXACT_71"),
        ("backup", "BACKUP_EXACT_71"),
        ("verify", "VERIFY_EXACT_71"),
        ("delete", "DELETE_EXACT_71"),
    ):
        command = subparsers.add_parser(name)
        command.add_argument("--all", action="store_true", required=True)
        command.add_argument("--execute", action="store_true")
        command.add_argument("--confirm", help=f"literal token: {token}")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "plan":
            cmd_plan()
        elif args.command == "prepare":
            cmd_prepare(args.execute, args.confirm)
        elif args.command == "preflight":
            cmd_preflight()
        elif args.command == "gate":
            cmd_gate()
        elif args.command == "backup":
            cmd_backup(args.execute, args.confirm)
        elif args.command == "verify":
            cmd_verify(args.execute, args.confirm)
        elif args.command == "delete":
            cmd_delete(args.execute, args.confirm)
        else:
            raise RuntimeError(f"unsupported command: {args.command}")
    except (OSError, RuntimeError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
