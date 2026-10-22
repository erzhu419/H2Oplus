#!/usr/bin/env python3
"""Synchronize only the registered v20 formal result partition."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
POLICY = REPRO / "h2oplus_minimal_remote_sync_policy_v1.json"
POLICY_SHA256 = "20b313b7b8cc13a0a8df7c2a137a4b0713c621a33ef8e2e314fb69d03de1a531"
EXPECTED_RESULTS = 1350
EXPECTED_MANIFESTS = 10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument("--node", choices=("jtl110gpu",), default="jtl110gpu")
    parser.add_argument("--remote-root", type=PurePosixPath, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--include-bounded-logs", action="store_true")
    parser.add_argument("--max-log-bytes", type=int, default=2 * 1024 * 1024)
    return parser.parse_args()


def load_scheduler(root: Path):
    resolved = root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(resolved))
    return importlib.import_module("skill.scheduler")


def validate_authorization(path: Path, expected_sha256: str) -> dict:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected_sha256:
        raise ValueError("formal execution authorization identity drifted")
    payload = json.loads(resolved.read_text())
    scope = payload.get("authorization_scope", {})
    if (
        scope.get("formal_confirmation_evaluation_authorized") is not True
        or scope.get("formal_confirmation_submission_authorized") is not True
        or scope.get("formal_held_out_outcome_access_authorized") is not True
        or scope.get("new_training_authorized") is not False
        or scope.get("checkpoint_selection_by_behavior_authorized") is not False
    ):
        raise ValueError("formal result synchronization is not authorized")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": expected_sha256,
    }


def parse_inventory(raw: str) -> dict[str, int]:
    records = {}
    for line in raw.splitlines():
        if not line:
            continue
        try:
            path, size = line.rsplit("\t", 1)
        except ValueError as exc:
            raise ValueError(f"malformed remote inventory row: {line!r}") from exc
        if path.startswith("/") or path.startswith("../") or "/../" in path:
            raise ValueError(f"unsafe remote inventory path: {path!r}")
        records[path] = int(size)
    return records


def remote_inventory(scheduler, node: str, root: PurePosixPath) -> dict[str, int]:
    command = (
        f"test -d {shlex.quote(str(root))} && "
        f"cd {shlex.quote(str(root))} && "
        "find . -type f -printf '%P\\t%s\\n' | LC_ALL=C sort"
    )
    rc, stdout, stderr = scheduler.run_on(
        node, command, timeout=120, check=False
    )
    if rc != 0:
        raise RuntimeError(f"remote formal inventory failed: {stderr or stdout}")
    return parse_inventory(stdout)


def is_result(path: str) -> bool:
    name = PurePosixPath(path).name
    return name.startswith("sumo_seed") and name.endswith(".json")


def select_files(
    inventory: dict[str, int], include_logs: bool, max_log_bytes: int
) -> dict[str, int]:
    selected = {}
    for path, size in inventory.items():
        name = PurePosixPath(path).name
        allowed = (
            is_result(path)
            or name == "matrix_manifest.json"
            or name in {"host_execution.json", "host_summary.json"}
            or name.endswith(".status")
        )
        if name.endswith(".log"):
            allowed = include_logs and size <= max_log_bytes
        if allowed:
            selected[path] = size
    results = sum(is_result(path) for path in selected)
    manifests = sum(
        PurePosixPath(path).name == "matrix_manifest.json" for path in selected
    )
    if results != EXPECTED_RESULTS or manifests != EXPECTED_MANIFESTS:
        raise ValueError(
            "formal minimal-sync inventory is incomplete: "
            f"results={results}, manifests={manifests}"
        )
    return dict(sorted(selected.items()))


def local_inventory(root: Path) -> dict[str, int]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite formal sync record: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(encoded)
    os.replace(partial, path)


def main() -> int:
    args = parse_args()
    if sha256(POLICY.resolve(strict=True)) != POLICY_SHA256:
        raise ValueError("minimal remote-sync policy drifted")
    if args.max_log_bytes < 0:
        raise ValueError("max-log-bytes must be nonnegative")
    authorization = validate_authorization(
        args.execution_authorization, args.expected_authorization_sha256
    )
    scheduler = load_scheduler(args.scheduler_root)
    remote = remote_inventory(scheduler, args.node, args.remote_root)
    selected = select_files(
        remote, args.include_bounded_logs, args.max_log_bytes
    )
    local_root = args.local_root.expanduser().resolve()
    existing = local_inventory(local_root)
    unexpected_existing = sorted(set(existing) - set(selected))
    if unexpected_existing:
        raise ValueError(
            "local formal partition contains non-allowlisted files: "
            + ", ".join(unexpected_existing[:10])
        )

    local_root.mkdir(parents=True, exist_ok=True)
    target = scheduler._ssh_target_for_node(args.node)
    shell = scheduler._ssh_rsync_shell_for_node(args.node)
    with tempfile.NamedTemporaryFile("w", delete=False) as handle:
        files_from = Path(handle.name)
        handle.write("".join(f"{path}\n" for path in selected))
    try:
        command = [
            "rsync",
            "-a",
            "--checksum",
            "--partial",
            "--delay-updates",
            "--omit-dir-times",
            "--relative",
            f"--files-from={files_from}",
            "-e",
            shell,
            f"{target}:{str(args.remote_root).rstrip('/')}/",
            str(local_root) + "/",
        ]
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"formal allowlist rsync failed: {completed.stderr.strip()}"
            )
        verification = subprocess.run(
            [
                *command[:2],
                "--dry-run",
                "--itemize-changes",
                *command[2:],
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if verification.returncode != 0 or verification.stdout.strip():
            raise ValueError(
                "formal allowlist rsync verification did not converge: "
                f"{verification.stdout.strip()} {verification.stderr.strip()}"
            )
    finally:
        files_from.unlink(missing_ok=True)

    observed = local_inventory(local_root)
    if observed != selected:
        raise ValueError("formal minimal partition differs from remote allowlist")
    record = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-minimal-sync-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "task_id": args.task_id,
        "node": args.node,
        "remote_root": str(args.remote_root),
        "local_root": str(local_root),
        "execution_authorization": authorization,
        "minimal_sync_policy": {
            "path": str(POLICY),
            "sha256": POLICY_SHA256,
        },
        "allowlist_only": True,
        "selected_file_count": len(selected),
        "selected_total_bytes": sum(selected.values()),
        "scientific_result_json_count": EXPECTED_RESULTS,
        "matrix_manifest_count": EXPECTED_MANIFESTS,
        "bounded_logs_included": args.include_bounded_logs,
        "max_log_bytes": args.max_log_bytes if args.include_bounded_logs else 0,
        "checkpoint_optimizer_replay_or_workspace_files_synchronized": False,
        "formal_outcome_files_synchronized": True,
        "scientific_numeric_outcomes_parsed_by_sync": False,
        "passed": True,
    }
    atomic_json(args.record.expanduser().resolve(), record)
    print(json.dumps(record, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
