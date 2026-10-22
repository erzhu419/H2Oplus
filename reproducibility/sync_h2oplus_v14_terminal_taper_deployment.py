#!/usr/bin/env python3
"""Route-aware synchronization of the frozen v14 training deployment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
MANIFEST_SHA256 = "1df1873929f9d6273201d2266810ff5ab3d4fcb14f8db5fe55eaa91818ae569b"
BASE_MANIFEST_SHA256 = "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd"
OVERLAY_AGGREGATE_SHA256 = "4a500959403e823c5775044912b5d0a10a467afc16df50d45dda61f84a548483"
LOCAL_VERIFICATION_SHA256 = "df71c246d11f1cc734e57412e91450e297637b8200f31f380c1412c608d1cf90"
LOCAL_PREFLIGHT_SHA256 = "aa67f921ef7e855e9386a168acb3f57e8eac2d16996c9463faa5acced34080a8"
OVERLAY_ENTRY_COUNT = 92
NODES = ("jtl110gpu", "jtl110gpu2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_on_bounded(scheduler, node: str, command: str, *, timeout: int):
    attempts = []
    for attempt in range(1, 4):
        rc, stdout, stderr = scheduler.run_on(
            node, command, timeout=timeout, check=False
        )
        attempts.append(
            {
                "attempt": attempt,
                "returncode": rc,
                "stdout": stdout.strip(),
                "stderr": stderr.strip(),
            }
        )
        if rc == 0:
            break
        if attempt < 3:
            time.sleep(5 * attempt)
    return rc, stdout, stderr, attempts


def run_process_bounded(command: list[str]):
    attempts = []
    for attempt in range(1, 4):
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        attempts.append(
            {
                "attempt": attempt,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
            }
        )
        if completed.returncode == 0:
            break
        if attempt < 3:
            time.sleep(5 * attempt)
    return completed, attempts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v14_terminal_taper_pilot_v1/workspace",
    )
    parser.add_argument(
        "--local-verification",
        type=Path,
        default=REPRO
        / "h2oplus_v14_terminal_taper_deployment_verification_v1_local.json",
    )
    parser.add_argument(
        "--local-preflight",
        type=Path,
        default=REPRO
        / "h2oplus_v14_terminal_taper_runtime_preflight_v1_local.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_deployment_sync_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deployment = args.deployment_root.expanduser().resolve(strict=True)
    manifest = deployment / "deployment_manifest_v14_v1.json"
    base_manifest = deployment / "deployment_manifest.json"
    local_path = args.local_verification.expanduser().resolve(strict=True)
    preflight_path = args.local_preflight.expanduser().resolve(strict=True)
    payload = json.loads(manifest.read_text())
    local = json.loads(local_path.read_text())
    preflight = json.loads(preflight_path.read_text())
    if (
        sha256(manifest) != MANIFEST_SHA256
        or sha256(base_manifest) != BASE_MANIFEST_SHA256
        or payload.get("overlay_aggregate_sha256") != OVERLAY_AGGREGATE_SHA256
        or payload.get("overlay_entry_count") != OVERLAY_ENTRY_COUNT
        or sha256(local_path) != LOCAL_VERIFICATION_SHA256
        or sha256(preflight_path) != LOCAL_PREFLIGHT_SHA256
        or local.get("passed") is not True
        or local.get("deployment_manifest_sha256") != MANIFEST_SHA256
        or local.get("verified_jobs") != 15
        or local.get("verified_source_handoffs") != 5
        or local.get("local_training_or_sumo_executed") is not False
        or preflight.get("passed") is not True
        or preflight.get("mode") != "local"
        or preflight.get("verified_jobs") != 15
        or preflight.get("training_executed") is not False
        or preflight.get("sumo_executed") is not False
        or preflight.get("output_materialized") is not False
        or not all(row.get("passed") is True for row in preflight.get("jobs", []))
    ):
        raise ValueError("local v14 deployment identity or preflight drifted")

    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(scheduler_root / "skill"))
    scheduler = importlib.import_module("scheduler")
    rows = []
    for node in NODES:
        target = scheduler._ssh_target_for_node(node)
        shell = scheduler._ssh_rsync_shell_for_node(node)
        remote = str(deployment)
        rc, _, err, parent_attempts = run_on_bounded(
            scheduler,
            node,
            f"mkdir -p {shlex.quote(str(deployment.parent))}",
            timeout=60,
        )
        if rc != 0:
            raise RuntimeError(f"remote v14 deployment parent failed on {node}: {err}")
        command = [
            "rsync",
            "-a",
            "--checksum",
            "--partial",
            "--delay-updates",
            "--delete-delay",
            "--omit-dir-times",
            "-e",
            shell,
            str(deployment) + "/",
            f"{target}:{remote}/",
        ]
        completed, transfer_attempts = run_process_bounded(command)
        dry_run = subprocess.run(
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
        check_rc, check_out, check_err, check_attempts = run_on_bounded(
            scheduler,
            node,
            "sha256sum "
            f"{shlex.quote(remote + '/deployment_manifest_v14_v1.json')} "
            f"{shlex.quote(remote + '/deployment_manifest.json')}",
            timeout=60,
        )
        hashes = [line.split()[0] for line in check_out.splitlines() if line.strip()]
        passed = (
            completed.returncode == 0
            and dry_run.returncode == 0
            and not dry_run.stdout.strip()
            and check_rc == 0
            and hashes == [MANIFEST_SHA256, BASE_MANIFEST_SHA256]
        )
        rows.append(
            {
                "node": node,
                "passed": passed,
                "deployment_root": remote,
                "remote_parent_attempts": parent_attempts,
                "transfer_attempts": transfer_attempts,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "checksum_dry_run_returncode": dry_run.returncode,
                "checksum_dry_run_stdout": dry_run.stdout.strip(),
                "checksum_dry_run_stderr": dry_run.stderr.strip(),
                "remote_manifest_check_returncode": check_rc,
                "remote_manifest_check_attempts": check_attempts,
                "remote_manifest_check_stdout": check_out.strip(),
                "remote_manifest_check_stderr": check_err.strip(),
            }
        )

    record = {
        "schema_version": "transit-h2oplus-v14-terminal-taper-deployment-sync-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(row["passed"] for row in rows),
        "transport": "scheduleurm-route-aware-rsync-checksum",
        "checksum_mode": "post-transfer-rsync-checksum-dry-run-and-remote-manifest-sha256",
        "deployment_root": str(deployment),
        "deployment_manifest_sha256": MANIFEST_SHA256,
        "base_deployment_manifest_sha256": BASE_MANIFEST_SHA256,
        "overlay_aggregate_sha256": OVERLAY_AGGREGATE_SHA256,
        "overlay_entry_count": OVERLAY_ENTRY_COUNT,
        "local_verification": {
            "path": str(local_path),
            "sha256": sha256(local_path),
            "bytes": local_path.stat().st_size,
        },
        "local_runtime_preflight": {
            "path": str(preflight_path),
            "sha256": sha256(preflight_path),
            "bytes": preflight_path.stat().st_size,
        },
        "nodes": rows,
        "formal_held_out_outcomes_read": False,
        "local_or_remote_sumo_simulation_executed": False,
        "training_executed": False,
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
