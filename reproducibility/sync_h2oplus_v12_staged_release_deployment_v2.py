#!/usr/bin/env python3
"""Route-aware synchronization of the corrected sealed v12-v2 deployment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
MANIFEST_SHA256 = "07789d5c36fae1273ee34ae49d606c1fcec29270d09181222403804dd4990cb9"
BASE_MANIFEST_SHA256 = "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd"
OVERLAY_AGGREGATE_SHA256 = (
    "71cc6d5290076e35974b9916f5bb4bd3435d66f17f727e716f44ea3e3bcd9e97"
)
OVERLAY_ENTRY_COUNT = 62
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v12_staged_release_pilot_v2/workspace",
    )
    parser.add_argument(
        "--local-verification",
        type=Path,
        default=REPRO
        / "h2oplus_v12_staged_release_deployment_verification_v2_local.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_deployment_sync_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deployment = args.deployment_root.expanduser().resolve(strict=True)
    manifest = deployment / "deployment_manifest_v12_v2.json"
    base_manifest = deployment / "deployment_manifest.json"
    payload = json.loads(manifest.read_text())
    local_verification_path = args.local_verification.expanduser().resolve(strict=True)
    local_verification = json.loads(local_verification_path.read_text())
    if (
        sha256(manifest) != MANIFEST_SHA256
        or sha256(base_manifest) != BASE_MANIFEST_SHA256
        or payload.get("overlay_aggregate_sha256") != OVERLAY_AGGREGATE_SHA256
        or payload.get("overlay_entry_count") != OVERLAY_ENTRY_COUNT
        or local_verification.get("passed") is not True
        or local_verification.get("deployment_manifest_sha256") != MANIFEST_SHA256
        or local_verification.get("base_entries") != 522
        or local_verification.get("overlay_entries") != OVERLAY_ENTRY_COUNT
        or local_verification.get("local_training_or_sumo_executed") is not False
    ):
        raise ValueError("local v12-v2 deployment identity or verification drifted")

    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    sys.path.insert(0, str(scheduler_root / "skill"))
    scheduler = importlib.import_module("scheduler")
    rows = []
    for node in NODES:
        target = scheduler._ssh_target_for_node(node)
        shell = scheduler._ssh_rsync_shell_for_node(node)
        remote = str(deployment)
        rc, _, err = scheduler.run_on(
            node,
            f"mkdir -p {shlex.quote(str(deployment.parent))}",
            timeout=60,
            check=False,
        )
        if rc != 0:
            raise RuntimeError(f"remote v12-v2 deployment parent failed on {node}: {err}")
        base_command = [
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
        completed = subprocess.run(
            base_command, text=True, capture_output=True, check=False
        )
        dry_run = subprocess.run(
            [
                *base_command[:2],
                "--dry-run",
                "--itemize-changes",
                *base_command[2:],
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        check_command = (
            f"sha256sum {shlex.quote(remote + '/deployment_manifest_v12_v2.json')} "
            f"{shlex.quote(remote + '/deployment_manifest.json')}"
        )
        check_rc, check_out, check_err = scheduler.run_on(
            node, check_command, timeout=60, check=False
        )
        remote_hashes = [
            line.split()[0] for line in check_out.splitlines() if line.strip()
        ]
        passed = (
            completed.returncode == 0
            and dry_run.returncode == 0
            and not dry_run.stdout.strip()
            and check_rc == 0
            and remote_hashes == [MANIFEST_SHA256, BASE_MANIFEST_SHA256]
        )
        rows.append(
            {
                "node": node,
                "passed": passed,
                "deployment_root": remote,
                "returncode": completed.returncode,
                "stdout": completed.stdout.strip(),
                "stderr": completed.stderr.strip(),
                "checksum_dry_run_returncode": dry_run.returncode,
                "checksum_dry_run_stdout": dry_run.stdout.strip(),
                "checksum_dry_run_stderr": dry_run.stderr.strip(),
                "remote_manifest_check_returncode": check_rc,
                "remote_manifest_check_stdout": check_out.strip(),
                "remote_manifest_check_stderr": check_err.strip(),
            }
        )

    record = {
        "schema_version": "transit-h2oplus-v12-staged-release-deployment-sync-v2",
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
            "path": str(local_verification_path),
            "sha256": sha256(local_verification_path),
            "bytes": local_verification_path.stat().st_size,
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
