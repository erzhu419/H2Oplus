#!/usr/bin/env python3
"""Route-aware synchronization of the sealed v16 evaluation deployment."""

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
PARENT_MANIFEST_SHA256 = (
    "fe510525503e3e44cc83b33916be308d6d1fea2c39c299c7fc4e399a33891d6c"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite sync record: {path}")
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
        default=REPRO / "deployments/h2oplus_v16_headway_reserve_evaluation_v1/workspace",
    )
    parser.add_argument("--node", choices=("jtl110gpu",), default="jtl110gpu")
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-aggregate-sha256", required=True)
    parser.add_argument("--expected-entry-count", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v16_headway_reserve_evaluation_deployment_sync_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    deployment = args.deployment_root.expanduser().resolve(strict=True)
    manifest = deployment / "deployment_manifest.json"
    payload = json.loads(manifest.read_text())
    if (
        sha256(manifest) != args.expected_manifest_sha256
        or payload.get("schema_version")
        != "transit-h2oplus-v16-headway-reserve-evaluation-deployment-manifest-v1"
        or payload.get("aggregate_sha256") != args.expected_aggregate_sha256
        or payload.get("entry_count") != args.expected_entry_count
        or payload.get("physical_behavior_count") != 15
        or payload.get("logical_method_count") != 15
        or payload.get("new_actor_file_count") != 0
        or payload.get("expected_new_remote_sumo_episodes") != 135
        or payload.get("critical_matrix_source_file_count") != 51
        or payload.get("scientific_source_file_count") != 14
        or payload.get("parent_v15_artifact_integrity_passed") is not True
        or payload.get("parent_v14_training_stability_passed") is not True
        or payload.get("prior_v14_and_v15_development_outcomes_used_for_design")
        is not True
        or payload.get("formal_held_out_outcomes_included") is not False
        or payload.get("formal_confirmation_stage_authorized_or_submitted") is not False
    ):
        raise ValueError("local v16 deployment identity drifted")

    sys.path.insert(0, str(scheduler_root))
    scheduler = importlib.import_module("skill.scheduler")
    target = scheduler._ssh_target_for_node(args.node)
    shell = scheduler._ssh_rsync_shell_for_node(args.node)
    remote = str(deployment)
    remote_parent = str(
        REPRO / "deployments/h2oplus_v15_hold_calibration_evaluation_v1/workspace"
    )
    clone_command = " ".join(
        (
            "if",
            "test",
            "-d",
            shlex.quote(remote + "/H2Oplus"),
            ";",
            "then",
            ":",
            ";",
            "else",
            "test",
            f'"$(sha256sum {shlex.quote(remote_parent + "/deployment_manifest.json")} | awk \'{{print $1}}\')"',
            "=",
            shlex.quote(PARENT_MANIFEST_SHA256),
            "&&",
            "mkdir",
            "-p",
            shlex.quote(str(deployment.parent)),
            "&&",
            "cp",
            "-al",
            shlex.quote(remote_parent),
            shlex.quote(remote),
            ";",
            "fi",
        )
    )
    rc, _, err = scheduler.run_on(
        args.node,
        clone_command,
        timeout=60,
        check=False,
    )
    if rc != 0:
        raise RuntimeError(f"remote v15-to-v16 deployment clone failed: {err}")
    command = [
        "rsync",
        "-a",
        "--checksum",
        "--partial",
        "--delay-updates",
        "--omit-dir-times",
        "-e",
        shell,
        str(deployment) + "/",
        f"{target}:{remote}/",
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    record = {
        "schema_version": (
            "transit-h2oplus-v16-headway-reserve-evaluation-deployment-sync-v1"
        ),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "passed": completed.returncode == 0,
        "transport": "scheduleurm-parent-hardlink-clone-plus-route-aware-rsync-checksum",
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "deployment_root": str(deployment),
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "expected_new_remote_sumo_episodes": 135,
        "formal_held_out_outcomes_read": False,
        "local_or_remote_sumo_simulation_executed": False,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
