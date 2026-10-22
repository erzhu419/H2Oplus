#!/usr/bin/env python3
"""Route-aware synchronization of the sealed v18-v3 deployment."""

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
    "7f6595be6c1f239e1c5c16f85aad7667ed9e87841c02c8eae9b15d0439004600"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v18-v3 sync record: {path}")
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
        default=REPRO
        / "deployments/h2oplus_v18_empty_tail_reserve_evaluation_v3/workspace",
    )
    parser.add_argument("--node", choices=("jtl110gpu",), default="jtl110gpu")
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-aggregate-sha256", required=True)
    parser.add_argument("--expected-entry-count", type=int, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO
        / "h2oplus_v18_empty_tail_reserve_evaluation_deployment_sync_v3.json",
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
        != "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-manifest-v3"
        or payload.get("aggregate_sha256") != args.expected_aggregate_sha256
        or payload.get("entry_count") != args.expected_entry_count
        or payload.get("physical_behavior_count") != 15
        or payload.get("logical_method_count") != 15
        or payload.get("expected_new_remote_sumo_episodes") != 135
        or payload.get("critical_matrix_source_file_count") != 59
        or payload.get("scientific_source_file_count") != 24
        or payload.get("namespace_binding_correction_only") is not True
        or payload.get("scientific_behavior_unchanged_from_v18_v1_and_v2")
        is not True
        or payload.get("v18_v1_result_count") != 0
        or payload.get("v18_v2_result_count") != 0
        or payload.get("v18_v1_or_v2_results_reused") is not False
        or payload.get("formal_held_out_outcomes_included") is not False
    ):
        raise ValueError("local v18-v3 deployment identity drifted")

    sys.path.insert(0, str(scheduler_root))
    scheduler = importlib.import_module("skill.scheduler")
    target = scheduler._ssh_target_for_node(args.node)
    shell = scheduler._ssh_rsync_shell_for_node(args.node)
    remote = str(deployment)
    remote_parent = str(
        REPRO
        / "deployments/h2oplus_v18_empty_tail_reserve_evaluation_v2/workspace"
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
    rc, _, err = scheduler.run_on(args.node, clone_command, timeout=60, check=False)
    if rc != 0:
        raise RuntimeError(f"remote v18-v2-to-v3 clone failed: {err}")
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
            "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-sync-v3"
        ),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "passed": completed.returncode == 0,
        "transport": "scheduleurm-v18-v2-hardlink-clone-plus-rsync-checksum",
        "parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "deployment_root": str(deployment),
        "manifest_sha256": args.expected_manifest_sha256,
        "aggregate_sha256": args.expected_aggregate_sha256,
        "entry_count": args.expected_entry_count,
        "expected_new_remote_sumo_episodes": 135,
        "v18_v1_or_v2_results_reused": False,
        "formal_held_out_outcomes_read": False,
        "local_or_remote_sumo_simulation_executed": False,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    atomic_json(args.output.expanduser().resolve(), record)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
