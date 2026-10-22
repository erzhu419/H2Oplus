#!/usr/bin/env python3
"""Route-aware synchronization of the sealed v10 evaluation deployment."""

from __future__ import annotations

import argparse
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
MANIFEST_SHA256 = "ebb36040134679458f7ee1e818e8a3631c6929916b95e4b627542defc9fa2d4c"
AGGREGATE_SHA256 = "2da927fe95618ddfc28e0e2d795c69c379760eee770f77d323e38fd829ad23e8"


def sha256(path: Path) -> str:
    import hashlib

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
    parser.add_argument("--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm")
    parser.add_argument(
        "--deployment-root",
        type=Path,
        default=REPRO / "deployments/h2oplus_v10_delayed_evaluation_v1/workspace",
    )
    parser.add_argument("--node", choices=("jtl110gpu",), default="jtl110gpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v10_delayed_evaluation_deployment_sync_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    deployment = args.deployment_root.expanduser().resolve(strict=True)
    manifest = deployment / "deployment_manifest.json"
    payload = json.loads(manifest.read_text())
    if (
        sha256(manifest) != MANIFEST_SHA256
        or payload.get("aggregate_sha256") != AGGREGATE_SHA256
        or payload.get("entry_count") != 517
    ):
        raise ValueError("local v10 deployment identity drifted")
    sys.path.insert(0, str(scheduler_root))
    scheduler = importlib.import_module("skill.scheduler")
    target = scheduler._ssh_target_for_node(args.node)
    shell = scheduler._ssh_rsync_shell_for_node(args.node)
    remote = str(deployment)
    rc, _, err = scheduler.run_on(
        args.node,
        f"mkdir -p {shlex.quote(str(deployment.parent))}",
        timeout=60,
        check=False,
    )
    if rc != 0:
        raise RuntimeError(f"remote deployment parent creation failed: {err}")
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
        "schema_version": "transit-h2oplus-v10-delayed-evaluation-deployment-sync-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "passed": completed.returncode == 0,
        "transport": "scheduleurm-route-aware-rsync-checksum",
        "deployment_root": str(deployment),
        "manifest_sha256": MANIFEST_SHA256,
        "aggregate_sha256": AGGREGATE_SHA256,
        "entry_count": 517,
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
