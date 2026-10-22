#!/usr/bin/env python3
"""Verify the corrected v12-v2 training deployment on both remote GPU hosts."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SCHEMA = "transit-h2oplus-v12-staged-release-remote-verification-v2"
NODES = ("jtl110gpu", "jtl110gpu2")
EXPECTED_MANIFEST_SHA256 = (
    "07789d5c36fae1273ee34ae49d606c1fcec29270d09181222403804dd4990cb9"
)
EXPECTED_BASE_MANIFEST_SHA256 = (
    "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd"
)
EXPECTED_LOCAL_VERIFICATION_SHA256 = (
    "d1a5d1142081e6cb2fe5c4184e5fe41061d691455865396b9a43057cf3c5bf5b"
)
EXPECTED_SYNC_SHA256 = (
    "10bd31aa3bbb0bc61d6e4bd4abac1efb4d0f84cf58495979ba3f315b8adc83cf"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


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
        "--sync-record",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_deployment_sync_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO
        / "h2oplus_v12_staged_release_deployment_verification_v2_remote.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    local_verification = args.local_verification.expanduser().resolve(strict=True)
    sync_record = args.sync_record.expanduser().resolve(strict=True)
    if sha256(local_verification) != EXPECTED_LOCAL_VERIFICATION_SHA256:
        raise ValueError("v12-v2 local verification drifted")
    if sha256(sync_record) != EXPECTED_SYNC_SHA256:
        raise ValueError("v12-v2 deployment sync record drifted")

    sys.path.insert(0, str(scheduler_root / "skill"))
    scheduler = importlib.import_module("scheduler")
    remote_root = str(deployment_root)
    environment = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": (
            f"{remote_root}/reproducibility/toolchain/"
            "libxcrypt-4.4.36-hd590300_1/include"
        ),
        "PATH": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin",
        "PYTHONPATH": remote_root,
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    remote_code = f'''
import json
import os
import sys
from pathlib import Path

root = Path({remote_root!r}).resolve(strict=True)
sys.path.insert(0, str(root))
from reproducibility.verify_h2oplus_v12_staged_release_deployment_v2 import verify

verification = verify(root)
print(json.dumps({{
    "hostname": os.uname().nodename,
    "environment": {{key: os.environ.get(key) for key in {tuple(environment)!r}}},
    "verification": verification,
}}, sort_keys=True))
'''
    prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in environment.items()
    )
    command = (
        f"cd {shlex.quote(remote_root)} && {prefix} "
        "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python "
        f"-c {shlex.quote(remote_code)}"
    )
    rows = []
    for node in NODES:
        returncode, stdout, stderr = scheduler.run_on(
            node, command, timeout=900, check=False
        )
        observation: dict[str, Any] = {}
        parse_error = None
        if returncode == 0:
            try:
                observation = json.loads(stdout.strip())
            except Exception as error:
                parse_error = str(error)
        verification = observation.get("verification", {})
        passed = bool(
            returncode == 0
            and parse_error is None
            and observation.get("environment") == environment
            and verification.get("passed") is True
            and verification.get("deployment_manifest_sha256")
            == EXPECTED_MANIFEST_SHA256
            and verification.get("base_v7_deployment_manifest_sha256")
            == EXPECTED_BASE_MANIFEST_SHA256
            and verification.get("base_entries") == 522
            and verification.get("overlay_entries") == 62
            and verification.get("training_jobs") == 30
            and verification.get("prefix_jobs") == 5
            and verification.get("branch_jobs") == 25
            and verification.get("offline_initializers") == 5
            and verification.get("development_evaluation_submitted") is False
            and verification.get("formal_held_out_outcomes_read") is False
            and verification.get("local_training_or_sumo_executed") is False
        )
        rows.append(
            {
                "node": node,
                "passed": passed,
                "returncode": returncode,
                "parse_error": parse_error,
                "observation": observation,
                "stderr": stderr.strip(),
            }
        )

    payload = {
        "schema_version": SCHEMA,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": all(row["passed"] for row in rows),
        "deployment_root": remote_root,
        "deployment_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "base_deployment_manifest_sha256": EXPECTED_BASE_MANIFEST_SHA256,
        "local_verification": descriptor(local_verification),
        "deployment_sync": descriptor(sync_record),
        "required_environment": environment,
        "nodes": rows,
        "formal_held_out_outcomes_read": False,
        "local_or_remote_sumo_simulation_executed": False,
        "training_executed": False,
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
