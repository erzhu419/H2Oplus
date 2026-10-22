#!/usr/bin/env python3
"""Verify the frozen v14 deployment on both GPU hosts without training."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
SCHEMA = "transit-h2oplus-v14-terminal-taper-remote-verification-v1"
NODES = ("jtl110gpu", "jtl110gpu2")
MANIFEST_SHA256 = "1df1873929f9d6273201d2266810ff5ab3d4fcb14f8db5fe55eaa91818ae569b"
BASE_MANIFEST_SHA256 = "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd"
DIAGNOSIS_SHA256 = "1f00a72052df2c12bbbbf5f20da67a5242897eea310e463afc4858447d1e8710"
LOCAL_VERIFICATION_SHA256 = "df71c246d11f1cc734e57412e91450e297637b8200f31f380c1412c608d1cf90"
SYNC_SHA256 = "b8ee3fd884ddd2a68ecb8b1d1d9e3b52fa55fb4c05b52d2523d5c43df03a8572"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_on_bounded(scheduler, node: str, command: str):
    attempts = []
    for attempt in range(1, 4):
        returncode, stdout, stderr = scheduler.run_on(
            node, command, timeout=900, check=False
        )
        attempts.append(
            {
                "attempt": attempt,
                "returncode": returncode,
                "stderr": stderr.strip(),
            }
        )
        if returncode == 0:
            break
        if attempt < 3:
            time.sleep(5 * attempt)
    return returncode, stdout, stderr, attempts


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
        "--sync-record",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_deployment_sync_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO
        / "h2oplus_v14_terminal_taper_deployment_verification_v1_remote.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    deployment = args.deployment_root.expanduser().resolve(strict=True)
    local_path = args.local_verification.expanduser().resolve(strict=True)
    sync_path = args.sync_record.expanduser().resolve(strict=True)
    if sha256(local_path) != LOCAL_VERIFICATION_SHA256:
        raise ValueError("v14 local verification drifted")
    if sha256(sync_path) != SYNC_SHA256:
        raise ValueError("v14 deployment sync record drifted")

    sys.path.insert(0, str(scheduler_root / "skill"))
    scheduler = importlib.import_module("scheduler")
    remote_root = str(deployment)
    environment = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": (
            "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
            "v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/"
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
from reproducibility.verify_h2oplus_v14_terminal_taper_deployment import verify

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
        returncode, stdout, stderr, attempts = run_on_bounded(
            scheduler, node, command
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
            and verification.get("deployment_manifest_sha256") == MANIFEST_SHA256
            and verification.get("base_deployment_manifest_sha256")
            == BASE_MANIFEST_SHA256
            and verification.get("diagnosis_sha256") == DIAGNOSIS_SHA256
            and verification.get("base_entries") == 522
            and verification.get("overlay_entries") == 92
            and verification.get("verified_jobs") == 15
            and verification.get("verified_source_handoffs") == 5
            and verification.get("verified_initializers") == 5
            and verification.get("frozen_base_bytecode_files") == 1
            and verification.get("runtime_source_path_count") == 16
            and verification.get("runtime_source_map_matches_frozen_admissions")
            is True
            and verification.get("sealed_v12_files") == 8
            and verification.get("sealed_v13_files") == 10
            and verification.get("predesignated_candidate") == "terminal-quarter"
            and verification.get("formal_held_out_outcomes_read") is False
            and verification.get("local_training_or_sumo_executed") is False
        )
        rows.append(
            {
                "node": node,
                "passed": passed,
                "returncode": returncode,
                "attempts": attempts,
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
        "deployment_manifest_sha256": MANIFEST_SHA256,
        "base_deployment_manifest_sha256": BASE_MANIFEST_SHA256,
        "diagnosis_sha256": DIAGNOSIS_SHA256,
        "local_verification": descriptor(local_path),
        "deployment_sync": descriptor(sync_path),
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
