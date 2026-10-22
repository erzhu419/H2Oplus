#!/usr/bin/env python3
"""Execute every frozen v13 v2 admission path without training or SUMO."""

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
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
DEPLOYMENT_ROOT = REPRO / "deployments/h2oplus_v13_proximal_release_pilot_v2"
WORKSPACE = DEPLOYMENT_ROOT / "workspace"
CONTROL = WORKSPACE / "reproducibility/h2oplus_v13_control_v2"
PLAN = CONTROL / "training_plan.json"
PROTOCOL = CONTROL / "protocol.json"
CORRECTION = CONTROL / "pretraining_correction.json"
DEPLOYMENT_MANIFEST = WORKSPACE / "deployment_manifest_v13_v2.json"
BASE_DEPLOYMENT_MANIFEST = WORKSPACE / "deployment_manifest.json"
LOCAL_VERIFICATION = (
    REPRO / "h2oplus_v13_proximal_release_deployment_verification_v2_local.json"
)
PYTHON = "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python"
PREFLIGHT_SCHEMA = "transit-h2oplus-v13-runtime-admission-preflight-v2"
RECORD_SCHEMA = "transit-h2oplus-v13-proximal-release-preflight-matrix-v2"
EXPECTED_HASHES = {
    "training_plan": "e67989fd379ea74fc2c3d3a6ce749b2f1b3bd3134724e0b2628c938dd777ab63",
    "protocol": "8f330bc7a55b35c9c10ae03434ddbecd30528e4529b01e734dbbd6c44bdd5f12",
    "pretraining_correction": "d5a0682f4511c511caeeb345494949b5325e8d73f1c69c0eed98e8f4e11c5b48",
    "deployment_manifest": "ce401b074dd83a5bd5f892d314217967a24c9e4c55b4d714784234371f195cb7",
    "base_deployment_manifest": "db1bc95b9763a202cb3398abc550aa62b49d7c548eac034021c1ba21697f89cd",
    "local_verification": "dedf39fa8796b6a7d9b3e239ad1fa204324bb7800bc30562c5cc20699900db07",
}
NODE_BY_SEED = {
    6201: "jtl110gpu",
    6202: "jtl110gpu2",
    6203: "jtl110gpu",
    6204: "jtl110gpu2",
    6205: "jtl110gpu",
}


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
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local", "remote"), required=True)
    parser.add_argument(
        "--scheduler-root", type=Path, default=ROOT.parent.parent / "scheduleurm"
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def required_environment() -> dict[str, str]:
    return {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": (
            "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
            "v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/"
            "libxcrypt-4.4.36-hd590300_1/include"
        ),
        "PATH": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin",
        "PYTHONPATH": str(WORKSPACE),
        "TMPDIR": "/home/erzhu419/.cache/h2oplus-v7/tmp",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "H2OPLUS_V13_WORKSPACE_ROOT": str(WORKSPACE),
        "H2OPLUS_V13_PROTOCOL": str(PROTOCOL),
        "H2OPLUS_V13_PROTOCOL_SHA256": EXPECTED_HASHES["protocol"],
        "H2OPLUS_V13_BASE_DEPLOYMENT_MANIFEST": str(BASE_DEPLOYMENT_MANIFEST),
        "H2OPLUS_V13_BASE_DEPLOYMENT_MANIFEST_SHA256": EXPECTED_HASHES[
            "base_deployment_manifest"
        ],
    }


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


def remote_absence(scheduler, node: str, output_root: Path) -> dict[str, Any]:
    command = f"test ! -e {shlex.quote(str(output_root))}"
    rc, stdout, stderr, attempts = run_on_bounded(
        scheduler, node, command, timeout=60
    )
    return {
        "node": node,
        "absent": rc == 0,
        "returncode": rc,
        "stdout": stdout.strip(),
        "stderr": stderr.strip(),
        "attempts": attempts,
    }


def parse_observation(stdout: str) -> tuple[dict[str, Any], str | None]:
    try:
        value = json.loads(stdout.strip())
    except Exception as error:
        return {}, str(error)
    if not isinstance(value, dict):
        return {}, "preflight output is not a JSON object"
    return value, None


def observation_passed(
    observation: dict[str, Any], job: dict[str, Any], source_map: dict[str, str]
) -> bool:
    admission = observation.get("admission", {})
    return bool(
        observation.get("schema_version") == PREFLIGHT_SCHEMA
        and observation.get("passed") is True
        and observation.get("study_stage") == "development"
        and observation.get("formal_held_out_outcomes_read") is False
        and observation.get("training_executed") is False
        and observation.get("sumo_executed") is False
        and observation.get("output_materialized") is False
        and observation.get("run_name") == job["name"]
        and observation.get("run_signature") == job["run_signature"]
        and observation.get("variant") == job["variant"]
        and observation.get("seed") == job["seed"]
        and observation.get("protocol_sha256") == EXPECTED_HASHES["protocol"]
        and observation.get("wrapper_sha256")
        == source_map["SimpleSAC/train_transit_v13_proximal_release.py"]
        and observation.get("v13_source_sha256") == source_map
        and observation.get("branch_handoff_manifest_sha256")
        == job["source_handoff"]["manifest_sha256"]
        and observation.get("branch_checkpoint_state_sha256")
        == job["source_handoff"]["checkpoint_state_sha256"]
        and admission.get("schema_version")
        == "transit-training-admission-v13-proximal-release-development-v2"
        and admission.get("study_stage") == "development"
        and admission.get("selected_arguments") == job
        and admission.get("v13_protocol_sha256") == EXPECTED_HASHES["protocol"]
        and admission.get("v13_source_sha256") == source_map
    )


def main() -> int:
    args = parse_args()
    paths = {
        "training_plan": PLAN,
        "protocol": PROTOCOL,
        "pretraining_correction": CORRECTION,
        "deployment_manifest": DEPLOYMENT_MANIFEST,
        "base_deployment_manifest": BASE_DEPLOYMENT_MANIFEST,
        "local_verification": LOCAL_VERIFICATION,
    }
    for label, path in paths.items():
        if sha256(path.expanduser().resolve(strict=True)) != EXPECTED_HASHES[label]:
            raise ValueError(f"frozen v13 v2 preflight input drifted: {label}")

    plan = json.loads(PLAN.read_text())
    local_verification = json.loads(LOCAL_VERIFICATION.read_text())
    jobs = plan.get("jobs", [])
    source_map = plan.get("v13_source_sha256", {})
    output_root = Path(plan["remote_output_root"])
    if (
        plan.get("schema_version")
        != "transit-h2oplus-v13-proximal-release-training-plan-v2"
        or len(jobs) != 15
        or len({job.get("run_signature") for job in jobs}) != 15
        or {job.get("seed") for job in jobs} != set(NODE_BY_SEED)
        or len(source_map) != 13
        or local_verification.get("passed") is not True
        or local_verification.get("verified_jobs") != 15
        or local_verification.get("local_training_or_sumo_executed") is not False
    ):
        raise ValueError("v13 v2 preflight matrix is not frozen and complete")

    scheduler = None
    if args.mode == "remote":
        scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
        sys.path.insert(0, str(scheduler_root / "skill"))
        scheduler = importlib.import_module("scheduler")

    if args.mode == "local":
        absence_before = [{"node": "local", "absent": not output_root.exists()}]
    else:
        absence_before = [
            remote_absence(scheduler, node, output_root)
            for node in sorted(set(NODE_BY_SEED.values()))
        ]
    if not all(row["absent"] for row in absence_before):
        raise FileExistsError("v13 v2 output root exists before preflight")

    environment = required_environment()
    prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in environment.items()
    )
    wrapper = WORKSPACE / "SimpleSAC/train_transit_v13_proximal_release.py"
    rows = []
    for job in jobs:
        argv = [PYTHON, str(wrapper), "--admission-preflight-only", *job["command_arguments"]]
        shell_command = (
            f"cd {shlex.quote(str(WORKSPACE))} && {prefix} {shlex.join(argv)}"
        )
        node = "local" if args.mode == "local" else NODE_BY_SEED[job["seed"]]
        attempts = []
        if args.mode == "local":
            process_environment = os.environ.copy()
            process_environment.update(environment)
            completed = subprocess.run(
                argv,
                cwd=WORKSPACE,
                env=process_environment,
                text=True,
                capture_output=True,
                timeout=180,
                check=False,
            )
            rc, stdout, stderr = (
                completed.returncode,
                completed.stdout,
                completed.stderr,
            )
            attempts.append(
                {
                    "attempt": 1,
                    "returncode": rc,
                    "stdout": stdout.strip(),
                    "stderr": stderr.strip(),
                }
            )
        else:
            rc, stdout, stderr, attempts = run_on_bounded(
                scheduler, node, shell_command, timeout=180
            )
        observation, parse_error = parse_observation(stdout) if rc == 0 else ({}, None)
        output_still_absent = args.mode != "local" or not output_root.exists()
        passed = bool(
            rc == 0
            and parse_error is None
            and output_still_absent
            and observation_passed(observation, job, source_map)
            and not any(
                token in stderr.lower()
                for token in ("traceback", "segmentation fault", "sumo error")
            )
        )
        rows.append(
            {
                "name": job["name"],
                "seed": job["seed"],
                "variant": job["variant"],
                "run_signature": job["run_signature"],
                "node": node,
                "passed": passed,
                "returncode": rc,
                "parse_error": parse_error,
                "observation": observation,
                "stderr": stderr.strip(),
                "attempts": attempts,
            }
        )
        if not passed:
            break

    if args.mode == "local":
        absence_after = [{"node": "local", "absent": not output_root.exists()}]
    else:
        absence_after = [
            remote_absence(scheduler, node, output_root)
            for node in sorted(set(NODE_BY_SEED.values()))
        ]
    output = args.output
    if output is None:
        output = REPRO / (
            "h2oplus_v13_proximal_release_runtime_preflight_v2_"
            f"{args.mode}.json"
        )
    payload = {
        "schema_version": RECORD_SCHEMA,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "passed": (
            len(rows) == 15
            and all(row["passed"] for row in rows)
            and all(row["absent"] for row in absence_after)
        ),
        "study_stage": "development",
        "frozen_inputs": {label: descriptor(path) for label, path in paths.items()},
        "preflight_program": descriptor(Path(__file__)),
        "required_environment": environment,
        "scheduler_managed_environment_during_training": {
            "CUDA_VISIBLE_DEVICES": "exactly one physical GPU assigned by scheduleurm"
        },
        "preflight_cuda_visible_devices": "",
        "output_root": str(output_root),
        "output_root_absence_before": absence_before,
        "output_root_absence_after": absence_after,
        "verified_jobs": len(rows),
        "jobs": rows,
        "training_executed": False,
        "sumo_executed": False,
        "output_materialized": False,
        "formal_held_out_outcomes_read": False,
    }
    atomic_json(output.expanduser().resolve(), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
