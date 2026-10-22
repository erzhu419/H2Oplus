#!/usr/bin/env python3
"""Run dependency-ordered v7 trainer gates on an authenticated GPU server."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility.audit_device_cache_precision_v6 import (
    SCHEMA_VERSION as PRECISION_SCHEMA,
    precision_audit_source_hashes,
    validate_v7_selection_evidence,
)
from reproducibility.run_hybrid_entropy_pilot_v7 import (
    SCHEMA_VERSION as ENTROPY_SCHEMA,
    entropy_pilot_source_hashes,
)
from reproducibility.run_reward_loss_pilot_v7 import (
    SCHEMA_VERSION as REWARD_LOSS_SCHEMA,
    reward_loss_source_hashes,
)


SCHEMA_VERSION = "transit-v7-server-trainer-gate-chain-v1"
PILOT_ROOT = ROOT / "bus_h2o" / "datasets_v7" / "formal_v2_pilot_r2"
FORMAL_ROOT = ROOT / "bus_h2o" / "datasets_v7" / "formal_v2_full"
PILOT_DATASET = PILOT_ROOT / "transit_smdp_v7_pilot_r2.h5"
PILOT_AUDIT = PILOT_ROOT / "transit_smdp_v7_pilot_r2.audit.json"
REWARD_AUDIT = PILOT_ROOT / "reward_conditioning_v7_pilot_r2.json"
FORMAL_DATASET = FORMAL_ROOT / "transit_smdp_v7_formal_v2.h5"
DEFAULT_REWARD_LOSS_OUTPUT = (
    PILOT_ROOT / "reward_loss_pilot_v7_formal_gate_v2.json"
)
DEFAULT_ENTROPY_OUTPUT = PILOT_ROOT / "hybrid_entropy_pilot_v7_formal_gate_v2.json"
DEFAULT_PRECISION_OUTPUT = FORMAL_ROOT / "device_cache_precision_v7_formal_v2.json"
DEFAULT_MANIFEST_OUTPUT = FORMAL_ROOT / "server_trainer_gate_chain_v7_formal_v2.json"
DEFAULT_COMPILER = Path(
    "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/"
    "x86_64-conda-linux-gnu-gcc"
)
TOOLCHAIN_INCLUDE = (
    ROOT / "reproducibility" / "toolchain" / "libxcrypt-4.4.36-hd590300_1" / "include"
)
SYSTEM_LIBCURL_PRELOAD = Path("/usr/lib/x86_64-linux-gnu/libcurl.so.4")
SERVER_HOSTNAME_ALIASES = {
    "jtl110gpu": {"jtl110gpu", "huiwei-Super-Server"},
    "jtl110gpu2": {"jtl110gpu2", "erzhu419-Super-Server"},
    "node007": {"node007"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--server-label",
        required=True,
        choices=tuple(SERVER_HOSTNAME_ALIASES),
        help="canonical scheduler label for the physical GPU server",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--reward-loss-output", type=Path, default=DEFAULT_REWARD_LOSS_OUTPUT)
    parser.add_argument("--entropy-output", type=Path, default=DEFAULT_ENTROPY_OUTPUT)
    parser.add_argument("--precision-output", type=Path, default=DEFAULT_PRECISION_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse only already-valid completed stage artifacts",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"gate artifact is not a JSON object: {path}")
    return payload


def _run(command: list[str], *, environment: dict[str, str]) -> None:
    print(json.dumps({"command": command}, sort_keys=True), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _descriptor(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def _selected_reward_loss(path: Path) -> dict[str, Any]:
    payload = _load(path)
    selection = payload.get("configuration_selection", {})
    selected = selection.get("selected")
    if (
        payload.get("schema_version") != REWARD_LOSS_SCHEMA
        or payload.get("study_stage") != "numerical-pilot"
        or payload.get("dataset", {}).get("sha256") != sha256(PILOT_DATASET)
        or payload.get("reward_audit", {}).get("sha256") != sha256(REWARD_AUDIT)
        or payload.get("source_sha256") != reward_loss_source_hashes()
        or selection.get("passed") is not True
        or not isinstance(selected, dict)
    ):
        raise ValueError(f"reward/loss gate artifact is stale or failed: {path}")
    return selected


def _selected_entropy(path: Path, reward_loss_path: Path) -> float:
    payload = _load(path)
    gate = payload.get("target_entropy_gate", {})
    if (
        payload.get("schema_version") != ENTROPY_SCHEMA
        or payload.get("study_stage") != "numerical-pilot"
        or payload.get("dataset", {}).get("sha256") != sha256(PILOT_DATASET)
        or payload.get("reward_loss_configuration", {}).get("pilot_sha256")
        != sha256(reward_loss_path)
        or payload.get("source_sha256") != entropy_pilot_source_hashes()
        or gate.get("passed") is not True
        or gate.get("selected_target_entropy") is None
    ):
        raise ValueError(f"entropy gate artifact is stale or failed: {path}")
    return float(gate["selected_target_entropy"])


def _validate_precision(
    path: Path,
    reward_loss_path: Path,
    entropy_path: Path,
) -> dict[str, Any]:
    payload = _load(path)
    evidence = payload.get("selection_evidence", {})
    decision = payload.get("precision_decision", {})
    if (
        payload.get("schema_version") != PRECISION_SCHEMA
        or payload.get("dataset_sha256") != sha256(FORMAL_DATASET)
        or payload.get("mdp_version") != "v7"
        or payload.get("method") != "h2oplus"
        or payload.get("phase") != "online"
        or payload.get("source_sha256") != precision_audit_source_hashes("v7")
        or evidence.get("reward_loss_pilot", {}).get("sha256")
        != sha256(reward_loss_path)
        or evidence.get("entropy_pilot", {}).get("sha256") != sha256(entropy_path)
        or decision.get("passed") is not True
    ):
        raise ValueError(f"precision gate artifact is stale or failed: {path}")
    return decision


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite trainer-gate manifest {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _validate_server_identity(server_label: str) -> str:
    hostname = socket.gethostname().split(".", 1)[0]
    if hostname not in SERVER_HOSTNAME_ALIASES[server_label]:
        raise RuntimeError(
            f"server label {server_label!r} does not match hostname {hostname!r}"
        )
    return hostname


def main() -> None:
    args = parse_args()
    hostname = _validate_server_identity(args.server_label)
    python = args.python.expanduser().resolve(strict=True)
    reward_loss_output = args.reward_loss_output.expanduser().resolve()
    entropy_output = args.entropy_output.expanduser().resolve()
    precision_output = args.precision_output.expanduser().resolve()
    manifest_output = args.manifest_output.expanduser().resolve()
    if manifest_output.exists():
        raise FileExistsError(f"trainer-gate chain is already frozen: {manifest_output}")
    for path in (PILOT_DATASET, PILOT_AUDIT, REWARD_AUDIT, FORMAL_DATASET):
        path.resolve(strict=True)
    compiler = DEFAULT_COMPILER.resolve(strict=True)
    toolchain_include = TOOLCHAIN_INCLUDE.resolve(strict=True)
    libc_url = SYSTEM_LIBCURL_PRELOAD.resolve(strict=True)

    environment = dict(os.environ)
    environment.update(
        {
            "CC": str(compiler),
            "CPATH": str(toolchain_include),
            "LD_PRELOAD": str(libc_url),
            "TMPDIR": "/tmp",
            "TEMP": "/tmp",
            "TMP": "/tmp",
            "PYTHONPATH": str(ROOT),
        }
    )
    common = [str(python), "-u"]

    if reward_loss_output.exists():
        if not args.resume:
            raise FileExistsError(f"reward/loss output already exists: {reward_loss_output}")
    else:
        _run(
            [
                *common,
                "reproducibility/run_reward_loss_pilot_v7.py",
                "--dataset",
                str(PILOT_DATASET),
                "--dataset-audit",
                str(PILOT_AUDIT),
                "--reward-audit",
                str(REWARD_AUDIT),
                "--device",
                args.device,
                "--updates",
                "2000",
                "--max-offline-rows",
                "100000",
                "--diagnostic-rows",
                "2048",
                "--cpu-threads",
                str(args.cpu_threads),
                "--device-cache",
                "float32",
                "--target-entropy",
                "0",
                "--output",
                str(reward_loss_output),
            ],
            environment=environment,
        )
    selected = _selected_reward_loss(reward_loss_output)

    if entropy_output.exists():
        if not args.resume:
            raise FileExistsError(f"entropy output already exists: {entropy_output}")
    else:
        _run(
            [
                *common,
                "reproducibility/run_hybrid_entropy_pilot_v7.py",
                "--dataset",
                str(PILOT_DATASET),
                "--dataset-audit",
                str(PILOT_AUDIT),
                "--reward-loss-pilot",
                str(reward_loss_output),
                "--reward-scale",
                str(selected["reward_scale"]),
                "--critic-loss",
                str(selected["critic_loss"]),
                "--huber-delta",
                str(selected["huber_delta"]),
                "--device",
                args.device,
                "--offline-updates",
                "2000",
                "--online-updates",
                "2000",
                "--online-rows",
                "4096",
                "--max-offline-rows",
                "100000",
                "--diagnostic-rows",
                "2048",
                "--cpu-threads",
                str(args.cpu_threads),
                "--device-cache",
                "float32",
                "--output",
                str(entropy_output),
            ],
            environment=environment,
        )
    target_entropy = _selected_entropy(entropy_output, reward_loss_output)
    validate_v7_selection_evidence(
        reward_loss_output,
        entropy_output,
        reward_scale=float(selected["reward_scale"]),
        critic_loss=str(selected["critic_loss"]),
        huber_delta=float(selected["huber_delta"]),
        target_entropy=target_entropy,
    )

    if precision_output.exists():
        if not args.resume:
            raise FileExistsError(f"precision output already exists: {precision_output}")
    else:
        _run(
            [
                *common,
                "reproducibility/audit_device_cache_precision_v6.py",
                "--dataset",
                str(FORMAL_DATASET),
                "--mdp-version",
                "v7",
                "--method",
                "h2oplus",
                "--phase",
                "online",
                "--device",
                args.device,
                "--seed",
                "9201",
                "--max-offline-rows",
                "100000",
                "--online-rows",
                "4096",
                "--sample-rows",
                "8192",
                "--paired-updates",
                "2000",
                "--cpu-threads",
                str(args.cpu_threads),
                "--reward-scale",
                str(selected["reward_scale"]),
                "--critic-loss",
                str(selected["critic_loss"]),
                "--huber-delta",
                str(selected["huber_delta"]),
                "--target-entropy",
                str(target_entropy),
                "--reward-loss-pilot",
                str(reward_loss_output),
                "--entropy-pilot",
                str(entropy_output),
                "--output",
                str(precision_output),
            ],
            environment=environment,
        )
    precision_decision = _validate_precision(
        precision_output, reward_loss_output, entropy_output
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution_boundary": "registered GPU server; H2O+ trainer admission only",
        "execution_host": {
            "scheduler_label": args.server_label,
            "hostname": hostname,
        },
        "dependency_order": ["reward_loss", "hybrid_entropy", "device_precision"],
        "inputs": {
            "pilot_dataset": _descriptor(PILOT_DATASET),
            "pilot_dataset_audit": _descriptor(PILOT_AUDIT),
            "reward_audit": _descriptor(REWARD_AUDIT),
            "formal_dataset": _descriptor(FORMAL_DATASET),
        },
        "outputs": {
            "reward_loss": _descriptor(reward_loss_output),
            "hybrid_entropy": _descriptor(entropy_output),
            "device_precision": _descriptor(precision_output),
        },
        "selected_arguments": {
            **selected,
            "target_entropy": target_entropy,
            "device_cache_observation_dtype": precision_decision[
                "selected_observation_storage_dtype"
            ],
        },
        "training_runtime_lock": _load(precision_output)["runtime_fingerprint"][
            "required"
        ],
        "runner": _descriptor(Path(__file__)),
    }
    _write_new(manifest_output, payload)
    print(
        json.dumps(
            {
                "manifest": str(manifest_output),
                "selected_arguments": payload["selected_arguments"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
