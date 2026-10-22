#!/usr/bin/env python3
"""Rebuild source-bound v7 numerical gates after the event-cap fix."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility.audit_device_cache_precision_v6 import (
    SCHEMA_VERSION as PRECISION_SCHEMA,
    precision_audit_source_hashes,
)
from reproducibility.build_training_admission_v7 import (
    SCHEMA_VERSION as ADMISSION_SCHEMA,
)
from reproducibility.run_hybrid_entropy_pilot_v7 import (
    SCHEMA_VERSION as ENTROPY_SCHEMA,
    entropy_pilot_source_hashes,
)
from reproducibility.run_reward_loss_pilot_v7 import (
    SCHEMA_VERSION as REWARD_LOSS_SCHEMA,
    reward_loss_source_hashes,
)
from SimpleSAC.train_transit_v3 import source_hashes


SCHEMA_VERSION = "transit-v7-event-cap-regate-v1"
REQUIRED_PRELOAD = "/usr/lib/x86_64-linux-gnu/libcurl.so.4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-protocol-audit", type=Path, required=True)
    parser.add_argument("--training-admission-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-external-gpu-processes", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _descriptor(path: Path) -> dict[str, Any]:
    payload = _load(path)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "schema_version": payload.get("schema_version"),
    }


def _reward_loss_valid(path: Path) -> bool:
    payload = _load(path)
    selection = payload.get("configuration_selection", {})
    selected = selection.get("selected", {})
    return bool(
        payload.get("schema_version") == REWARD_LOSS_SCHEMA
        and payload.get("source_sha256") == reward_loss_source_hashes()
        and selection.get("passed") is True
        and selected.get("reward_scale") == 0.0001
        and selected.get("critic_loss") == "huber"
        and selected.get("huber_delta") == 1.0
    )


def _entropy_valid(path: Path, reward_loss: Path) -> bool:
    payload = _load(path)
    config = payload.get("reward_loss_configuration", {})
    gate = payload.get("target_entropy_gate", {})
    return bool(
        payload.get("schema_version") == ENTROPY_SCHEMA
        and payload.get("source_sha256") == entropy_pilot_source_hashes()
        and config.get("pilot_sha256") == sha256(reward_loss)
        and config.get("reward_scale") == 0.0001
        and config.get("critic_loss") == "huber"
        and config.get("huber_delta") == 1.0
        and gate.get("passed") is True
        and gate.get("selected_target_entropy") == 0.0
    )


def _precision_valid(path: Path, reward_loss: Path, entropy: Path) -> bool:
    payload = _load(path)
    selection = payload.get("selection_evidence", {})
    decision = payload.get("precision_decision", {})
    return bool(
        payload.get("schema_version") == PRECISION_SCHEMA
        and payload.get("source_sha256") == precision_audit_source_hashes("v7")
        and payload.get("mdp_version") == "v7"
        and payload.get("method") == "h2oplus"
        and payload.get("phase") == "online"
        and selection.get("reward_loss_pilot", {}).get("sha256")
        == sha256(reward_loss)
        and selection.get("entropy_pilot", {}).get("sha256") == sha256(entropy)
        and decision.get("passed") is True
        and decision.get("selected_observation_storage_dtype") == "float32"
    )


def _admission_valid(
    path: Path,
    reward_loss: Path,
    entropy: Path,
    precision: Path,
    method_audit: Path,
) -> bool:
    payload = _load(path)
    evidence = payload.get("evidence", {})
    return bool(
        payload.get("schema_version") == ADMISSION_SCHEMA
        and payload.get("passed") is True
        and payload.get("training_source_sha256") == source_hashes("v7")
        and evidence.get("reward_loss_pilot", {}).get("sha256")
        == sha256(reward_loss)
        and evidence.get("entropy_pilot", {}).get("sha256") == sha256(entropy)
        and evidence.get("device_precision_audit", {}).get("sha256")
        == sha256(precision)
        and evidence.get("method_protocol_audit", {}).get("sha256")
        == sha256(method_audit)
    )


def _run_stage(
    name: str,
    output: Path,
    command: list[str],
    validator: Callable[[Path], bool],
    summary: dict[str, Any],
) -> None:
    started = time.time()
    if output.is_file():
        if not validator(output):
            raise RuntimeError(f"existing {name} artifact is stale: {output}")
        status = "reused"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        log_path = output.with_suffix(output.suffix + ".log")
        print(f"START {name}: {' '.join(command)}", flush=True)
        with log_path.open("a") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"{name} failed with return code {result.returncode}; see {log_path}"
            )
        if not output.is_file() or not validator(output):
            raise RuntimeError(f"{name} produced an invalid artifact: {output}")
        status = "completed"
    summary["stages"][name] = {
        "status": status,
        "elapsed_seconds": round(time.time() - started, 3),
        "command": command,
        "artifact": _descriptor(output),
    }
    summary["updated_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(Path(summary["summary_path"]), summary)
    print(f"DONE {name}: {status} {output}", flush=True)


def main() -> None:
    args = parse_args()
    preload_entries = os.environ.get("LD_PRELOAD", "").split(":")
    if REQUIRED_PRELOAD not in preload_entries:
        raise RuntimeError(f"LD_PRELOAD must include {REQUIRED_PRELOAD}")

    output_dir = args.output_dir.expanduser().resolve()
    method_audit = args.method_protocol_audit.expanduser().resolve()
    admission = args.training_admission_output.expanduser().resolve()
    if not method_audit.is_file():
        raise FileNotFoundError(method_audit)
    output_dir.mkdir(parents=True, exist_ok=True)

    pilot_root = ROOT / "bus_h2o" / "datasets_v7" / "formal_v2_pilot_r2"
    full_root = ROOT / "bus_h2o" / "datasets_v7" / "formal_v2_full"
    reward_loss = output_dir / "reward_loss_pilot_v7_event_cap_fix_v1.json"
    entropy = output_dir / "hybrid_entropy_pilot_v7_event_cap_fix_v1.json"
    precision = output_dir / "device_cache_precision_v7_event_cap_fix_v1.json"
    summary_path = output_dir / "event_cap_regate_summary_v1.json"
    allow_external = (
        ["--allow-external-gpu-processes"]
        if args.allow_external_gpu_processes
        else []
    )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "summary_path": str(summary_path),
        "root": str(ROOT),
        "device": args.device,
        "allow_external_gpu_processes": args.allow_external_gpu_processes,
        "event_cap_fix": {
            "source": str(ROOT / "SimpleSAC" / "train_transit_v3.py"),
            "sha256": sha256(ROOT / "SimpleSAC" / "train_transit_v3.py"),
        },
        "method_protocol_audit": _descriptor(method_audit),
        "stages": {},
    }
    if summary_path.exists():
        previous = _load(summary_path)
        if previous.get("event_cap_fix") != summary["event_cap_fix"]:
            raise RuntimeError("existing regate summary belongs to another source")
        summary["created_utc"] = previous.get("created_utc", summary["created_utc"])
        summary["stages"] = previous.get("stages", {})
    _write_json(summary_path, summary)

    _run_stage(
        "reward_loss",
        reward_loss,
        [
            sys.executable,
            str(ROOT / "reproducibility" / "run_reward_loss_pilot_v7.py"),
            "--dataset",
            str(pilot_root / "transit_smdp_v7_pilot_r2.h5"),
            "--dataset-audit",
            str(pilot_root / "transit_smdp_v7_pilot_r2.audit.json"),
            "--reward-audit",
            str(pilot_root / "reward_conditioning_v7_pilot_r2.json"),
            "--device",
            args.device,
            "--updates",
            "2000",
            "--max-offline-rows",
            "100000",
            "--diagnostic-rows",
            "2048",
            "--cpu-threads",
            "8",
            "--device-cache",
            "float32",
            "--target-entropy",
            "0.0",
            *allow_external,
            "--output",
            str(reward_loss),
        ],
        _reward_loss_valid,
        summary,
    )
    _run_stage(
        "entropy",
        entropy,
        [
            sys.executable,
            str(ROOT / "reproducibility" / "run_hybrid_entropy_pilot_v7.py"),
            "--dataset",
            str(pilot_root / "transit_smdp_v7_pilot_r2.h5"),
            "--dataset-audit",
            str(pilot_root / "transit_smdp_v7_pilot_r2.audit.json"),
            "--reward-loss-pilot",
            str(reward_loss),
            "--reward-scale",
            "0.0001",
            "--critic-loss",
            "huber",
            "--huber-delta",
            "1.0",
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
            "8",
            "--device-cache",
            "float32",
            *allow_external,
            "--output",
            str(entropy),
        ],
        lambda path: _entropy_valid(path, reward_loss),
        summary,
    )
    _run_stage(
        "device_precision",
        precision,
        [
            sys.executable,
            str(ROOT / "reproducibility" / "audit_device_cache_precision_v6.py"),
            "--dataset",
            str(full_root / "transit_smdp_v7_formal_v2.h5"),
            "--mdp-version",
            "v7",
            "--method",
            "h2oplus",
            "--phase",
            "online",
            "--device",
            args.device,
            "--seed",
            "9601",
            "--max-offline-rows",
            "100000",
            "--online-rows",
            "4096",
            "--sample-rows",
            "8192",
            "--paired-updates",
            "2000",
            "--cpu-threads",
            "8",
            "--reward-scale",
            "0.0001",
            "--critic-loss",
            "huber",
            "--huber-delta",
            "1.0",
            "--target-entropy",
            "0.0",
            "--reward-loss-pilot",
            str(reward_loss),
            "--entropy-pilot",
            str(entropy),
            *allow_external,
            "--output",
            str(precision),
        ],
        lambda path: _precision_valid(path, reward_loss, entropy),
        summary,
    )
    _run_stage(
        "training_admission",
        admission,
        [
            sys.executable,
            str(ROOT / "reproducibility" / "build_training_admission_v7.py"),
            "--dataset",
            str(full_root / "transit_smdp_v7_formal_v2.h5"),
            "--pilot-dataset",
            str(pilot_root / "transit_smdp_v7_pilot_r2.h5"),
            "--dataset-audit",
            str(full_root / "transit_smdp_v7_formal_v2.audit.json"),
            "--serialized-credit-audit",
            str(full_root / "transit_smdp_v7_formal_v2.credit_audit.json"),
            "--sim-env",
            str(ROOT / "bus_h2o" / "calibrated_env"),
            "--zero-fidelity-audit",
            str(ROOT / "bus_h2o" / "calibrated_env" / "fidelity_audit_v7_formal_v4.json"),
            "--intervention-fidelity-audit",
            str(
                ROOT
                / "bus_h2o"
                / "calibrated_env"
                / "intervention_fidelity_audit_v7_formal_v3.json"
            ),
            "--reward-audit",
            str(pilot_root / "reward_conditioning_v7_pilot_r2.json"),
            "--reward-loss-pilot",
            str(reward_loss),
            "--entropy-pilot",
            str(entropy),
            "--device-precision-audit",
            str(precision),
            "--method-protocol-audit",
            str(method_audit),
            "--output",
            str(admission),
        ],
        lambda path: _admission_valid(
            path, reward_loss, entropy, precision, method_audit
        ),
        summary,
    )
    summary["status"] = "completed"
    summary["completed_utc"] = datetime.now(timezone.utc).isoformat()
    summary["training_admission"] = _descriptor(admission)
    _write_json(summary_path, summary)
    print(f"DONE event-cap regate: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
