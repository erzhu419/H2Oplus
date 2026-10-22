#!/usr/bin/env python3
"""Verify the frozen v11 development-evaluation deployment on jtl110gpu."""

from __future__ import annotations

import argparse
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
SCHEMA = "transit-h2oplus-v11-actor-hold-evaluation-remote-verification-v1"
EXPECTED_MANIFEST_SHA256 = "b579fade4a0c4ea6dac164620dd50818818fc49666d365ddff63a26ab989246d"
EXPECTED_AGGREGATE_SHA256 = "c8db97ab6d63000df936b987c7bc1eff88dd1dba8ca4d4a9baff3e698cb16741"
EXPECTED_RUNTIME_SIGNATURE = "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
EXPECTED_ENTRY_COUNT = 545
EXPECTED_SOURCE_COUNT = 41
EXPECTED_SOURCE_FREEZE_SHA256 = "fe61024af63450c6502767c2fcee7915c2b2e8216cf23812f63523e241b237c7"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
        default=REPRO / "deployments/h2oplus_v11_actor_hold_evaluation_v1/workspace",
    )
    parser.add_argument("--node", choices=("jtl110gpu",), default="jtl110gpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO
        / "h2oplus_v11_actor_hold_evaluation_deployment_v1_jtl110gpu_verification.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    manifest = deployment_root / "deployment_manifest.json"
    if manifest.stat().st_size <= 0:
        raise ValueError("local deployment manifest is empty")
    sys.path.insert(0, str(scheduler_root))
    scheduler = importlib.import_module("skill.scheduler")
    remote_root = str(deployment_root)
    remote_repo = f"{remote_root}/H2Oplus"
    remote_sumo = f"{remote_root}/SUMO_ruiguang/online_control"
    remote_edge = f"{remote_repo}/bus_h2o/network_data/a_sorted_busline_edge.xml"
    remote_lock = (
        f"{remote_repo}/reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json"
    )
    remote_methods = (
        f"{remote_repo}/reproducibility/h2oplus_v11_actor_hold_evaluation_methods_v1.json"
    )
    environment = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": (
            "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/deployments/"
            "v7_formal_v2_training_v7r1/workspace/reproducibility/toolchain/"
            "libxcrypt-4.4.36-hd590300_1/include"
        ),
        "PATH": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin",
        "PYTHONPATH": remote_repo,
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    remote_code = f'''
import hashlib
import json
import os
from pathlib import Path

import torch

from SimpleSAC import evaluate_transit_v11_actor_hold as evaluator
from bus_h2o.sumo_provenance import (
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    load_sumo_runtime_lock,
    sumo_runtime_fingerprint,
)
from reproducibility import run_evaluation_matrix_v11_actor_hold as matrix

root = Path({remote_root!r}).resolve(strict=True)
manifest_path = root / "deployment_manifest.json"
manifest_bytes = manifest_path.read_bytes()
manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
manifest = json.loads(manifest_bytes)
entries = manifest.get("entries", [])
errors = []
total_bytes = 0
for entry in entries:
    path = root / entry["path"]
    if not path.is_file():
        errors.append("missing:" + entry["path"])
        continue
    size = path.stat().st_size
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    total_bytes += size
    if size != entry["size_bytes"] or digest != entry["sha256"]:
        errors.append("drift:" + entry["path"])
aggregate = hashlib.sha256(
    json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()
sumo_dir = Path({remote_sumo!r})
edge_xml = Path({remote_edge!r})
child_sources = {{str(path.resolve()) for path in evaluator.evaluation_source_paths(sumo_dir, edge_xml)}}
parent_sources = {{str(path.resolve()) for path in matrix.evaluation_sources(sumo_dir, edge_xml)}}
runtime_lock, runtime_identity = load_sumo_runtime_lock(Path({remote_lock!r}))
active_runtime = sumo_runtime_fingerprint(
    FORMAL_V7_SUMO_BACKEND,
    expected_version=FORMAL_V7_SUMO_VERSION,
)
methods_path = Path({remote_methods!r})
methods_payload = json.loads(methods_path.read_text())
methods = matrix.load_development_methods(methods_path, matrix.METHODS_SCHEMA)
physical = {{row["name"]: row for row in methods}}
witness_errors = []
witness_count = 0
for identity in methods_payload.get("identity_evidence", []):
    seed = identity.get("training_seed")
    rows = identity.get("v11_source_policies", [])
    actor_hashes = []
    temperature_hashes = []
    for witness in rows:
        raw = Path(witness["policy"]["path"])
        policy = raw if raw.is_absolute() else methods_path.parent / raw
        artifact = torch.load(policy.resolve(strict=True), map_location="cpu")
        extra = artifact.get("extra", {{}})
        actor_hash = evaluator.actor_digest(artifact.get("actor", {{}}))
        temperature_hash = extra.get("v11_temperature_tensor_sha256")
        witness_count += 1
        actor_hashes.append(actor_hash)
        temperature_hashes.append(temperature_hash)
        if (
            hashlib.sha256(policy.read_bytes()).hexdigest() != witness["policy"]["sha256"]
            or actor_hash != witness["actor_tensor_sha256"]
            or temperature_hash != witness["temperature_tensor_sha256"]
            or extra.get("training_seed") != seed
            or extra.get("v11_protocol", {{}}).get("sha256")
            != methods_payload.get("v11_protocol_sha256")
        ):
            witness_errors.append("witness-drift:" + str(seed) + ":" + str(witness.get("online_events")))
    initializer = physical[f"h2oplus-seed{{seed}}-events000000000"]["actor_tensor_sha256"]
    if (
        len(rows) != 4
        or len(set(actor_hashes[:3])) != 1
        or actor_hashes[0] != initializer
        or len(set(temperature_hashes[:3])) != 1
    ):
        witness_errors.append("hold-identity-drift:" + str(seed))
source_freeze_path = root / "H2Oplus/reproducibility/h2oplus_v11_actor_hold_evaluation_source_freeze_v1.json"
source_freeze_sha256 = hashlib.sha256(source_freeze_path.read_bytes()).hexdigest()
environment = {{key: os.environ.get(key) for key in {tuple(environment)!r}}}
print(json.dumps({{
    "hostname": os.uname().nodename,
    "manifest_sha256": manifest_sha256,
    "aggregate_sha256": aggregate,
    "manifest_aggregate_sha256": manifest.get("aggregate_sha256"),
    "entry_count": len(entries),
    "verified_bytes": total_bytes,
    "file_errors": errors,
    "evaluator_source_file_count": len(child_sources),
    "matrix_source_file_count": len(parent_sources),
    "source_sets_equal": child_sources == parent_sources,
    "only_evaluator": sorted(child_sources - parent_sources),
    "only_matrix": sorted(parent_sources - child_sources),
    "runtime_signature": active_runtime["portable_signature"],
    "runtime_lock_signature": runtime_identity["runtime_signature"],
    "runtime_identity_equal": (
        active_runtime["portable_identity"]
        == runtime_lock["runtime_fingerprint"]["portable_identity"]
    ),
    "runtime_version": active_runtime["runtime_version"],
    "runtime_backend": active_runtime["backend"],
    "physical_method_count": len(methods),
    "unique_actor_tensor_count": len({{row["actor_tensor_sha256"] for row in methods}}),
    "logical_alias_count": methods_payload.get("logical_method_budget_alias_count"),
    "methods_paths_are_relative": all(
        not Path(row["checkpoint"]).is_absolute() for row in methods_payload.get("methods", [])
    ),
    "hold_witness_count": witness_count,
    "hold_witness_errors": witness_errors,
    "source_freeze_sha256": source_freeze_sha256,
    "environment": environment,
}}, sort_keys=True))
'''
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in environment.items())
    command = (
        f"cd {shlex.quote(remote_repo)} && {prefix} "
        "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python "
        f"-c {shlex.quote(remote_code)}"
    )
    returncode, stdout, stderr = scheduler.run_on(
        args.node, command, timeout=900, check=False
    )
    observation: dict[str, Any] = {}
    parse_error = None
    if returncode == 0:
        try:
            observation = json.loads(stdout.strip())
        except Exception as error:
            parse_error = str(error)
    passed = bool(
        returncode == 0
        and parse_error is None
        and observation.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256
        and observation.get("aggregate_sha256") == EXPECTED_AGGREGATE_SHA256
        and observation.get("manifest_aggregate_sha256") == EXPECTED_AGGREGATE_SHA256
        and observation.get("entry_count") == EXPECTED_ENTRY_COUNT
        and observation.get("file_errors") == []
        and observation.get("evaluator_source_file_count") == EXPECTED_SOURCE_COUNT
        and observation.get("matrix_source_file_count") == EXPECTED_SOURCE_COUNT
        and observation.get("source_sets_equal") is True
        and observation.get("only_evaluator") == []
        and observation.get("only_matrix") == []
        and observation.get("runtime_signature") == EXPECTED_RUNTIME_SIGNATURE
        and observation.get("runtime_lock_signature") == EXPECTED_RUNTIME_SIGNATURE
        and observation.get("runtime_identity_equal") is True
        and observation.get("runtime_version") == "1.25.0"
        and observation.get("runtime_backend") == "libsumo"
        and observation.get("physical_method_count") == 45
        and observation.get("unique_actor_tensor_count") == 45
        and observation.get("logical_alias_count") == 80
        and observation.get("methods_paths_are_relative") is True
        and observation.get("hold_witness_count") == 20
        and observation.get("hold_witness_errors") == []
        and observation.get("source_freeze_sha256") == EXPECTED_SOURCE_FREEZE_SHA256
        and observation.get("environment") == environment
    )
    payload = {
        "schema_version": SCHEMA,
        "verified_utc": utc_now(),
        "node": args.node,
        "passed": passed,
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
        "remote_sumo_simulation_executed": False,
        "verification_scope": (
            "file hashes, exact source contract, imports, physical actor provenance, "
            "hold witnesses, and runtime fingerprint only"
        ),
        "expected": {
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "aggregate_sha256": EXPECTED_AGGREGATE_SHA256,
            "entry_count": EXPECTED_ENTRY_COUNT,
            "source_file_count": EXPECTED_SOURCE_COUNT,
            "runtime_signature": EXPECTED_RUNTIME_SIGNATURE,
            "source_freeze_sha256": EXPECTED_SOURCE_FREEZE_SHA256,
            "environment": environment,
        },
        "observation": observation,
        "remote_returncode": returncode,
        "remote_stderr": stderr.strip(),
        "parse_error": parse_error,
    }
    output = args.output.expanduser().resolve()
    atomic_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
