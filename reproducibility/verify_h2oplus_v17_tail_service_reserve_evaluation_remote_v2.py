#!/usr/bin/env python3
"""Verify the frozen v17-v2 deployment under remote Python 3.8 without SUMO episodes."""

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
SCHEMA = "transit-h2oplus-v17-tail-service-reserve-evaluation-remote-verification-v2"
EXPECTED_RUNTIME_SIGNATURE = "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
EXPECTED_SOURCE_COUNT = 53


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite remote verification: {path}")
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
        default=REPRO / "deployments/h2oplus_v17_tail_service_reserve_evaluation_v2/workspace",
    )
    parser.add_argument("--node", choices=("jtl110gpu",), default="jtl110gpu")
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-aggregate-sha256", required=True)
    parser.add_argument("--expected-entry-count", type=int, required=True)
    parser.add_argument("--expected-source-freeze-sha256", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO
        / "h2oplus_v17_tail_service_reserve_evaluation_deployment_v2_jtl110gpu_verification.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    manifest = deployment_root / "deployment_manifest.json"
    if sha256(manifest) != args.expected_manifest_sha256:
        raise ValueError("local deployment manifest identity drifted")

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
        f"{remote_repo}/reproducibility/"
        "h2oplus_v17_tail_service_reserve_evaluation_methods_v1.json"
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
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    remote_code = f'''
import hashlib
import json
import os
import platform
from pathlib import Path

from SimpleSAC import evaluate_transit_v17_tail_service_reserve as evaluator
from bus_h2o.sumo_provenance import (
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    load_sumo_runtime_lock,
    sumo_runtime_fingerprint,
)
from reproducibility import run_evaluation_matrix_v17_tail_service_reserve as matrix
from reproducibility import run_h2oplus_v17_tail_service_reserve_evaluation_dispatch_v2 as dispatch

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

root = Path({remote_root!r}).resolve(strict=True)
manifest_path = root / "deployment_manifest.json"
manifest_sha256 = digest(manifest_path)
manifest = json.loads(manifest_path.read_text())
entries = manifest.get("entries", [])
errors = []
total_bytes = 0
for entry in entries:
    path = root / entry["path"]
    if not path.is_file():
        errors.append("missing:" + entry["path"])
        continue
    size = path.stat().st_size
    file_digest = digest(path)
    total_bytes += size
    if size != entry["size_bytes"] or file_digest != entry["sha256"]:
        errors.append("drift:" + entry["path"])
aggregate = hashlib.sha256(
    json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

sumo_dir = Path({remote_sumo!r})
edge_xml = Path({remote_edge!r})
evaluator_sources = {{
    str(path.resolve()) for path in evaluator.evaluation_source_paths(sumo_dir, edge_xml)
}}
matrix_sources = {{
    str(path.resolve()) for path in matrix.evaluation_sources(sumo_dir, edge_xml)
}}
runtime_lock, runtime_identity = load_sumo_runtime_lock(Path({remote_lock!r}))
active_runtime = sumo_runtime_fingerprint(
    FORMAL_V7_SUMO_BACKEND,
    expected_version=FORMAL_V7_SUMO_VERSION,
)

methods_path = Path({remote_methods!r})
methods_payload = json.loads(methods_path.read_text())
methods = matrix.load_development_methods(methods_path, matrix.METHODS_SCHEMA)
behavior_keys = {{
    (
        row["training_seed"],
        row["actor_tensor_sha256"],
        row["parent_base_hold_scale"],
        row["parent_headway_balance_gain"],
        row["parent_reserve_horizon_fraction"],
        row["tail_service_floor_seconds"],
        row["activation_start_fraction"],
        row["activation_end_fraction"],
        row["max_onboard_capacity_fraction"],
    )
    for row in methods
}}
candidate_methods = [
    row
    for row in methods
    if row["algorithm"] == "h2oplus-v17-tail-service-reserve-thirty"
]
parent_errors = []
for row in methods:
    expected_parent = (
        f"h2oplus-v16-headway-reserve-three-quarter-"
        f"seed{{row['training_seed']}}-events000025000"
    )
    if row["parent_method"] != expected_parent:
        parent_errors.append(row["name"])

source_freeze_path = (
    root / "H2Oplus/reproducibility/"
    "h2oplus_v17_tail_service_reserve_evaluation_source_freeze_v2.json"
)
source_freeze_sha256 = digest(source_freeze_path)
dispatch_path = (
    root / "H2Oplus/reproducibility/"
    "run_h2oplus_v17_tail_service_reserve_evaluation_dispatch_v2.py"
)
dispatch_text = dispatch_path.read_text()
compatibility_variants = [
    dispatch.v17_algorithm_variant(
        "h2oplus-v17-tail-service-reserve-" + suffix
    )
    for suffix in ("twenty", "thirty", "forty")
]
invalid_prefix_rejected = False
try:
    dispatch.v17_algorithm_variant("h2oplus-v17-invalid-thirty")
except ValueError:
    invalid_prefix_rejected = True
expected_environment = {environment!r}
observed_environment = {{key: os.environ.get(key) for key in expected_environment}}
print(json.dumps({{
    "hostname": os.uname().nodename,
    "python_version": platform.python_version(),
    "manifest_schema": manifest.get("schema_version"),
    "manifest_sha256": manifest_sha256,
    "aggregate_sha256": aggregate,
    "manifest_aggregate_sha256": manifest.get("aggregate_sha256"),
    "entry_count": len(entries),
    "verified_bytes": total_bytes,
    "manifest_total_file_bytes": manifest.get("total_file_bytes"),
    "file_errors": errors,
    "evaluator_source_file_count": len(evaluator_sources),
    "matrix_source_file_count": len(matrix_sources),
    "source_sets_equal": evaluator_sources == matrix_sources,
    "only_evaluator": sorted(evaluator_sources - matrix_sources),
    "only_matrix": sorted(matrix_sources - evaluator_sources),
    "runtime_signature": active_runtime["portable_signature"],
    "runtime_lock_signature": runtime_identity["runtime_signature"],
    "runtime_identity_equal": (
        active_runtime["portable_identity"]
        == runtime_lock["runtime_fingerprint"]["portable_identity"]
    ),
    "runtime_version": active_runtime["runtime_version"],
    "runtime_backend": active_runtime["backend"],
    "physical_method_count": len(methods),
    "logical_method_count": methods_payload.get("logical_method_count"),
    "alias_count": len(methods_payload.get("aliases", [])),
    "unique_behavior_count": len(behavior_keys),
    "unique_actor_tensor_count": len({{row["actor_tensor_sha256"] for row in methods}}),
    "parent_base_hold_scales": sorted(
        {{row["parent_base_hold_scale"] for row in methods}}
    ),
    "parent_headway_balance_gains": sorted(
        {{row["parent_headway_balance_gain"] for row in methods}}
    ),
    "parent_reserve_horizon_fractions": sorted(
        {{row["parent_reserve_horizon_fraction"] for row in methods}}
    ),
    "tail_service_floor_seconds": sorted(
        {{row["tail_service_floor_seconds"] for row in methods}}
    ),
    "activation_start_fractions": sorted(
        {{row["activation_start_fraction"] for row in methods}}
    ),
    "activation_end_fractions": sorted(
        {{row["activation_end_fraction"] for row in methods}}
    ),
    "max_onboard_capacity_fractions": sorted(
        {{row["max_onboard_capacity_fraction"] for row in methods}}
    ),
    "candidate_method_count": len(candidate_methods),
    "expected_new_remote_sumo_episodes": methods_payload.get(
        "expected_new_remote_sumo_episodes"
    ),
    "methods_paths_are_relative": all(
        not Path(row["checkpoint"]).is_absolute()
        for row in methods_payload.get("methods", [])
    ),
    "parent_errors": parent_errors,
    "prior_v14_v15_v16_development_outcomes_used_for_design": methods_payload.get(
        "prior_v14_v15_v16_development_outcomes_used_for_design"
    ),
    "formal_held_out_outcomes_read": methods_payload.get(
        "formal_held_out_outcomes_read"
    ),
    "v17_development_evaluation_outcomes_read": methods_payload.get(
        "v17_development_evaluation_outcomes_read"
    ),
    "source_freeze_sha256": source_freeze_sha256,
    "compatibility_variants": compatibility_variants,
    "invalid_prefix_rejected": invalid_prefix_rejected,
    "dispatch_uses_python39_string_api": (
        ".removeprefix(" in dispatch_text or ".removesuffix(" in dispatch_text
    ),
    "environment": observed_environment,
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
        and observation.get("manifest_schema")
        == "transit-h2oplus-v17-tail-service-reserve-evaluation-deployment-manifest-v2"
        and observation.get("python_version") == "3.8.20"
        and observation.get("manifest_sha256") == args.expected_manifest_sha256
        and observation.get("aggregate_sha256") == args.expected_aggregate_sha256
        and observation.get("manifest_aggregate_sha256") == args.expected_aggregate_sha256
        and observation.get("entry_count") == args.expected_entry_count
        and observation.get("verified_bytes")
        == observation.get("manifest_total_file_bytes")
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
        and observation.get("physical_method_count") == 15
        and observation.get("logical_method_count") == 15
        and observation.get("alias_count") == 15
        and observation.get("unique_behavior_count") == 15
        and observation.get("unique_actor_tensor_count") == 5
        and observation.get("parent_base_hold_scales") == [0.125]
        and observation.get("parent_headway_balance_gains") == [1.0]
        and observation.get("parent_reserve_horizon_fractions") == [0.75]
        and observation.get("tail_service_floor_seconds") == [20.0, 30.0, 40.0]
        and observation.get("activation_start_fractions") == [0.5]
        and observation.get("activation_end_fractions") == [0.75]
        and observation.get("max_onboard_capacity_fractions") == [0.25]
        and observation.get("candidate_method_count") == 5
        and observation.get("expected_new_remote_sumo_episodes") == 135
        and observation.get("methods_paths_are_relative") is True
        and observation.get("parent_errors") == []
        and observation.get(
            "prior_v14_v15_v16_development_outcomes_used_for_design"
        )
        is True
        and observation.get("formal_held_out_outcomes_read") is False
        and observation.get("v17_development_evaluation_outcomes_read") is False
        and observation.get("source_freeze_sha256")
        == args.expected_source_freeze_sha256
        and observation.get("compatibility_variants")
        == ["twenty", "thirty", "forty"]
        and observation.get("invalid_prefix_rejected") is True
        and observation.get("dispatch_uses_python39_string_api") is False
        and observation.get("environment") == environment
    )
    payload = {
        "schema_version": SCHEMA,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "passed": passed,
        "formal_held_out_outcomes_read": False,
        "local_training_or_sumo_executed": False,
        "remote_sumo_simulation_executed": False,
        "verification_scope": (
            "file hashes, exact source contract, Python 3.8 dispatch import and prefix "
            "parser, 15 tail-service behavior identities, sealed v16 parents/v14 "
            "actors, and runtime fingerprint only"
        ),
        "expected": {
            "manifest_sha256": args.expected_manifest_sha256,
            "aggregate_sha256": args.expected_aggregate_sha256,
            "entry_count": args.expected_entry_count,
            "source_file_count": EXPECTED_SOURCE_COUNT,
            "runtime_signature": EXPECTED_RUNTIME_SIGNATURE,
            "source_freeze_sha256": args.expected_source_freeze_sha256,
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
