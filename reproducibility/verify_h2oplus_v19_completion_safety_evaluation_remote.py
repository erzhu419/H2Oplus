#!/usr/bin/env python3
"""Verify v19-v2 on remote Python 3.8 without running a SUMO episode."""

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
SCHEMA = "transit-h2oplus-v19-completion-safety-evaluation-remote-verification-v1"
EXPECTED_RUNTIME_SIGNATURE = (
    "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
)
EXPECTED_SOURCE_COUNT = 55
EXPECTED_PROTOCOL = (
    "fa5f629588d35b1435e28bd39931f49d9851811ea3e308d62672f9eacf1f7184"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite v19 remote verification: {path}")
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
        / "deployments/h2oplus_v19_completion_safety_evaluation_v2/workspace",
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
        / "h2oplus_v19_completion_safety_evaluation_deployment_v2_jtl110gpu_verification_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    manifest = deployment_root / "deployment_manifest.json"
    if sha256(manifest) != args.expected_manifest_sha256:
        raise ValueError("local v19-v2 deployment manifest drifted")
    sys.path.insert(0, str(scheduler_root))
    scheduler = importlib.import_module("skill.scheduler")

    remote_root = str(deployment_root)
    remote_repo = f"{remote_root}/H2Oplus"
    remote_sumo = f"{remote_root}/SUMO_ruiguang/online_control"
    remote_edge = f"{remote_repo}/bus_h2o/network_data/a_sorted_busline_edge.xml"
    remote_lock = f"{remote_repo}/reproducibility/h2oplus_v8_stage_a_sumo_runtime_lock_py38.json"
    remote_methods = f"{remote_repo}/reproducibility/h2oplus_v19_completion_safety_evaluation_methods_v1.json"
    remote_dataset = f"{remote_repo}/bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"
    remote_scenario = (
        f"{remote_repo}/bus_h2o/development_scenarios_v19/completion_safety_v1/"
        "demand_0p85_jitter120s_seed7401/manifest_portable_v8.json"
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
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "TORCH_NUM_THREADS": "1",
        "H2OPLUS_V19_COMPLETION_SAFETY_PROTOCOL_SHA256": EXPECTED_PROTOCOL,
    }
    remote_code = rf'''
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from SimpleSAC import evaluate_transit_v19_completion_safety_reserve as evaluator
from bus_h2o.sumo_provenance import (
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    load_sumo_runtime_lock,
    sumo_runtime_fingerprint,
)
from reproducibility import run_evaluation_matrix_v19_completion_safety as matrix
from reproducibility import run_h2oplus_v19_completion_safety_evaluation_dispatch as dispatch

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

root = Path({remote_root!r}).resolve(strict=True)
repo = Path({remote_repo!r}).resolve(strict=True)
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
    total_bytes += size
    if size != entry["size_bytes"] or digest(path) != entry["sha256"]:
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
candidate_methods = [
    row for row in methods
    if row["algorithm"] == "h2oplus-v19-completion-safety-quarter-load-sixty"
]
behavior_keys = {{
    (
        row["training_seed"],
        row["actor_tensor_sha256"],
        row["v19_reserve_enabled"],
        row["v19_reserve_floor_seconds"],
        row["v19_activation_start_fraction"],
        row["v19_activation_end_fraction"],
        row["v19_max_onboard_fraction"],
    )
    for row in methods
}}
source_freeze_path = repo / "reproducibility/h2oplus_v19_completion_safety_source_freeze_v2.json"
source_freeze_sha256 = digest(source_freeze_path)
source_freeze = json.loads(source_freeze_path.read_text())
freeze_sources = {{
    row["path"]: row["sha256"]
    for row in source_freeze["matrix_sources"] + source_freeze["operational_sources"]
}}
freeze_source_errors = [
    relative for relative, expected in freeze_sources.items()
    if digest(repo / relative) != expected
]

dataset = Path({remote_dataset!r}).resolve(strict=True)
scenario = Path({remote_scenario!r}).resolve(strict=True)
dataset_hash = digest(dataset)
stat = dataset.stat()
probe = r"""
import json
import sys
from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7
from SimpleSAC import evaluate_transit_v19_completion_safety_reserve as target

expected = json.loads(sys.argv[1])
dataset_hash = sys.argv[2]

def no_sumo_base_main():
    args = target.base.parse_args()
    _, evidence = target.base.build_policy(
        args, TransitMDPConfigV7(), None, dataset_hash
    )
    reserve = evidence["completion_safety_reserve"]
    observed = {{
        "parent_floor": reserve["parent_tail_service_floor_seconds"],
        "enabled": reserve["reserve_enabled"],
        "reserve_floor": reserve["reserve_floor_seconds"],
        "start": reserve["activation_start_fraction"],
        "end": reserve["activation_end_fraction"],
        "max_onboard": reserve["max_onboard_capacity_fraction"],
        "parent_method": evidence["parent_method"],
        "protocol": reserve["v19_protocol_sha256"],
        "formal_held_out_outcomes_read": reserve["formal_held_out_outcomes_read"],
    }}
    if observed != expected:
        raise RuntimeError(json.dumps({{"observed": observed, "expected": expected}}))
    print(json.dumps({{"v19_policy_build": observed}}, sort_keys=True))

target.base.main = no_sumo_base_main
sys.argv = [str(target.__file__), *sys.argv[3:]]
target.main()
"""
expected_flag_counts = {{
    "--expected-v19-parent-floor-seconds": 1,
    "--expected-v19-reserve-enabled": 1,
    "--expected-v19-reserve-floor-seconds": 1,
    "--expected-v19-activation-start-fraction": 1,
    "--expected-v19-activation-end-fraction": 1,
    "--expected-v19-max-onboard-fraction": 1,
    "--expected-v19-parent-method-name": 1,
}}
policy_builds = []
for index, method in enumerate(methods):
    smoke_output = Path("/tmp") / (
        "h2oplus_v19_policy_build_" + str(os.getpid()) + "_" + str(index) + ".json"
    )
    if smoke_output.exists():
        smoke_output.unlink()
    job = {{
        "evaluator": repo / "SimpleSAC/evaluate_transit_v19_completion_safety_reserve.py",
        "method": method,
        "sumo_seed": 7301,
        "scenario": {{"path": scenario}},
        "dataset": dataset,
        "output": smoke_output,
        "signature": "v19-policy-build-" + str(index),
        "device": "cpu",
        "max_events": 0,
        "max_sim_time_seconds": 0,
        "sumo_dir": sumo_dir,
        "edge_xml": edge_xml,
        "mdp_version": "v7",
        "sumo_backend": "libsumo",
        "sumo_runtime_version": active_runtime["runtime_version"],
        "sumo_runtime_signature": active_runtime["portable_signature"],
        "sumo_runtime_lock": Path({remote_lock!r}),
        "sumo_runtime_lock_sha256": digest(Path({remote_lock!r})),
        "dataset_sha256": dataset_hash,
        "dataset_size_bytes": stat.st_size,
        "dataset_mtime_ns": stat.st_mtime_ns,
        "event_log": None,
    }}
    smoke_command = matrix.job_command(job)
    expected = {{
        "parent_floor": method["tail_service_floor_seconds"],
        "enabled": bool(method["v19_reserve_enabled"]),
        "reserve_floor": method["v19_reserve_floor_seconds"],
        "start": method["v19_activation_start_fraction"],
        "end": method["v19_activation_end_fraction"],
        "max_onboard": method["v19_max_onboard_fraction"],
        "parent_method": method["v17_parent_method"],
        "protocol": {EXPECTED_PROTOCOL!r},
        "formal_held_out_outcomes_read": False,
    }}
    flag_counts = {{flag: smoke_command.count(flag) for flag in expected_flag_counts}}
    completed = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(expected), dataset_hash, *smoke_command[2:]],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
        timeout=120,
    )
    observed = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except Exception:
            continue
        if "v19_policy_build" in candidate:
            observed = candidate["v19_policy_build"]
            break
    output_created = smoke_output.exists()
    if output_created:
        smoke_output.unlink()
    policy_builds.append({{
        "name": method["name"],
        "algorithm": method["algorithm"],
        "training_seed": method["training_seed"],
        "returncode": completed.returncode,
        "flag_counts": flag_counts,
        "expected": expected,
        "observed": observed,
        "output_created": output_created,
        "stdout_tail": completed.stdout[-2000:],
    }})

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
    "freeze_source_errors": freeze_source_errors,
    "manifest_freeze_sources_equal": manifest.get("evaluation_source_sha256") == freeze_sources,
    "runtime_signature": active_runtime["portable_signature"],
    "runtime_lock_signature": runtime_identity["runtime_signature"],
    "runtime_identity_equal": active_runtime["portable_identity"] == runtime_lock["runtime_fingerprint"]["portable_identity"],
    "runtime_version": active_runtime["runtime_version"],
    "runtime_backend": active_runtime["backend"],
    "physical_method_count": len(methods),
    "logical_method_count": methods_payload.get("logical_method_count"),
    "behavior_key_count": len(behavior_keys),
    "candidate_method_count": len(candidate_methods),
    "source_freeze_sha256": source_freeze_sha256,
    "source_freeze_revision": source_freeze.get("evidence_revision"),
    "source_freeze_expected_runtime_sources": source_freeze.get("expected_matrix_source_file_count"),
    "dispatch_execution_schema": dispatch.EXECUTION_SCHEMA,
    "dispatch_summary_schema": dispatch.SUMMARY_SCHEMA,
    "dispatch_required_environment": dispatch.REQUIRED_ENVIRONMENT,
    "policy_build_count": len(policy_builds),
    "policy_builds": policy_builds,
    "all_policy_builds_passed": all(
        row["returncode"] == 0
        and row["flag_counts"] == expected_flag_counts
        and row["observed"] == row["expected"]
        and row["output_created"] is False
        for row in policy_builds
    ),
    "expected_environment": expected_environment,
    "observed_environment": observed_environment,
    "environment_matches": observed_environment == expected_environment,
    "formal_held_out_outcomes_read": False,
    "remote_sumo_simulation_executed": False,
}}, sort_keys=True))
'''
    env_prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in environment.items()
    )
    command = (
        f"env {env_prefix} /home/erzhu419/anaconda3/envs/LSTM-RL/bin/python "
        f"-c {shlex.quote(remote_code)}"
    )
    rc, stdout, stderr = scheduler.run_on(
        args.node, command, timeout=1200, check=False
    )
    observation: dict[str, Any] = {}
    parse_error = None
    if rc == 0:
        try:
            observation = json.loads(stdout.strip().splitlines()[-1])
        except Exception as error:
            parse_error = f"{type(error).__name__}: {error}"
    required_environment = {
        key: environment[key]
        for key in (
            "LD_PRELOAD",
            "CC",
            "CPATH",
            "PYTHONDONTWRITEBYTECODE",
            "PYTHONUNBUFFERED",
            "CUDA_VISIBLE_DEVICES",
            "H2OPLUS_V19_COMPLETION_SAFETY_PROTOCOL_SHA256",
        )
    }
    passed = bool(
        rc == 0
        and parse_error is None
        and observation.get("hostname") == "huiwei-Super-Server"
        and observation.get("python_version") == "3.8.20"
        and observation.get("manifest_schema")
        == "transit-h2oplus-v19-completion-safety-evaluation-deployment-manifest-v2"
        and observation.get("manifest_sha256") == args.expected_manifest_sha256
        and observation.get("aggregate_sha256") == args.expected_aggregate_sha256
        and observation.get("manifest_aggregate_sha256")
        == args.expected_aggregate_sha256
        and observation.get("entry_count") == args.expected_entry_count
        and observation.get("file_errors") == []
        and observation.get("verified_bytes")
        == observation.get("manifest_total_file_bytes")
        and observation.get("evaluator_source_file_count") == EXPECTED_SOURCE_COUNT
        and observation.get("matrix_source_file_count") == EXPECTED_SOURCE_COUNT
        and observation.get("source_sets_equal") is True
        and observation.get("only_evaluator") == []
        and observation.get("only_matrix") == []
        and observation.get("freeze_source_errors") == []
        and observation.get("manifest_freeze_sources_equal") is True
        and observation.get("runtime_signature") == EXPECTED_RUNTIME_SIGNATURE
        and observation.get("runtime_lock_signature") == EXPECTED_RUNTIME_SIGNATURE
        and observation.get("runtime_identity_equal") is True
        and observation.get("runtime_version") == "1.25.0"
        and observation.get("runtime_backend") == "libsumo"
        and observation.get("physical_method_count") == 15
        and observation.get("logical_method_count") == 15
        and observation.get("behavior_key_count") == 15
        and observation.get("candidate_method_count") == 5
        and observation.get("source_freeze_sha256")
        == args.expected_source_freeze_sha256
        and observation.get("source_freeze_revision")
        == "v2-exact-protocol-environment-binding"
        and observation.get("source_freeze_expected_runtime_sources") == 55
        and observation.get("dispatch_execution_schema")
        == "transit-h2oplus-v19-completion-safety-evaluation-host-execution-v1"
        and observation.get("dispatch_summary_schema")
        == "transit-h2oplus-v19-completion-safety-evaluation-host-summary-v1"
        and observation.get("dispatch_required_environment") == required_environment
        and observation.get("policy_build_count") == 15
        and observation.get("all_policy_builds_passed") is True
        and observation.get("environment_matches") is True
        and observation.get("formal_held_out_outcomes_read") is False
        and observation.get("remote_sumo_simulation_executed") is False
    )
    report = {
        "schema_version": SCHEMA,
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "passed": passed,
        "returncode": rc,
        "stderr": stderr.strip(),
        "parse_error": parse_error,
        "observation": observation,
        "expected": {
            "manifest_sha256": args.expected_manifest_sha256,
            "aggregate_sha256": args.expected_aggregate_sha256,
            "entry_count": args.expected_entry_count,
            "source_file_count": EXPECTED_SOURCE_COUNT,
            "source_freeze_sha256": args.expected_source_freeze_sha256,
            "runtime_signature": EXPECTED_RUNTIME_SIGNATURE,
            "policy_build_count": 15,
        },
        "verification_scope": (
            "927 deployment files, exact 55-file source equality, runtime identity, "
            "15 physical behaviors and 15 no-SUMO actor policy builds"
        ),
        "formal_held_out_outcomes_read": False,
        "remote_sumo_simulation_executed": False,
    }
    atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
