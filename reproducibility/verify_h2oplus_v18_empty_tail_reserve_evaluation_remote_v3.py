#!/usr/bin/env python3
"""Verify v18-v3 on remote Python 3.8 without running a SUMO episode."""

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
SCHEMA = "transit-h2oplus-v18-empty-tail-reserve-evaluation-remote-verification-v5"
EXPECTED_RUNTIME_SIGNATURE = (
    "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
)
EXPECTED_SOURCE_COUNT = 59


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
        default=REPRO
        / "deployments/h2oplus_v18_empty_tail_reserve_evaluation_v3/workspace",
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
        / "h2oplus_v18_empty_tail_reserve_evaluation_deployment_v3_jtl110gpu_verification_v5.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    manifest = deployment_root / "deployment_manifest.json"
    if sha256(manifest) != args.expected_manifest_sha256:
        raise ValueError("local v18-v3 deployment manifest drifted")
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
        "h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json"
    )
    remote_dataset = (
        f"{remote_repo}/bus_h2o/datasets_v7/formal_v2_full/"
        "transit_smdp_v7_formal_v2.h5"
    )
    remote_scenario = (
        f"{remote_repo}/bus_h2o/development_scenarios_v18/empty_tail_reserve_v1/"
        "demand_0p85_jitter120s_seed7201/manifest_portable_v8.json"
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
    }
    remote_code = rf'''
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from SimpleSAC import evaluate_transit_v18_empty_tail_reserve_v3 as evaluator
from bus_h2o.sumo_provenance import (
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    load_sumo_runtime_lock,
    sumo_runtime_fingerprint,
)
from reproducibility import run_evaluation_matrix_v18_empty_tail_reserve_v3 as matrix
from reproducibility import run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch_v3 as dispatch

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
    if row["algorithm"] == "h2oplus-v18-empty-tail-reserve-empty-forty"
]
behavior_keys = {{
    (
        row["training_seed"],
        row["actor_tensor_sha256"],
        row["empty_reserve_enabled"],
        row["empty_reserve_floor_seconds"],
        row["empty_activation_start_fraction"],
        row["empty_activation_end_fraction"],
        row["empty_max_onboard_capacity_fraction"],
    )
    for row in methods
}}
source_freeze_path = (
    repo / "reproducibility/h2oplus_v18_empty_tail_reserve_source_freeze_v3.json"
)
source_freeze_sha256 = digest(source_freeze_path)
source_freeze = json.loads(source_freeze_path.read_text())

dataset = Path({remote_dataset!r}).resolve(strict=True)
scenario = Path({remote_scenario!r}).resolve(strict=True)
stat = dataset.stat()
probe = r"""
import json
import sys
from SimpleSAC import evaluate_transit_v18_empty_tail_reserve_v3 as target

expected = json.loads(sys.argv[1])

def frozen_validator_probe(args, _config, _station_max, _dataset_hash):
    observed = {{
        "parent_floor": args.expected_parent_tail_service_floor_seconds,
        "enabled": args.expected_empty_reserve_enabled,
        "empty_floor": args.expected_empty_reserve_floor_seconds,
        "start": args.expected_activation_start_fraction,
        "end": args.expected_activation_end_fraction,
        "max_onboard": args.expected_max_onboard_capacity_fraction,
        "parent_method": args.expected_v17_parent_method_name,
    }}
    if observed != expected:
        raise RuntimeError(json.dumps({{"observed": observed, "expected": expected}}))
    print(json.dumps({{"v18_final_binding": observed}}, sort_keys=True))
    return None, {{}}

def no_sumo_base_main():
    args = target.base.parse_args()
    target.base.build_policy(args, None, None, None)

target.v1.build_development_policy = frozen_validator_probe
target.base.main = no_sumo_base_main
sys.argv = [str(target.__file__), *sys.argv[2:]]
target.main()
"""
expected_flag_counts = {{
    "--expected-activation-start-fraction": 1,
    "--expected-empty-activation-start-fraction": 1,
    "--expected-activation-end-fraction": 1,
    "--expected-empty-activation-end-fraction": 1,
    "--expected-max-onboard-capacity-fraction": 1,
    "--expected-empty-max-onboard-capacity-fraction": 1,
}}
binding_smokes = []
for index, method in enumerate(methods):
    smoke_output = Path("/tmp") / (
        "h2oplus_v18_v3_binding_smoke_" + str(os.getpid()) + "_" + str(index) + ".json"
    )
    if smoke_output.exists():
        smoke_output.unlink()
    job = {{
        "evaluator": repo / "SimpleSAC/evaluate_transit_v18_empty_tail_reserve_v3.py",
        "method": method,
        "sumo_seed": 7101,
        "scenario": {{"path": scenario}},
        "dataset": dataset,
        "output": smoke_output,
        "signature": "v18-v3-final-binding-smoke-" + str(index),
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
        "dataset_sha256": digest(dataset),
        "dataset_size_bytes": stat.st_size,
        "dataset_mtime_ns": stat.st_mtime_ns,
        "event_log": None,
    }}
    smoke_command = matrix.job_command(job)
    expected = {{
        "parent_floor": method["tail_service_floor_seconds"],
        "enabled": bool(method["empty_reserve_enabled"]),
        "empty_floor": method["empty_reserve_floor_seconds"],
        "start": method["empty_activation_start_fraction"],
        "end": method["empty_activation_end_fraction"],
        "max_onboard": method["empty_max_onboard_capacity_fraction"],
        "parent_method": method["v17_parent_method"],
    }}
    flag_counts = {{flag: smoke_command.count(flag) for flag in expected_flag_counts}}
    completed = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(expected), *smoke_command[2:]],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        timeout=120,
    )
    observed = None
    for line in reversed(completed.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except Exception:
            continue
        if "v18_final_binding" in candidate:
            observed = candidate["v18_final_binding"]
            break
    output_created = smoke_output.exists()
    if output_created:
        smoke_output.unlink()
    binding_smokes.append({{
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
    "behavior_key_count": len(behavior_keys),
    "candidate_method_count": len(candidate_methods),
    "source_freeze_sha256": source_freeze_sha256,
    "source_freeze_revision": source_freeze.get("evidence_revision"),
    "source_freeze_expected_runtime_sources": source_freeze.get("expected_matrix_source_file_count"),
    "dispatch_execution_schema": dispatch.EXECUTION_SCHEMA,
    "dispatch_summary_schema": dispatch.SUMMARY_SCHEMA,
    "binding_smoke_count": len(binding_smokes),
    "binding_smokes": binding_smokes,
    "all_binding_smokes_passed": all(
        row["returncode"] == 0
        and row["flag_counts"] == expected_flag_counts
        and row["observed"] == row["expected"]
        and row["output_created"] is False
        for row in binding_smokes
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
        args.node, command, timeout=900, check=False
    )
    observation: dict[str, Any] = {}
    parse_error = None
    if rc == 0:
        try:
            observation = json.loads(stdout.strip().splitlines()[-1])
        except Exception as error:
            parse_error = f"{type(error).__name__}: {error}"
    passed = bool(
        rc == 0
        and parse_error is None
        and observation.get("hostname") == "huiwei-Super-Server"
        and observation.get("python_version") == "3.8.20"
        and observation.get("manifest_schema")
        == "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-manifest-v3"
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
        == "v3-final-namespace-binding-correction"
        and observation.get("source_freeze_expected_runtime_sources") == 59
        and observation.get("dispatch_execution_schema")
        == "transit-h2oplus-v18-empty-tail-reserve-evaluation-host-execution-v3"
        and observation.get("dispatch_summary_schema")
        == "transit-h2oplus-v18-empty-tail-reserve-evaluation-host-summary-v3"
        and observation.get("binding_smoke_count") == 15
        and observation.get("all_binding_smokes_passed") is True
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
            "binding_smoke_count": 15,
        },
        "verification_scope": (
            "836 deployment files, exact 59-file source equality, runtime identity, "
            "15 physical behaviors and 15 final-boundary parser probes"
        ),
        "formal_held_out_outcomes_read": False,
        "remote_sumo_simulation_executed": False,
    }
    atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
