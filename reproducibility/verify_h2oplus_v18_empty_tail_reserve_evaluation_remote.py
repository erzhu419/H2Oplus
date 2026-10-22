#!/usr/bin/env python3
"""Verify v18 on remote Python 3.8 without running a SUMO episode."""

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
SCHEMA = "transit-h2oplus-v18-empty-tail-reserve-evaluation-remote-verification-v2"
EXPECTED_RUNTIME_SIGNATURE = (
    "264d3c412b8e6c42320995208d2e009eb809b85626047206c4de752c2dae06e7"
)
EXPECTED_SOURCE_COUNT = 55


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
        / "deployments/h2oplus_v18_empty_tail_reserve_evaluation_v1/workspace",
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
        / "h2oplus_v18_empty_tail_reserve_evaluation_jtl110gpu_verification_v2.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scheduler_root = args.scheduler_root.expanduser().resolve(strict=True)
    deployment_root = args.deployment_root.expanduser().resolve(strict=True)
    manifest = deployment_root / "deployment_manifest.json"
    if sha256(manifest) != args.expected_manifest_sha256:
        raise ValueError("local v18 deployment manifest identity drifted")

    sys.path.insert(0, str(scheduler_root))
    scheduler = importlib.import_module("skill.scheduler")
    remote_root = str(deployment_root)
    remote_repo = f"{remote_root}/H2Oplus"
    remote_sumo = f"{remote_root}/SUMO_ruiguang/online_control"
    remote_edge = f"{remote_repo}/bus_h2o/network_data/a_sorted_busline_edge.xml"
    remote_lock = (
        f"{remote_repo}/reproducibility/"
        "h2oplus_v8_stage_a_sumo_runtime_lock_py38.json"
    )
    remote_methods = (
        f"{remote_repo}/reproducibility/"
        "h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json"
    )
    remote_environment = (
        f"{remote_repo}/reproducibility/"
        "h2oplus_v18_empty_tail_reserve_environment_v1.json"
    )
    environment = {
        "LD_PRELOAD": "/usr/lib/x86_64-linux-gnu/libcurl.so.4",
        "CC": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/x86_64-conda-linux-gnu-gcc",
        "CPATH": (
            "/home/erzhu419/mine_code/sumo-rl/H2Oplus/reproducibility/"
            "deployments/v7_formal_v2_training_v7r1/workspace/reproducibility/"
            "toolchain/libxcrypt-4.4.36-hd590300_1/include"
        ),
        "PATH": "/home/erzhu419/anaconda3/envs/LSTM-RL/bin:/usr/bin:/bin",
        "PYTHONPATH": remote_repo,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    remote_code = rf'''
import hashlib
import json
import os
import platform
from pathlib import Path

from SimpleSAC import evaluate_transit_v18_empty_tail_reserve as evaluator
from bus_h2o.sumo_provenance import (
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    load_sumo_runtime_lock,
    sumo_runtime_fingerprint,
)
from reproducibility import analyze_h2oplus_v18_empty_tail_reserve as analyzer
from reproducibility import run_evaluation_matrix_v18_empty_tail_reserve as matrix
from reproducibility import run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch as dispatch

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

root = Path({remote_root!r}).resolve(strict=True)
repo = root / "H2Oplus"
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
        row["algorithm"],
        row["empty_reserve_enabled"],
        row["empty_max_onboard_capacity_fraction"],
    )
    for row in methods
}}
variant_counts = {{
    variant: sum(
        row["algorithm"] == "h2oplus-v18-empty-tail-reserve-" + variant
        for row in methods
    )
    for variant in ("baseline", "empty-forty", "one-seat-forty")
}}
parent_errors = []
for row in methods:
    expected_parent = (
        f"h2oplus-v17-tail-service-reserve-twenty-"
        f"seed{{row['training_seed']}}-events000025000"
    )
    if row["v17_parent_method"] != expected_parent:
        parent_errors.append(row["name"])

environment_path = Path({remote_environment!r})
environment_payload = json.loads(environment_path.read_text())
blocks = dispatch.verify_design(environment_path, environment_payload)
source_freeze_path = (
    repo / "reproducibility/h2oplus_v18_empty_tail_reserve_source_freeze_v1.json"
)
source_freeze_sha256 = digest(source_freeze_path)
source_freeze = json.loads(source_freeze_path.read_text())
frozen_core_errors = []
for row in source_freeze.get("matrix_sources", []):
    matches = [
        path for path in matrix_sources
        if path.endswith(row["path"])
        and digest(path) == row["sha256"]
    ]
    if len(matches) != 1:
        frozen_core_errors.append(row["path"])
dispatch_path = (
    repo / "reproducibility/"
    "run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch.py"
)
scientific_text = "\n".join(
    (repo / row["path"]).read_text()
    for row in source_freeze.get("matrix_sources", [])
    if row["path"].endswith(".py")
)
scientific_text += "\n" + dispatch_path.read_text()
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
    "frozen_core_errors": frozen_core_errors,
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
    "variant_counts": variant_counts,
    "empty_reserve_enabled_values": sorted(
        {{row["empty_reserve_enabled"] for row in methods}}
    ),
    "empty_reserve_floor_seconds": sorted(
        {{row["empty_reserve_floor_seconds"] for row in methods}}
    ),
    "empty_activation_start_fractions": sorted(
        {{row["empty_activation_start_fraction"] for row in methods}}
    ),
    "empty_activation_end_fractions": sorted(
        {{row["empty_activation_end_fraction"] for row in methods}}
    ),
    "empty_max_onboard_capacity_fractions": sorted(
        {{row["empty_max_onboard_capacity_fraction"] for row in methods}}
    ),
    "expected_new_remote_sumo_episodes": methods_payload.get(
        "expected_new_remote_sumo_episodes"
    ),
    "methods_paths_are_relative": all(
        not Path(row["checkpoint"]).is_absolute()
        for row in methods_payload.get("methods", [])
    ),
    "parent_errors": parent_errors,
    "environment_block_count": len(blocks),
    "environment_sumo_seeds": sorted(row["sumo_seed"] for row in blocks),
    "environment_is_disjoint": environment_payload.get("disjointness", {{}}).get(
        "verified_disjoint"
    ),
    "prior_development_outcomes_used_for_design": methods_payload.get(
        "prior_v14_v15_v16_v17_development_outcomes_used_for_design"
    ),
    "formal_held_out_outcomes_read": methods_payload.get(
        "formal_held_out_outcomes_read"
    ),
    "v18_development_evaluation_outcomes_read": methods_payload.get(
        "v18_development_evaluation_outcomes_read"
    ),
    "source_freeze_sha256": source_freeze_sha256,
    "analyzer_sha256": digest(Path(analyzer.__file__).resolve()),
    "frozen_analyzer_sha256": source_freeze.get("analyzer", {{}}).get("sha256"),
    "uses_python39_string_api": (
        ".removeprefix(" in scientific_text or ".removesuffix(" in scientific_text
    ),
    "environment": observed_environment,
}}, sort_keys=True))
'''
    prefix = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in environment.items()
    )
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
        == "transit-h2oplus-v18-empty-tail-reserve-evaluation-deployment-manifest-v1"
        and observation.get("python_version") == "3.8.20"
        and observation.get("manifest_sha256") == args.expected_manifest_sha256
        and observation.get("aggregate_sha256") == args.expected_aggregate_sha256
        and observation.get("manifest_aggregate_sha256")
        == args.expected_aggregate_sha256
        and observation.get("entry_count") == args.expected_entry_count
        and observation.get("verified_bytes")
        == observation.get("manifest_total_file_bytes")
        and observation.get("file_errors") == []
        and observation.get("evaluator_source_file_count") == EXPECTED_SOURCE_COUNT
        and observation.get("matrix_source_file_count") == EXPECTED_SOURCE_COUNT
        and observation.get("source_sets_equal") is True
        and observation.get("only_evaluator") == []
        and observation.get("only_matrix") == []
        and observation.get("frozen_core_errors") == []
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
        and observation.get("variant_counts")
        == {"baseline": 5, "empty-forty": 5, "one-seat-forty": 5}
        and observation.get("empty_reserve_enabled_values") == [False, True]
        and observation.get("empty_reserve_floor_seconds") == [40.0]
        and observation.get("empty_activation_start_fractions") == [0.25]
        and observation.get("empty_activation_end_fractions") == [0.75]
        and observation.get("empty_max_onboard_capacity_fractions") == [0.0, 0.02]
        and observation.get("expected_new_remote_sumo_episodes") == 135
        and observation.get("methods_paths_are_relative") is True
        and observation.get("parent_errors") == []
        and observation.get("environment_block_count") == 3
        and observation.get("environment_sumo_seeds") == [7101, 7102, 7103]
        and observation.get("environment_is_disjoint") is True
        and observation.get("prior_development_outcomes_used_for_design") is True
        and observation.get("formal_held_out_outcomes_read") is False
        and observation.get("v18_development_evaluation_outcomes_read") is False
        and observation.get("source_freeze_sha256")
        == args.expected_source_freeze_sha256
        and observation.get("analyzer_sha256")
        == observation.get("frozen_analyzer_sha256")
        and observation.get("uses_python39_string_api") is False
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
            "file hashes, exact 55-file source equality, Python 3.8 imports, "
            "15 v18 behavior identities, nine fresh development cells and SUMO "
            "runtime fingerprint only"
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
