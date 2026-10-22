#!/usr/bin/env python3
"""Build the no-SUMO independent design audit for the v20 control matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.run_evaluation_matrix_v20_full_control import (  # noqa: E402
    ALGORITHMS,
    CANDIDATE,
    DIRECT_PARENT,
    METHODS_SCHEMA,
    RAW_PARENT,
    SEEDS,
    load_development_methods,
)


EXPECTED = {
    "authorization": "ab535766ffad546c07e4291bfd2f0519d7cfa0e83f4cb3f4022c369d4cb19a06",
    "rationale": "e93a4114760e8755d27dcd7042ca666f4aefdf17e03d3bb55990068cf552998c",
    "registration": "f930a3612720097d4f5e5b635501405245f04ffd54445f389a23800d06f7f7d8",
    "protocol": "7211d32aeb1416840de442ca43fa997a918dfa282afcec5009091f78834ffa5d",
    "methods": "77162e22cc3b748fc2ad65a229ebca5c9212fd2af172b37ba81cdaded2d2e2e9",
    "environment": "7210826548c94aaa8fbbbb73b62eec5450fec131f2ae3dd1d1dbc93922dff2ba",
    "v19_audit": "267370a31bed2bc8053842dc28e9223ae76c517ef3877603b3c9c81588b8636b",
    "v19_closeout": "3e95e4cf964983257fd3fabc5cab2987e1a249b5b9a25d3ae455f081932b3f74",
    "v19_verification": "ec44263eebc8ba61049973f34aadf5154d0b07f5ccc096d3659662002b95f9ab",
    "data_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
}
PATHS = {
    "authorization": REPRO / "h2oplus_v20_full_control_authorization_v1.json",
    "rationale": REPRO / "h2oplus_v20_full_control_rationale_v1.json",
    "registration": REPRO / "h2oplus_v20_full_control_registration_v1.json",
    "protocol": REPRO / "h2oplus_v20_full_control_protocol_v1.json",
    "methods": REPRO / "h2oplus_v20_full_control_evaluation_methods_v1.json",
    "environment": REPRO / "h2oplus_v20_full_control_environment_v1.json",
    "v19_audit": REPRO / "h2oplus_v19_completion_safety_results/evaluation_audit_v2.json",
    "v19_closeout": REPRO / "h2oplus_v19_completion_safety_results/evaluation_closeout_v2.json",
    "v19_verification": REPRO / (
        "h2oplus_v19_completion_safety_results/"
        "evaluation_closeout_verification_v2.json"
    ),
    "data_audit": REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
    "data_verification": REPRO / (
        "h2oplus_v17_data_environment_convergence_verification_v1.json"
    ),
}
DEFAULT_OUTPUT = REPRO / "h2oplus_v20_full_control_design_audit_v1.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_inputs() -> tuple[dict[str, dict], dict[str, dict]]:
    payloads, bindings = {}, {}
    for name, path in PATHS.items():
        observed = sha256(path.resolve(strict=True))
        if observed != EXPECTED[name]:
            raise ValueError(f"frozen v20 design input drifted: {name}: {observed}")
        payloads[name] = json.loads(path.read_text())
        bindings[name] = descriptor(path)
    return payloads, bindings


def environment_audit(environment_path: Path, payload: dict) -> dict:
    blocks = payload.get("blocks", [])
    names, sumo_seeds, demand_seeds = set(), set(), set()
    cells, manifests = set(), []
    for block in blocks:
        name = block.get("name")
        sumo_seed = block.get("sumo_seed")
        demand_seed = block.get("demand_realization_seed")
        if (
            not isinstance(name, str)
            or name in names
            or not isinstance(sumo_seed, int)
            or isinstance(sumo_seed, bool)
            or sumo_seed in sumo_seeds
            or not isinstance(demand_seed, int)
            or isinstance(demand_seed, bool)
            or demand_seed in demand_seeds
        ):
            raise ValueError(f"malformed v20 environment block: {block!r}")
        paths = block.get("scenario_manifests", [])
        if len(paths) != 3:
            raise ValueError(f"v20 block does not contain three scales: {name}")
        scales = set()
        for raw in paths:
            path = (environment_path.parent / raw).resolve(strict=True)
            manifest = json.loads(path.read_text())
            scale = float(manifest.get("demand_scale"))
            if (
                manifest.get("schema_version") != "transit-passenger-demand-v1"
                or manifest.get("selection_seed") != demand_seed
                or float(manifest.get("depart_jitter_seconds")) != 120.0
                or scale not in {0.85, 1.0, 1.15}
            ):
                raise ValueError(f"v20 scenario identity drifted: {path}")
            cell = (sumo_seed, demand_seed, scale)
            if cell in cells:
                raise ValueError(f"duplicate v20 environment cell: {cell}")
            cells.add(cell)
            scales.add(scale)
            manifests.append(descriptor(path))
        if scales != {0.85, 1.0, 1.15}:
            raise ValueError(f"v20 demand-scale grid drifted: {name}")
        names.add(name)
        sumo_seeds.add(sumo_seed)
        demand_seeds.add(demand_seed)
    overlaps = {
        key: value
        for key, value in payload.get("disjointness", {}).items()
        if key.endswith("_overlap") and value
    }
    expected_names = {
        f"v20-full-control-development-{index:02d}" for index in range(1, 4)
    }
    if (
        names != expected_names
        or sumo_seeds != {7501, 7502, 7503}
        or demand_seeds != {7601, 7602, 7603}
        or len(cells) != 9
        or overlaps
        or payload.get("disjointness", {}).get("verified_disjoint") is not True
    ):
        raise ValueError("v20 fresh-environment disjointness drifted")
    return {
        "block_count": len(names),
        "environment_cell_count": len(cells),
        "sumo_seeds": sorted(sumo_seeds),
        "demand_realization_seeds": sorted(demand_seeds),
        "demand_scales": [0.85, 1.0, 1.15],
        "scenario_manifests": sorted(manifests, key=lambda item: item["path"]),
        "verified_disjoint": True,
        "nonempty_overlap_fields": overlaps,
    }


def method_audit(path: Path) -> dict:
    rows = load_development_methods(path, METHODS_SCHEMA)
    pairs = {(row["algorithm"], row["training_seed"]) for row in rows}
    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS}
    actors = {row["actor_tensor_sha256"] for row in rows}
    candidate = [row for row in rows if row["algorithm"] == CANDIDATE]
    direct = [row for row in rows if row["algorithm"] == DIRECT_PARENT]
    raw = [row for row in rows if row["algorithm"] == RAW_PARENT]
    if (
        len(rows) != 45
        or pairs != expected
        or len(candidate) != 5
        or len(direct) != 5
        or len(raw) != 5
        or any(row.get("online_events") != 25_000 for row in rows)
        or any(row.get("v20_role") != "predesignated-candidate" for row in candidate)
    ):
        raise ValueError("v20 physical method grid drifted")
    for seed in SEEDS:
        actor_hashes = {
            row["actor_tensor_sha256"]
            for row in rows
            if row["training_seed"] == seed
            and row["algorithm"] in {CANDIDATE, DIRECT_PARENT, RAW_PARENT}
        }
        if len(actor_hashes) != 1:
            raise ValueError(f"v20 parent actor lineage drifted for seed {seed}")
    return {
        "physical_method_count": len(rows),
        "algorithm_count": len(ALGORITHMS),
        "algorithms": list(ALGORITHMS),
        "training_seeds": list(SEEDS),
        "algorithm_seed_grid_complete": True,
        "unique_actor_tensor_count": len(actors),
        "candidate_direct_and_raw_parent_actor_matched_by_seed": True,
        "checkpoint_selection_by_behavior": False,
    }


def main() -> int:
    args = parse_args()
    payloads, bindings = load_inputs()
    authorization = payloads["authorization"]
    registration = payloads["registration"]
    protocol = payloads["protocol"]
    environment = payloads["environment"]
    v19_audit = payloads["v19_audit"]
    v19_closeout = payloads["v19_closeout"]
    v19_verification = payloads["v19_verification"]
    data_audit = payloads["data_audit"]
    data_verification = payloads["data_verification"]
    scope = authorization.get("authorization_scope", {})
    if (
        authorization.get("authorization_text")
        != "\u90a3\u5c31\u4e0b\u4e00\u6b65"
        or authorization.get("formal_held_out_outcomes_read") is not False
        or scope.get("remote_development_evaluation_authorized") is not True
        or scope.get("remote_sumo_episode_cap") != 405
        or scope.get("unique_task_count_cap") != 1
        or scope.get("remote_host") != "jtl110gpu"
        or scope.get("cpu_cores") != 12
        or scope.get("ram_mb") != 32768
        or scope.get("gpu_count") != 0
        or registration.get("matrix_type") != "full-development-control-matrix"
        or registration.get("new_training_runs") != 0
        or registration.get("formal_confirmation_stage_authorized_or_submitted")
        is not False
        or protocol.get("v20_development_evaluation_outcomes_read") is not False
        or v19_audit.get("artifact_integrity", {}).get("passed") is not True
        or v19_audit.get("parent_training_stability", {}).get("passed") is not True
        or v19_closeout.get("passed") is not True
        or v19_closeout.get("eligible_for_full_development_control_matrix") is not True
        or v19_verification.get("passed") is not True
        or v19_verification.get("artifact_integrity_verified") is not True
        or v19_verification.get("parent_training_stability_verified") is not True
        or data_audit.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_verification.get("passed") is not True
    ):
        raise ValueError("v20 authorization or parent evidence boundary drifted")
    environment_evidence = environment_audit(PATHS["environment"], environment)
    methods_evidence = method_audit(PATHS["methods"])
    result_root = REPRO / "h2oplus_v20_full_control_results"
    existing_results = list(result_root.rglob("sumo_seed*.json")) if result_root.exists() else []
    if existing_results:
        raise ValueError("v20 design audit must precede all v20 scientific outcomes")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v20 design audit: {output}")
    result = {
        "schema_version": "transit-h2oplus-v20-full-control-design-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed-before-any-v20-development-outcome",
        "study_stage": "development-design",
        "matrix_type": "full-development-control-matrix",
        "formal_held_out_outcomes_read": False,
        "v20_development_evaluation_outcomes_read": False,
        "local_sumo_executed": False,
        "local_training_executed": False,
        "prior_v14_through_v19_development_outcomes_used_for_design": True,
        "new_training_runs": 0,
        "frozen_inputs": bindings,
        "methods": methods_evidence,
        "environment": environment_evidence,
        "episode_accounting": {
            "physical_methods": 45,
            "environment_cells": 9,
            "expected_remote_sumo_episodes": 405,
            "matrix_manifests": 3,
            "jobs_per_manifest": 135,
        },
        "parent_v19_integrity_and_stability_explicitly_passed": True,
        "material_data_environment_validity_factors_remain": True,
        "scientific_outcomes_available": False,
        "formal_confirmation_eligible": False,
        "formal_confirmation_authorized_or_submitted": False,
        "supports_formal_held_out_effect_claim": False,
        "passed": True,
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_bytes(encoded)
    os.replace(partial, output)
    print(json.dumps({"path": str(output), "sha256": hashlib.sha256(encoded).hexdigest(), "passed": True}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
