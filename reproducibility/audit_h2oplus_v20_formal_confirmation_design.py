#!/usr/bin/env python3
"""Audit the frozen v20 formal-confirmation preparation without running SUMO."""

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

from bus_h2o.sumo_provenance import load_validated_scenario, sha256  # noqa: E402
from reproducibility.prepare_h2oplus_v8_development_stage_a import (  # noqa: E402
    _runtime_key,
)
from reproducibility.run_evaluation_matrix_v20_formal_confirmation import (  # noqa: E402
    ALGORITHMS,
    CANDIDATE,
    DIRECT_PARENT,
    METHODS_SCHEMA,
    RAW_PARENT,
    SEEDS,
    load_formal_methods,
)


EXPECTED = {
    "authorization": "7a7b79f029a5437a80a31637c1bc057571849623b9b04a92a67bcf66586882c9",
    "rationale": "72e6d2961846e458721f121ae75eded862bed0556c929dc38015a3b2d9103a4d",
    "registration": "c33111cd72b741a801804cd065f42629094d45c0fe03a5b88f9d95dba62149ce",
    "protocol": "10aacced63f4c8a1b7f6f55024f46dfdebbcae6737c6bbafb79d227b2ad44664",
    "methods": "b4bcad09e13cb50c94d5541f2a52340ce99f15b281b5ddef57be38e792d052cd",
    "environment": "50b831923b76a81c773cf7dc19dcc2e10c729ea8fc7b3be6982fc01b7c3cc141",
    "v20_analysis": "f7de8ca7ea82098a89383f1fe0e9e234f8c1acab1c030d66494100744cd9455b",
    "v20_audit": "2e516ab3ea0cff10dc88607d6fb8adf3e3888a822e32a14f84a86953a466f583",
    "v20_closeout": "dbd1da5442219f404f90da6fbc415b0755f7dddd41d28b9b581039d359882132",
    "v20_verification": "08b25f8b229a9fefbfec198f8198aff5db4f9f2680f16d9bf5521985cdfe7f7e",
    "data_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    "minimal_sync_policy": "20b313b7b8cc13a0a8df7c2a137a4b0713c621a33ef8e2e314fb69d03de1a531",
}
PATHS = {
    "authorization": REPRO
    / "h2oplus_v20_formal_confirmation_preparation_authorization_v1.json",
    "rationale": REPRO / "h2oplus_v20_formal_confirmation_rationale_v1.json",
    "registration": REPRO / "h2oplus_v20_formal_confirmation_registration_v1.json",
    "protocol": REPRO / "h2oplus_v20_formal_confirmation_protocol_v1.json",
    "methods": REPRO
    / "h2oplus_v20_formal_confirmation_evaluation_methods_v1.json",
    "environment": REPRO / "h2oplus_v20_formal_confirmation_environment_v1.json",
    "v20_analysis": REPRO
    / "h2oplus_v20_full_control_results/full_control_analysis_v1.json",
    "v20_audit": REPRO
    / "h2oplus_v20_full_control_results/evaluation_audit_v1.json",
    "v20_closeout": REPRO
    / "h2oplus_v20_full_control_results/evaluation_closeout_v1.json",
    "v20_verification": REPRO
    / "h2oplus_v20_full_control_results/evaluation_closeout_verification_v1.json",
    "data_audit": REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
    "data_verification": REPRO
    / "h2oplus_v17_data_environment_convergence_verification_v1.json",
    "minimal_sync_policy": REPRO / "h2oplus_minimal_remote_sync_policy_v1.json",
}
DEFAULT_OUTPUT = REPRO / "h2oplus_v20_formal_confirmation_design_audit_v1.json"
FORMAL_BLOCKS = {
    f"v20-formal-confirmation-{index:02d}" for index in range(1, 11)
}
SUMO_SEEDS = set(range(8101, 8111))
DEMAND_SEEDS = set(range(8201, 8211))
DEMAND_SCALES = {0.75, 1.0, 1.25}


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
        resolved = path.resolve(strict=True)
        observed = sha256(resolved)
        if observed != EXPECTED[name]:
            raise ValueError(
                f"frozen formal-confirmation design input drifted: {name}: {observed}"
            )
        payloads[name] = json.loads(resolved.read_text())
        bindings[name] = descriptor(resolved)
    return payloads, bindings


def environment_audit(environment_path: Path, payload: dict) -> dict:
    names, sumo_seeds, demand_seeds = set(), set(), set()
    cells, manifests = set(), []
    descriptors = {
        (row["block"], float(row["demand_scale"])): row
        for row in payload.get("scenario_descriptors", [])
    }
    for block in payload.get("blocks", []):
        name = block.get("name")
        sumo_seed = block.get("sumo_seed")
        demand_seed = block.get("demand_realization_seed")
        paths = block.get("scenario_manifests", [])
        if (
            name in names
            or not isinstance(name, str)
            or not isinstance(sumo_seed, int)
            or isinstance(sumo_seed, bool)
            or sumo_seed in sumo_seeds
            or not isinstance(demand_seed, int)
            or isinstance(demand_seed, bool)
            or demand_seed in demand_seeds
            or len(paths) != 3
        ):
            raise ValueError(f"malformed formal-confirmation block: {block!r}")
        scales = set()
        for raw in paths:
            path = (environment_path.parent / raw).resolve(strict=True)
            _, evidence = load_validated_scenario(path)
            manifest = json.loads(path.read_text())
            scale = float(manifest.get("demand_scale"))
            registered = descriptors.get((name, scale), {})
            if (
                manifest.get("schema_version") != "transit-passenger-demand-v1"
                or manifest.get("selection_seed") != demand_seed
                or float(manifest.get("depart_jitter_seconds")) != 120.0
                or scale not in DEMAND_SCALES
                or sha256(path) != registered.get("manifest_sha256")
                or _runtime_key(evidence) != registered.get("runtime_content_sha256")
            ):
                raise ValueError(f"formal-confirmation scenario drifted: {path}")
            cell = (sumo_seed, demand_seed, scale)
            if cell in cells:
                raise ValueError(f"duplicate formal-confirmation cell: {cell}")
            cells.add(cell)
            scales.add(scale)
            manifests.append(descriptor(path))
        if scales != DEMAND_SCALES:
            raise ValueError(f"formal-confirmation demand grid drifted: {name}")
        names.add(name)
        sumo_seeds.add(sumo_seed)
        demand_seeds.add(demand_seed)
    overlaps = {
        key: value
        for key, value in payload.get("disjointness", {}).items()
        if key.endswith("_overlap") and value
    }
    if (
        names != FORMAL_BLOCKS
        or sumo_seeds != SUMO_SEEDS
        or demand_seeds != DEMAND_SEEDS
        or len(cells) != 30
        or len(descriptors) != 30
        or overlaps
        or payload.get("disjointness", {}).get("verified_disjoint") is not True
        or payload.get("prior_used_formal_v7_environment_reused") is not False
    ):
        raise ValueError("formal-confirmation environment disjointness drifted")
    return {
        "block_count": len(names),
        "environment_cell_count": len(cells),
        "sumo_seeds": sorted(sumo_seeds),
        "demand_realization_seeds": sorted(demand_seeds),
        "demand_scales": sorted(DEMAND_SCALES),
        "scenario_manifests": sorted(manifests, key=lambda item: item["path"]),
        "verified_disjoint": True,
        "prior_used_formal_v7_environment_reused": False,
        "nonempty_overlap_fields": overlaps,
    }


def method_audit(path: Path) -> dict:
    rows = load_formal_methods(path, METHODS_SCHEMA)
    pairs = {(row["algorithm"], row["training_seed"]) for row in rows}
    expected = {(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS}
    actors = {row["actor_tensor_sha256"] for row in rows}
    if (
        len(rows) != 45
        or pairs != expected
        or sum(row["algorithm"] == CANDIDATE for row in rows) != 5
        or sum(row["algorithm"] == DIRECT_PARENT for row in rows) != 5
        or sum(row["algorithm"] == RAW_PARENT for row in rows) != 5
        or any(row.get("online_events") != 25_000 for row in rows)
        or any(
            row.get("v20_role") != "predesignated-candidate"
            for row in rows
            if row["algorithm"] == CANDIDATE
        )
    ):
        raise ValueError("formal-confirmation physical method grid drifted")
    for seed in SEEDS:
        lineage = {
            row["actor_tensor_sha256"]
            for row in rows
            if row["training_seed"] == seed
            and row["algorithm"] in {CANDIDATE, DIRECT_PARENT, RAW_PARENT}
        }
        if len(lineage) != 1:
            raise ValueError(f"formal parent actor lineage drifted for seed {seed}")
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
    v20_analysis = payloads["v20_analysis"]
    v20_audit = payloads["v20_audit"]
    v20_closeout = payloads["v20_closeout"]
    v20_verification = payloads["v20_verification"]
    scope = authorization.get("authorization_scope", {})
    if (
        authorization.get("authorization_text") != "可以 那就继续"
        or authorization.get("formal_held_out_outcomes_read") is not False
        or scope.get("formal_confirmation_design_preparation_authorized") is not True
        or scope.get("formal_analysis_source_freeze_authorized") is not True
        or scope.get("formal_confirmation_evaluation_authorized") is not False
        or scope.get("formal_confirmation_submission_authorized") is not False
        or scope.get("formal_held_out_outcome_access_authorized") is not False
        or scope.get("remote_sumo_episode_cap") != 0
        or scope.get("unique_task_count_cap") != 0
        or registration.get("matrix_type") != "endpoint-only-formal-confirmation"
        or registration.get("formal_confirmation_evaluation_authorized_or_submitted")
        is not False
        or protocol.get("matrix_type") != "endpoint-only-formal-confirmation"
        or protocol.get("evaluation", {}).get("submission_authorized") is not False
        or protocol.get("analysis", {}).get("submission_authorized") is not False
        or protocol.get("analysis", {}).get("independent_analysis_unit")
        != "formal-environment-block"
        or protocol.get("sample_size_justification", {}).get("independent_blocks")
        != 10
        or v20_analysis.get("development_scientific_result", {}).get(
            "full_development_control_matrix_passed"
        )
        is not True
        or v20_audit.get("passed") is not True
        or v20_audit.get("artifact_integrity", {}).get("passed") is not True
        or v20_closeout.get("passed") is not True
        or v20_closeout.get("formal_confirmation_eligible") is not True
        or v20_closeout.get("formal_confirmation_authorized_or_submitted") is not False
        or v20_verification.get("passed") is not True
        or v20_verification.get("formal_confirmation_eligible") is not True
        or payloads["data_audit"].get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or payloads["data_verification"].get("passed") is not True
    ):
        raise ValueError("formal preparation authorization or parent evidence drifted")
    if protocol.get("dataset", {}).get("sha256") != (
        "6724dc9fd00f983d93fe5216bf559ccbdf4bbb70f1f31b7fc1b5146387b93899"
    ) or protocol.get("sumo_runtime_lock", {}).get("sha256") != (
        "ac60ae58e0e6db6bb1386ffdfe0d164b96f81e49e4983b8b062ceddb65da51a4"
    ):
        raise ValueError("formal dataset or runtime lock identity drifted")

    environment_evidence = environment_audit(PATHS["environment"], environment)
    methods_evidence = method_audit(PATHS["methods"])
    result_root = REPRO / "h2oplus_v20_formal_confirmation_results"
    existing_results = (
        list(result_root.rglob("sumo_seed*.json")) if result_root.exists() else []
    )
    if existing_results:
        raise ValueError("formal design audit must precede all formal outcomes")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal design audit: {output}")
    result = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-design-audit-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed-before-any-v20-formal-confirmation-outcome",
        "study_stage": "formal-confirmation-preparation",
        "matrix_type": "endpoint-only-formal-confirmation",
        "formal_held_out_outcomes_read": False,
        "formal_confirmation_outcomes_read": False,
        "formal_confirmation_evaluation_authorized": False,
        "formal_confirmation_submission_authorized": False,
        "local_sumo_executed": False,
        "local_training_executed": False,
        "prior_v14_through_v20_development_outcomes_used_for_design": True,
        "new_training_runs": 0,
        "frozen_inputs": bindings,
        "methods": methods_evidence,
        "environment": environment_evidence,
        "episode_accounting": {
            "physical_methods": 45,
            "environment_cells": 30,
            "planned_remote_sumo_episodes": 1350,
            "matrix_manifests": 10,
            "jobs_per_manifest": 135,
            "authorized_remote_sumo_episodes": 0,
        },
        "confirmatory_analysis": {
            "independent_environment_blocks": 10,
            "paired_observations_averaged_per_block": 15,
            "minimum_directional_block_wins": 9,
            "one_sided_alpha": 0.025,
            "primary_test": "exact-paired-block-sign-test",
            "completion_noninferiority_margin": 0.003,
            "multiplicity": "intersection-union",
            "claim_limited_to_direct_behavior_parent": True,
        },
        "parent_v20_artifact_integrity_and_training_stability_passed": True,
        "material_data_environment_validity_factors_remain": True,
        "minimal_remote_sync_policy_bound": True,
        "scientific_outcomes_available": False,
        "supports_formal_held_out_effect_claim": False,
        "passed": True,
    }
    encoded = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_bytes(encoded)
    os.replace(partial, output)
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "passed": True,
                "formal_confirmation_submission_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
