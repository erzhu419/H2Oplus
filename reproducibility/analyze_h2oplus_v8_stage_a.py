#!/usr/bin/env python3
"""Apply the preregistered v8 Stage A gate to complete development results."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bus_h2o.sumo_provenance import sha256
from reproducibility.analyze_evaluation_matrix_v4 import (
    PRIMARY_METRIC,
    build_cubes,
    load_rows,
)


REGISTRATION_SCHEMA = "transit-h2oplus-v8-development-pilot-registration-v1"
FREEZE_SCHEMA = "transit-h2oplus-v8-stage-a-freeze-v1"
REPORT_SCHEMA = "transit-h2oplus-v8-stage-a-development-gate-v1"
REFERENCE = "h2oplus"
CANDIDATE = "h2oplus-lambda0p5"
EXPECTED_SEEDS = list(range(5101, 5106))
EXPECTED_BLOCKS = [f"v8-development-{index:02d}" for index in range(1, 6)]
EXPECTED_SCALES = [0.85, 1.0, 1.15]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, action="append", required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sign_sums(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            float(np.dot(values, np.asarray(signs, dtype=np.float64)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(values))
        ],
        dtype=np.float64,
    )


def exact_one_sided_sign_flip_p(differences: np.ndarray) -> float:
    """Return P(mean(sign * d) >= observed mean(d)) under all sign flips."""

    values = np.asarray(differences, dtype=np.float64).reshape(-1)
    if not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("paired differences must be nonempty and finite")
    split = len(values) // 2
    left = _sign_sums(values[:split])
    right = np.sort(_sign_sums(values[split:]))
    observed = float(values.sum())
    extreme = 0
    tolerance = 1e-12 * max(1.0, abs(observed))
    for value in left:
        threshold = observed - value - tolerance
        extreme += len(right) - int(np.searchsorted(right, threshold, side="left"))
    return extreme / float(2 ** len(values))


def _load_bound(path: Path, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"schema mismatch for {resolved}")
    return resolved, payload


def _registered_block_by_sumo_seed(registration: dict) -> dict[int, str]:
    blocks = registration.get("development_environment", {}).get("blocks", [])
    mapping = {
        int(block["sumo_seed"]): str(block["name"])
        for block in blocks
        if isinstance(block, dict)
        and not isinstance(block.get("sumo_seed"), bool)
        and block.get("sumo_seed") is not None
        and block.get("name") is not None
    }
    if (
        len(mapping) != len(EXPECTED_BLOCKS)
        or sorted(mapping.values()) != EXPECTED_BLOCKS
        or sorted(mapping) != list(range(4101, 4106))
    ):
        raise ValueError("Stage A registered block design drifted")
    return mapping


def load_complete_rows(
    matrix_roots: list[Path], registration: dict
) -> tuple[list[dict], list[dict]]:
    if len(matrix_roots) != 5:
        raise ValueError("Stage A requires exactly five merged matrix roots")
    rows = []
    evidence = []
    seen_blocks = set()
    block_by_sumo_seed = _registered_block_by_sumo_seed(registration)
    for root in matrix_roots:
        block_rows, block_evidence = load_rows(root)
        sumo_seeds = {int(row["sumo_seed"]) for row in block_rows}
        raw_blocks = {str(row["environment_block"]) for row in block_rows}
        if len(block_rows) != 30 or len(sumo_seeds) != 1 or len(raw_blocks) != 1:
            raise ValueError(f"Stage A block {root} must contain exactly 30 jobs")
        sumo_seed = next(iter(sumo_seeds))
        block = block_by_sumo_seed.get(sumo_seed)
        if block is None or root.expanduser().resolve().name != block:
            raise ValueError(f"Stage A block path or SUMO seed drifted: {root}")
        raw_block = next(iter(raw_blocks))
        if raw_block not in {block, f"sumo-seed-{sumo_seed}"}:
            raise ValueError(f"Stage A environment-block identity drifted: {raw_block}")
        if block in seen_blocks:
            raise ValueError(f"duplicate Stage A environment block {block}")
        seen_blocks.add(block)
        for row in block_rows:
            row["environment_block"] = block
        rows.extend(block_rows)
        evidence.append(block_evidence)
    if sorted(seen_blocks) != EXPECTED_BLOCKS or len(rows) != 150:
        raise ValueError("Stage A block coverage drifted")
    identities = {
        (
            item["mdp_version"],
            item["dataset_sha256"],
            json.dumps(item["contract"], sort_keys=True),
            json.dumps(item["source_sha256"], sort_keys=True),
        )
        for item in evidence
    }
    if len(identities) != 1 or next(iter(identities))[0] != "v7":
        raise ValueError("Stage A matrices do not share one v7 evaluation identity")
    return rows, evidence


def analyze_gate(rows: list[dict], registration: dict, freeze: dict) -> dict:
    stage = registration.get("stages", [None])[0]
    if (
        registration.get("status") != "registered-not-submitted"
        or registration.get("execution_authorized") is not False
        or registration.get("scientific_scope", {}).get(
            "supports_current_held_out_model_effect_claim"
        )
        is not False
        or not isinstance(stage, dict)
        or stage.get("name") != "A-existing-policy-development-screen"
        or stage.get("paired_unit")
        != "training seed x development environment block"
        or freeze.get("status") != "frozen-before-stage-A-outcomes"
        or freeze.get("formal_held_out_outcomes_read") is not False
        or freeze.get("training") is not False
    ):
        raise ValueError("Stage A registration or freeze scope drifted")
    if stage.get("advance_gate") != freeze.get("advance_gate"):
        raise ValueError("registration and freeze disagree on the Stage A gate")

    cubes, cells, metadata = build_cubes(rows)
    if set(cubes) != {REFERENCE, CANDIDATE}:
        raise ValueError(f"unexpected Stage A algorithms: {sorted(cubes)}")
    for algorithm in (REFERENCE, CANDIDATE):
        if cubes[algorithm]["training_seeds"] != EXPECTED_SEEDS:
            raise ValueError(f"training-seed coverage drifted for {algorithm}")
    blocks = sorted({str(item["environment_block"]) for item in metadata.values()})
    scales = sorted({float(item["demand_scale"]) for item in metadata.values()})
    if blocks != EXPECTED_BLOCKS or scales != EXPECTED_SCALES or len(cells) != 15:
        raise ValueError("Stage A development-cell coverage drifted")

    indices_by_block = {
        block: np.asarray(
            [
                index
                for index, cell in enumerate(cells)
                if metadata[cell]["environment_block"] == block
            ],
            dtype=np.int64,
        )
        for block in blocks
    }
    if any(len(indices) != 3 for indices in indices_by_block.values()):
        raise ValueError("each Stage A block must contain all three demand scales")

    reference_primary = cubes[REFERENCE]["metrics"][PRIMARY_METRIC]
    candidate_primary = cubes[CANDIDATE]["metrics"][PRIMARY_METRIC]
    reference_completion = cubes[REFERENCE]["metrics"]["passenger_completion_rate"]
    candidate_completion = cubes[CANDIDATE]["metrics"]["passenger_completion_rate"]
    if reference_primary.shape != (5, 15) or candidate_primary.shape != (5, 15):
        raise ValueError("Stage A primary metric cube must be 5 x 15")

    paired_units = []
    paired_differences = []
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        for block in blocks:
            indices = indices_by_block[block]
            reference_mean = float(reference_primary[seed_index, indices].mean())
            candidate_mean = float(candidate_primary[seed_index, indices].mean())
            difference = reference_mean - candidate_mean
            paired_differences.append(difference)
            paired_units.append(
                {
                    "training_seed": seed,
                    "environment_block": block,
                    "reference_mean": reference_mean,
                    "candidate_mean": candidate_mean,
                    "improvement_reference_minus_candidate": difference,
                }
            )
    paired_differences_array = np.asarray(paired_differences, dtype=np.float64)
    one_sided_p = exact_one_sided_sign_flip_p(paired_differences_array)

    reference_mean = float(reference_primary.mean())
    candidate_mean = float(candidate_primary.mean())
    relative_improvement = (reference_mean - candidate_mean) / reference_mean
    seed_means = []
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        baseline = float(reference_primary[seed_index].mean())
        candidate = float(candidate_primary[seed_index].mean())
        seed_means.append(
            {
                "training_seed": seed,
                "reference_mean": baseline,
                "candidate_mean": candidate,
                "candidate_lower": candidate < baseline,
            }
        )
    lower_seed_count = sum(row["candidate_lower"] for row in seed_means)

    scale_rows = []
    for scale in scales:
        indices = np.asarray(
            [
                index
                for index, cell in enumerate(cells)
                if math.isclose(metadata[cell]["demand_scale"], scale)
            ],
            dtype=np.int64,
        )
        baseline = float(reference_primary[:, indices].mean())
        candidate = float(candidate_primary[:, indices].mean())
        degradation = (candidate - baseline) / baseline
        scale_rows.append(
            {
                "demand_scale": scale,
                "reference_mean": baseline,
                "candidate_mean": candidate,
                "relative_degradation": degradation,
            }
        )

    reference_completion_mean = float(reference_completion.mean())
    candidate_completion_mean = float(candidate_completion.mean())
    completion_delta = candidate_completion_mean - reference_completion_mean
    thresholds = freeze["advance_gate"]
    checks = {
        "mean_relative_improvement": {
            "observed": relative_improvement,
            "required_at_least": float(thresholds["mean_relative_improvement_at_least"]),
            "passed": relative_improvement
            >= float(thresholds["mean_relative_improvement_at_least"]),
        },
        "training_seeds_with_lower_paired_mean": {
            "observed": lower_seed_count,
            "required_at_least": int(
                thresholds["training_seeds_with_lower_paired_mean_at_least"]
            ),
            "passed": lower_seed_count
            >= int(thresholds["training_seeds_with_lower_paired_mean_at_least"]),
        },
        "one_sided_paired_permutation": {
            "observed_p": one_sided_p,
            "required_less_than": float(
                thresholds["one_sided_paired_permutation_p_less_than"]
            ),
            "passed": one_sided_p
            < float(thresholds["one_sided_paired_permutation_p_less_than"]),
            "test": "exact sign flip over 25 preregistered training-seed x environment-block paired means",
        },
        "completion_rate_noninferiority": {
            "candidate_minus_reference": completion_delta,
            "margin": float(thresholds["completion_rate_noninferiority_margin"]),
            "passed": completion_delta
            >= -float(thresholds["completion_rate_noninferiority_margin"]),
        },
        "no_demand_scale_relative_degradation": {
            "maximum_observed": max(row["relative_degradation"] for row in scale_rows),
            "maximum_allowed": float(
                thresholds["no_demand_scale_mean_relative_degradation_greater_than"]
            ),
            "passed": all(
                row["relative_degradation"]
                <= float(
                    thresholds[
                        "no_demand_scale_mean_relative_degradation_greater_than"
                    ]
                )
                for row in scale_rows
            ),
        },
    }
    passed = all(check["passed"] for check in checks.values())
    return {
        "schema_version": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "A-existing-policy-development-screen",
        "study_stage": "development",
        "training": False,
        "formal_held_out_outcomes_read": False,
        "supports_current_held_out_model_effect_claim": False,
        "reference": REFERENCE,
        "candidate": CANDIDATE,
        "primary_endpoint": PRIMARY_METRIC,
        "primary_direction": "lower-is-better",
        "primary_means": {
            "reference": reference_mean,
            "candidate": candidate_mean,
            "relative_improvement": relative_improvement,
        },
        "completion_rate_means": {
            "reference": reference_completion_mean,
            "candidate": candidate_completion_mean,
            "candidate_minus_reference": completion_delta,
        },
        "paired_units": paired_units,
        "training_seed_means": seed_means,
        "demand_scale_means": scale_rows,
        "checks": checks,
        "stage_a_gate_passed": passed,
        "stage_b_gate_eligible": passed,
        "stage_b_authorized_or_submitted": False,
    }


def main() -> int:
    args = parse_args()
    registration_path, registration = _load_bound(
        args.registration, REGISTRATION_SCHEMA
    )
    freeze_path, freeze = _load_bound(args.freeze, FREEZE_SCHEMA)
    rows, matrices = load_complete_rows(args.matrix_root, registration)
    report = analyze_gate(rows, registration, freeze)
    report["evidence"] = {
        "registration": {
            "path": str(registration_path),
            "sha256": sha256(registration_path),
        },
        "freeze": {"path": str(freeze_path), "sha256": sha256(freeze_path)},
        "matrix_manifests": [
            {
                "path": item["matrix_manifest"],
                "sha256": item["matrix_manifest_sha256"],
                "evaluation_outputs_aggregate_sha256": item[
                    "evaluation_outputs_aggregate_sha256"
                ],
            }
            for item in matrices
        ],
        "analysis_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
