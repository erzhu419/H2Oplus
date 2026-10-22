#!/usr/bin/env python3
"""Compare sealed H2O+ development results with SAC baselines for v8 Stage A2."""

from __future__ import annotations

import argparse
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
from reproducibility.analyze_h2oplus_v8_stage_a import (
    exact_one_sided_sign_flip_p,
)


REGISTRATION_SCHEMA = "transit-h2oplus-v8-stage-a2-sac-baseline-registration-v1"
FREEZE_SCHEMA = "transit-h2oplus-v8-stage-a2-sac-baseline-freeze-v1"
REPORT_SCHEMA = "transit-h2oplus-v8-stage-a2-sac-baseline-analysis-v1"
REFERENCE = "h2oplus"
COMPARATORS = ("sac-mixed", "sac-online")
EXPECTED_SEEDS = list(range(5101, 5106))
EXPECTED_BLOCKS = [f"v8-development-{index:02d}" for index in range(1, 6)]
EXPECTED_SCALES = [0.85, 1.0, 1.15]
WORKSPACE_PATH_MARKER = "/workspace/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sac-matrix-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--reference-matrix-root", type=Path, action="append", required=True
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_bound(path: Path, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"schema mismatch for {resolved}")
    return resolved, payload


def _registered_blocks(registration: dict) -> dict[int, str]:
    blocks = registration.get("development_environment", {}).get("blocks", [])
    mapping = {
        int(block["sumo_seed"]): str(block["name"])
        for block in blocks
        if isinstance(block, dict)
    }
    if (
        len(mapping) != 5
        or sorted(mapping) != list(range(4101, 4106))
        or sorted(mapping.values()) != EXPECTED_BLOCKS
    ):
        raise ValueError("Stage A2 registered block design drifted")
    return mapping


def _portable_source_identity(source_sha256: dict[str, str]) -> dict[str, str]:
    """Remove only the deployment root while preserving path/content identity."""
    if not isinstance(source_sha256, dict) or not source_sha256:
        raise ValueError("evaluation source identity is empty or malformed")
    portable: dict[str, str] = {}
    for raw_path, digest in source_sha256.items():
        if (
            not isinstance(raw_path, str)
            or WORKSPACE_PATH_MARKER not in raw_path
            or not isinstance(digest, str)
            or len(digest) != 64
        ):
            raise ValueError("evaluation source identity is not deployment-portable")
        relative_path = raw_path.split(WORKSPACE_PATH_MARKER, 1)[1]
        if (
            not relative_path
            or relative_path.startswith("/")
            or ".." in Path(relative_path).parts
            or relative_path in portable
        ):
            raise ValueError("evaluation source identity has an invalid relative path")
        portable[relative_path] = digest
    return portable


def _evaluation_identity(item: dict) -> tuple[str, str, str, str]:
    return (
        str(item["mdp_version"]),
        str(item["dataset_sha256"]),
        json.dumps(item["contract"], sort_keys=True),
        json.dumps(_portable_source_identity(item["source_sha256"]), sort_keys=True),
    )


def _load_block_roots(
    roots: list[Path],
    registration: dict,
    *,
    expected_algorithms: set[str],
    rows_per_block: int,
    source_label: str,
) -> tuple[list[dict], list[dict]]:
    if len(roots) != 5:
        raise ValueError(f"{source_label} requires exactly five matrix roots")
    block_by_seed = _registered_blocks(registration)
    rows: list[dict] = []
    evidence: list[dict] = []
    seen_blocks: set[str] = set()
    for root in roots:
        block_rows, block_evidence = load_rows(root)
        sumo_seeds = {int(row["sumo_seed"]) for row in block_rows}
        if len(sumo_seeds) != 1:
            raise ValueError(f"{source_label} block has multiple SUMO seeds: {root}")
        sumo_seed = next(iter(sumo_seeds))
        block = block_by_seed.get(sumo_seed)
        resolved = root.expanduser().resolve()
        if block is None or resolved.name != block or block in seen_blocks:
            raise ValueError(f"{source_label} block identity drifted: {root}")
        selected = [row for row in block_rows if row.get("algorithm") in expected_algorithms]
        if len(selected) != rows_per_block:
            raise ValueError(
                f"{source_label} block {block} has {len(selected)} selected rows, "
                f"expected {rows_per_block}"
            )
        if {str(row["algorithm"]) for row in selected} != expected_algorithms:
            raise ValueError(f"{source_label} algorithm coverage drifted in {block}")
        for row in selected:
            row["environment_block"] = block
        seen_blocks.add(block)
        rows.extend(selected)
        evidence.append(block_evidence)
    if sorted(seen_blocks) != EXPECTED_BLOCKS:
        raise ValueError(f"{source_label} development-block coverage drifted")
    identities = {_evaluation_identity(item) for item in evidence}
    if len(identities) != 1 or next(iter(identities))[0] != "v7":
        raise ValueError(f"{source_label} does not share one v7 evaluation identity")
    return rows, evidence


def load_complete_rows(
    sac_roots: list[Path], reference_roots: list[Path], registration: dict
) -> tuple[list[dict], dict]:
    sac_rows, sac_evidence = _load_block_roots(
        sac_roots,
        registration,
        expected_algorithms=set(COMPARATORS),
        rows_per_block=30,
        source_label="Stage A2 SAC",
    )
    reference_rows, reference_evidence = _load_block_roots(
        reference_roots,
        registration,
        expected_algorithms={REFERENCE},
        rows_per_block=15,
        source_label="sealed Stage A H2O+ reference",
    )
    if len(sac_rows) != 150 or len(reference_rows) != 75:
        raise ValueError("Stage A2 requires 150 new SAC and 75 reused H2O+ rows")
    identities = {
        _evaluation_identity(item) for item in [*sac_evidence, *reference_evidence]
    }
    if len(identities) != 1:
        raise ValueError("new SAC and reused H2O+ results do not share one identity")
    return [*reference_rows, *sac_rows], {
        "sac_matrices": sac_evidence,
        "reference_matrices": reference_evidence,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, dict]:
    """Return Holm adjusted p-values and step-down decisions at alpha 0.05."""

    if set(p_values) != set(COMPARATORS):
        raise ValueError("Holm family must contain exactly the two SAC comparisons")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid p-value for {name}: {value}")
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return {
        name: {
            "raw_p": float(p_values[name]),
            "holm_adjusted_p": float(adjusted[name]),
            "familywise_alpha": 0.05,
            "passed": bool(adjusted[name] < 0.05),
        }
        for name in COMPARATORS
    }


def _comparison(
    cubes: dict,
    cells: list,
    metadata: dict,
    comparator: str,
    thresholds: dict,
) -> dict:
    reference_primary = cubes[REFERENCE]["metrics"][PRIMARY_METRIC]
    comparator_primary = cubes[comparator]["metrics"][PRIMARY_METRIC]
    reference_completion = cubes[REFERENCE]["metrics"]["passenger_completion_rate"]
    comparator_completion = cubes[comparator]["metrics"]["passenger_completion_rate"]
    if reference_primary.shape != (5, 15) or comparator_primary.shape != (5, 15):
        raise ValueError(f"primary metric cube drifted for {comparator}")

    blocks = EXPECTED_BLOCKS
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
        raise ValueError("each development block must contain three demand scales")

    paired_units = []
    differences = []
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        for block in blocks:
            indices = indices_by_block[block]
            reference_mean = float(reference_primary[seed_index, indices].mean())
            comparator_mean = float(comparator_primary[seed_index, indices].mean())
            difference = comparator_mean - reference_mean
            differences.append(difference)
            paired_units.append(
                {
                    "training_seed": seed,
                    "environment_block": block,
                    "h2oplus_mean": reference_mean,
                    "comparator_mean": comparator_mean,
                    "improvement_comparator_minus_h2oplus": difference,
                }
            )

    reference_mean = float(reference_primary.mean())
    comparator_mean = float(comparator_primary.mean())
    relative_improvement = (comparator_mean - reference_mean) / comparator_mean
    raw_p = exact_one_sided_sign_flip_p(np.asarray(differences, dtype=np.float64))

    seed_means = []
    for seed_index, seed in enumerate(EXPECTED_SEEDS):
        reference_seed_mean = float(reference_primary[seed_index].mean())
        comparator_seed_mean = float(comparator_primary[seed_index].mean())
        seed_means.append(
            {
                "training_seed": seed,
                "h2oplus_mean": reference_seed_mean,
                "comparator_mean": comparator_seed_mean,
                "h2oplus_lower": reference_seed_mean < comparator_seed_mean,
            }
        )

    scale_means = []
    for scale in EXPECTED_SCALES:
        indices = np.asarray(
            [
                index
                for index, cell in enumerate(cells)
                if math.isclose(float(metadata[cell]["demand_scale"]), scale)
            ],
            dtype=np.int64,
        )
        reference_scale = float(reference_primary[:, indices].mean())
        comparator_scale = float(comparator_primary[:, indices].mean())
        scale_means.append(
            {
                "demand_scale": scale,
                "h2oplus_mean": reference_scale,
                "comparator_mean": comparator_scale,
                "h2oplus_relative_degradation": (
                    reference_scale - comparator_scale
                )
                / comparator_scale,
            }
        )

    completion_delta = float(reference_completion.mean() - comparator_completion.mean())
    return {
        "comparator": comparator,
        "primary_means": {
            "h2oplus": reference_mean,
            "comparator": comparator_mean,
            "h2oplus_relative_improvement": relative_improvement,
        },
        "completion_rate_means": {
            "h2oplus": float(reference_completion.mean()),
            "comparator": float(comparator_completion.mean()),
            "h2oplus_minus_comparator": completion_delta,
        },
        "raw_one_sided_p": raw_p,
        "paired_units": paired_units,
        "training_seed_means": seed_means,
        "demand_scale_means": scale_means,
        "pre_holm_checks": {
            "directional_mean_superiority": {
                "observed": relative_improvement,
                "required_greater_than": 0.0,
                "passed": relative_improvement > 0.0,
            },
            "practical_mean_improvement": {
                "observed": relative_improvement,
                "required_at_least": float(
                    thresholds["mean_relative_improvement_at_least"]
                ),
                "passed": relative_improvement
                >= float(thresholds["mean_relative_improvement_at_least"]),
            },
            "training_seeds_with_lower_mean": {
                "observed": sum(item["h2oplus_lower"] for item in seed_means),
                "required_at_least": int(
                    thresholds["training_seeds_with_lower_paired_mean_at_least"]
                ),
                "passed": sum(item["h2oplus_lower"] for item in seed_means)
                >= int(thresholds["training_seeds_with_lower_paired_mean_at_least"]),
            },
            "completion_rate_noninferiority": {
                "h2oplus_minus_comparator": completion_delta,
                "margin": float(thresholds["completion_rate_noninferiority_margin"]),
                "passed": completion_delta
                >= -float(thresholds["completion_rate_noninferiority_margin"]),
            },
            "no_demand_scale_relative_degradation": {
                "maximum_observed": max(
                    item["h2oplus_relative_degradation"] for item in scale_means
                ),
                "maximum_allowed": float(
                    thresholds[
                        "no_demand_scale_mean_relative_degradation_greater_than"
                    ]
                ),
                "passed": all(
                    item["h2oplus_relative_degradation"]
                    <= float(
                        thresholds[
                            "no_demand_scale_mean_relative_degradation_greater_than"
                        ]
                    )
                    for item in scale_means
                ),
            },
        },
    }


def analyze(rows: list[dict], registration: dict, freeze: dict) -> dict:
    if (
        registration.get("status") != "registered-not-submitted"
        or registration.get("execution_authorized") is not False
        or registration.get("formal_held_out_outcomes_read") is not False
        or freeze.get("status") != "frozen-before-stage-A2-SAC-outcomes"
        or freeze.get("training") is not False
        or freeze.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("Stage A2 registration or freeze scope drifted")
    thresholds = registration.get("comparison_gate")
    if not isinstance(thresholds, dict) or thresholds != freeze.get("comparison_gate"):
        raise ValueError("Stage A2 comparison gate drifted")

    cubes, cells, metadata = build_cubes(rows)
    if set(cubes) != {REFERENCE, *COMPARATORS}:
        raise ValueError(f"unexpected Stage A2 algorithms: {sorted(cubes)}")
    for algorithm in (REFERENCE, *COMPARATORS):
        if cubes[algorithm]["training_seeds"] != EXPECTED_SEEDS:
            raise ValueError(f"training-seed coverage drifted for {algorithm}")
    blocks = sorted({str(item["environment_block"]) for item in metadata.values()})
    scales = sorted({float(item["demand_scale"]) for item in metadata.values()})
    if blocks != EXPECTED_BLOCKS or scales != EXPECTED_SCALES or len(cells) != 15:
        raise ValueError("Stage A2 development-cell coverage drifted")

    comparisons = {
        comparator: _comparison(cubes, cells, metadata, comparator, thresholds)
        for comparator in COMPARATORS
    }
    holm = holm_adjust(
        {
            comparator: comparisons[comparator]["raw_one_sided_p"]
            for comparator in COMPARATORS
        }
    )
    for comparator in COMPARATORS:
        checks = dict(comparisons[comparator].pop("pre_holm_checks"))
        checks["holm_corrected_one_sided_paired_permutation"] = holm[comparator]
        comparisons[comparator]["checks"] = checks
        comparisons[comparator]["h2oplus_superiority_confirmed"] = all(
            item["passed"] for item in checks.values()
        )

    return {
        "schema_version": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "A2-H2Oplus-vs-SAC-development-baseline-confirmation",
        "study_stage": "development",
        "training": False,
        "new_sumo_episode_count": 150,
        "reused_h2oplus_episode_count": 75,
        "formal_held_out_outcomes_read": False,
        "supports_current_formal_held_out_model_effect_claim": False,
        "primary_endpoint": PRIMARY_METRIC,
        "primary_direction": "lower-is-better",
        "paired_unit": "training seed x development environment block",
        "multiple_testing": {
            "procedure": "Holm step-down",
            "familywise_alpha": 0.05,
            "family": [f"h2oplus-vs-{name}" for name in COMPARATORS],
        },
        "comparisons": comparisons,
        "all_SAC_comparisons_confirmed": all(
            comparisons[name]["h2oplus_superiority_confirmed"]
            for name in COMPARATORS
        ),
        "stage_b_authorized_or_submitted": False,
    }


def main() -> int:
    args = parse_args()
    registration_path, registration = _load_bound(
        args.registration, REGISTRATION_SCHEMA
    )
    freeze_path, freeze = _load_bound(args.freeze, FREEZE_SCHEMA)
    rows, matrix_evidence = load_complete_rows(
        args.sac_matrix_root, args.reference_matrix_root, registration
    )
    report = analyze(rows, registration, freeze)
    report["evidence"] = {
        "registration": {
            "path": str(registration_path),
            "sha256": sha256(registration_path),
        },
        "freeze": {"path": str(freeze_path), "sha256": sha256(freeze_path)},
        "sac_matrix_manifests": [
            {
                "path": item["matrix_manifest"],
                "sha256": item["matrix_manifest_sha256"],
                "evaluation_outputs_aggregate_sha256": item[
                    "evaluation_outputs_aggregate_sha256"
                ],
            }
            for item in matrix_evidence["sac_matrices"]
        ],
        "reference_matrix_manifests": [
            {
                "path": item["matrix_manifest"],
                "sha256": item["matrix_manifest_sha256"],
                "evaluation_outputs_aggregate_sha256": item[
                    "evaluation_outputs_aggregate_sha256"
                ],
            }
            for item in matrix_evidence["reference_matrices"]
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
