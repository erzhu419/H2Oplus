#!/usr/bin/env python3
"""Analyze the frozen v19 completion-safety paired development screen."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256  # noqa: E402


EXPECTED_SHA256 = {
    "methods": "6d63b7dccf261859db7040d52b93482e17c8360d6db53b902de3f3f569729bb8",
    "protocol": "fa5f629588d35b1435e28bd39931f49d9851811ea3e308d62672f9eacf1f7184",
    "registration": "536741a185e2d9b10c99ed821292a51c64bf64e8cb35771075ab787565246fd4",
    "environment": "5cfb1318418b64e619d4be727c41a0ce0eccab8c096020482586d8286cee3a05",
    "v18_audit": "053feec4df5d1ffdaaea4e494815a7378fa011956a2bc2703f14a5d3dd1ac64c",
    "v18_closeout": "e2a0285e9d95bc999cbe371ac261afb1f4526ff7a6722ba2d9d114f67c4c401c",
    "v18_verification": "6a08fa047a4616589fba04cd39eec41e9450ef1ce1fd100933039cb7354fa35a",
    "design_audit": "2238ceb015a456b80c2be5d4d06ebd96d2293fdbc88cadcfb08f087d2dafe1e8",
    "design_verification": "3f8c0e03514b2fb6e2ed46249cfe8260cfdc3c9a2872316f6575de1504543a78",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
}
METHOD = re.compile(
    r"^(?P<algorithm>h2oplus-v19-completion-safety-"
    r"(?P<variant>baseline|quarter-load-fifty|quarter-load-sixty))"
    r"-seed(?P<seed>62\d{2})-events000025000$"
)
SEEDS = tuple(range(6201, 6206))
SCALES = (0.85, 1.0, 1.15)
BLOCKS = tuple(
    f"v19-completion-safety-development-{index:02d}" for index in range(1, 4)
)
ALGORITHMS = (
    "h2oplus-v19-completion-safety-baseline",
    "h2oplus-v19-completion-safety-quarter-load-fifty",
    "h2oplus-v19-completion-safety-quarter-load-sixty",
)
BASELINE = ALGORITHMS[0]
DIAGNOSTIC = ALGORITHMS[1]
CANDIDATE = ALGORITHMS[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--methods",
        type=Path,
        default=ROOT
        / "reproducibility/h2oplus_v19_completion_safety_evaluation_methods_v1.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "reproducibility/h2oplus_v19_completion_safety_protocol_v1.json",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT
        / "reproducibility/h2oplus_v19_completion_safety_registration_v1.json",
    )
    parser.add_argument(
        "--environment",
        type=Path,
        default=ROOT
        / "reproducibility/h2oplus_v19_completion_safety_environment_v1.json",
    )
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument(
        "--v18-audit",
        type=Path,
        default=ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_results/evaluation_audit_v3.json",
    )
    parser.add_argument(
        "--v18-closeout",
        type=Path,
        default=ROOT
        / "reproducibility/h2oplus_v18_empty_tail_reserve_results/evaluation_closeout_v3.json",
    )
    parser.add_argument(
        "--v18-verification",
        type=Path,
        default=ROOT
        / (
            "reproducibility/h2oplus_v18_empty_tail_reserve_results/"
            "evaluation_closeout_verification_v3.json"
        ),
    )
    parser.add_argument(
        "--design-audit",
        type=Path,
        default=ROOT
        / "reproducibility/h2oplus_v19_completion_safety_design_audit_v2.json",
    )
    parser.add_argument(
        "--design-verification",
        type=Path,
        default=ROOT
        / "reproducibility/h2oplus_v19_completion_safety_design_verification_v2.json",
    )
    parser.add_argument(
        "--data-environment-audit",
        type=Path,
        default=ROOT
        / "reproducibility/h2oplus_v17_data_environment_convergence_audit_v1.json",
    )
    parser.add_argument(
        "--data-environment-verification",
        type=Path,
        default=ROOT
        / "reproducibility/h2oplus_v17_data_environment_convergence_verification_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_frozen(path: Path, expected: str, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected:
        raise ValueError(f"frozen v19 identity drifted: {resolved}")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"frozen v19 schema drifted: {resolved}")
    return resolved, payload


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {label}")
    return result


def _nested(payload: dict, *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def _resolve_output(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_file():
        return path.resolve()
    local = root / path.name
    if local.is_file():
        return local.resolve()
    matches = list(root.rglob(path.name))
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"cannot resolve v19 output {raw!r} under {root}")


def _ledger_and_service_pass(payload: dict) -> bool:
    semantics = payload.get("bus_service_semantics_audit", {})
    violations = semantics.get("violations")
    service = payload.get("service", {})
    ledger = payload.get("ledger_audit", {})
    counts = ledger.get("count_source_audit", {})
    passengers = payload.get("passengers", {})
    welfare = payload.get("passenger_welfare", {})
    coverage = welfare.get("chronological_coverage_errors", {})
    return (
        semantics.get("passed") is True
        and semantics.get("require_all_departed_trips_completed") is True
        and isinstance(violations, dict)
        and all(value in ([], {}) for value in violations.values())
        and service.get("active_trips_at_horizon") == 0
        and service.get("completed_trips") == service.get("departed_trips")
        and service.get("completion_rate") == 1.0
        and ledger.get("legacy_sampled_trip_histories_used_as_ground_truth")
        is False
        and counts.get("max_in_vehicle_count_error") == 0
        and counts.get("max_waiting_count_error") == 0
        and counts.get("subscription_fallback_queries") == 0
        and passengers.get("legacy_sampled_trip_histories_used_for_welfare")
        is False
        and passengers.get("active_state_count_balance_error") == 0
        and welfare.get("primary") is True
        and isinstance(coverage, dict)
        and bool(coverage)
        and all(abs(float(value)) <= 1e-9 for value in coverage.values())
        and abs(
            float(welfare.get("maximum_reward_reconstruction_error", math.inf))
        )
        <= 1e-9
    )


def _load_methods(path: Path, protocol_hash: str) -> tuple[Path, dict[str, dict]]:
    resolved, payload = _load_frozen(
        path,
        EXPECTED_SHA256["methods"],
        "transit-evaluation-methods-v19-completion-safety-v1",
    )
    rows = payload.get("methods", [])
    aliases = payload.get("aliases", [])
    if (
        payload.get("study_stage") != "development"
        or payload.get("screen_type") != "paired-mechanism-screen"
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("v19_development_evaluation_outcomes_read") is not False
        or payload.get("predesignated_candidate") != CANDIDATE
        or payload.get("physical_method_count") != 15
        or payload.get("logical_method_count") != 15
        or payload.get("expected_new_remote_sumo_episodes") != 135
        or payload.get("v19_protocol_sha256") != protocol_hash
        or len(rows) != 15
        or len(aliases) != 15
    ):
        raise ValueError("v19 methods scope drifted")
    methods = {}
    behaviors = set()
    for row in rows:
        match = METHOD.fullmatch(row.get("name", ""))
        variant = match.group("variant") if match else ""
        expected_enabled = variant != "baseline"
        expected_floor = {
            "baseline": 20.0,
            "quarter-load-fifty": 50.0,
            "quarter-load-sixty": 60.0,
        }.get(variant)
        key = (
            row.get("training_seed"),
            row.get("actor_tensor_sha256"),
            variant,
            row.get("completion_safety_reserve_enabled"),
            row.get("completion_safety_reserve_floor_seconds"),
        )
        if (
            match is None
            or row["name"] in methods
            or key in behaviors
            or row.get("algorithm") != match.group("algorithm")
            or row.get("training_seed") != int(match.group("seed"))
            or row.get("online_events") != 25_000
            or row.get("artifact_lineage") != "v19-completion-safety"
            or row.get("parent_artifact_lineage") != "v17-tail-service-reserve"
            or row.get("parent_method")
            != (
                "h2oplus-v17-tail-service-reserve-twenty-"
                f"seed{row.get('training_seed')}-events000025000"
            )
            or row.get("parent_tail_service_floor_seconds") != 20.0
            or row.get("completion_safety_reserve_enabled") is not expected_enabled
            or row.get("completion_safety_reserve_floor_seconds") != expected_floor
            or row.get("activation_start_fraction") != 0.15
            or row.get("activation_end_fraction") != 0.75
            or row.get("max_onboard_capacity_fraction") != 0.25
            or row.get("predesignated_candidate")
            is not (variant == "quarter-load-sixty")
            or row.get("v19_protocol_sha256") != protocol_hash
        ):
            raise ValueError(f"malformed v19 method: {row!r}")
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = resolved.parent / checkpoint
        checkpoint = checkpoint.resolve(strict=True)
        if sha256(checkpoint) != row.get("checkpoint_sha256"):
            raise ValueError(f"v19 parent checkpoint drifted: {row['name']}")
        methods[row["name"]] = row
        behaviors.add(key)
    if {row.get("name") for row in aliases} != set(methods):
        raise ValueError("v19 aliases do not exactly cover physical methods")
    return resolved, methods


def _scientific_sources_bound(source_hashes: dict, source_freeze: dict) -> bool:
    for source in source_freeze.get("matrix_sources", []):
        suffix = str(source.get("path", ""))
        expected = source.get("sha256")
        if not any(
            str(path).endswith(suffix) and digest == expected
            for path, digest in source_hashes.items()
        ):
            return False
    return True


def _validate_policy(payload: dict, method: dict, protocol_hash: str) -> None:
    policy = payload.get("policy", {})
    reserve = policy.get("completion_safety_reserve", {})
    variant = method["algorithm"][len("h2oplus-v19-completion-safety-") :]
    expected = {
        "artifact_lineage": "v19-completion-safety-reserve",
        "parent_artifact_lineage": "v17-tail-service-reserve",
        "parent_method": method["parent_method"],
        "sha256": method["checkpoint_sha256"],
        "actor_tensor_sha256": method["actor_tensor_sha256"],
        "temperature_tensor_sha256": method["temperature_tensor_sha256"],
    }
    mismatches = {
        key: (policy.get(key), value)
        for key, value in expected.items()
        if policy.get(key) != value
    }
    training = policy.get("training_metadata", {})
    if (
        mismatches
        or reserve.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-v1"
        or reserve.get("variant") != variant
        or reserve.get("parent_tail_service_floor_seconds") != 20.0
        or reserve.get("reserve_enabled")
        is not method["completion_safety_reserve_enabled"]
        or reserve.get("reserve_floor_seconds")
        != method["completion_safety_reserve_floor_seconds"]
        or reserve.get("activation_start_fraction") != 0.15
        or reserve.get("activation_end_fraction") != 0.75
        or reserve.get("max_onboard_capacity_fraction") != 0.25
        or reserve.get("requires_forward_bus_present") is not True
        or reserve.get("requires_backward_bus_absent") is not True
        or reserve.get("requires_positive_eligible_arrival_rate") is not True
        or reserve.get("parent_action_preserved_outside_guard") is not True
        or reserve.get("physical_max_hold_preserved") is not True
        or reserve.get("predesignated_candidate")
        is not (variant == "quarter-load-sixty")
        or reserve.get("checkpoint_selection_by_behavior") is not False
        or reserve.get("v19_protocol_sha256") != protocol_hash
        or reserve.get("formal_held_out_outcomes_read") is not False
        or training.get("formal_held_out_outcomes_read") is not False
        or training.get("training_seed") != method["training_seed"]
        or training.get("run_signature") != method["training_run_signature"]
        or training.get("training_checkpoint_sha256")
        != method["training_checkpoint_sha256"]
    ):
        raise ValueError(f"v19 result policy provenance drifted: {mismatches}")


def _load_rows(
    roots: list[Path],
    methods_path: Path,
    methods: dict[str, dict],
    protocol_hash: str,
    source_freeze: dict,
) -> tuple[list[dict], list[dict]]:
    if len(roots) != 3:
        raise ValueError("v19 analysis requires exactly three matrix blocks")
    rows = []
    manifests = []
    seen_signatures = set()
    seen_blocks = set()
    for raw_root in roots:
        root = raw_root.expanduser().resolve(strict=True)
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        source_hashes = manifest.get("source_sha256")
        blocks = {str(job.get("environment_block")) for job in jobs}
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != sha256(methods_path)
            or len(jobs) != 45
            or len(blocks) != 1
            or not isinstance(source_hashes, dict)
            or not _scientific_sources_bound(source_hashes, source_freeze)
        ):
            raise ValueError(f"v19 matrix is incomplete or source-drifted: {root}")
        block = next(iter(blocks))
        if block not in BLOCKS or block in seen_blocks:
            raise ValueError(f"unexpected or duplicate v19 block: {block}")
        seen_blocks.add(block)
        manifests.append(
            {
                "block": block,
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
                "jobs": len(jobs),
                "source_file_count": len(source_hashes),
            }
        )
        for job in jobs:
            name = job.get("method", {}).get("name")
            if name not in methods:
                raise ValueError("v19 matrix contains an unregistered behavior")
            method = methods[name]
            output = _resolve_output(root, job["output"])
            payload = json.loads(output.read_text())
            scale = float(job["scenario"]["demand_scale"])
            signature = job.get("signature")
            runtime = payload.get("runtime", {})
            version = runtime.get("sumo_runtime_fingerprint", {}).get(
                "backend_api", {}
            )
            if (
                payload.get("schema_version") != "transit-evaluation-v7"
                or payload.get("formal_held_out_outcomes_read") not in (None, False)
                or payload.get("evaluation_job", {}).get("signature") != signature
                or payload.get("method") != name
                or payload.get("source_sha256") != source_hashes
                or scale not in SCALES
                or signature in seen_signatures
                or runtime.get("device") != "cpu"
                or runtime.get("python_version") != "3.8.20"
                or runtime.get("sumo_backend") != "libsumo"
                or version.get("version_line") != "SUMO 1.25.0"
                or not _ledger_and_service_pass(payload)
            ):
                raise ValueError(f"v19 result binding or runtime drifted: {output}")
            seen_signatures.add(signature)
            _validate_policy(payload, method, protocol_hash)
            rows.append(
                {
                    "algorithm": method["algorithm"],
                    "seed": method["training_seed"],
                    "block": block,
                    "scale": scale,
                    "primary": _finite(
                        _nested(
                            payload,
                            "passenger_welfare",
                            "accrued_generalized_seconds_per_departed_passenger",
                        ),
                        "primary endpoint",
                    ),
                    "completion": _finite(
                        _nested(payload, "passengers", "completion_rate"),
                        "completion rate",
                    ),
                    "unfinished_passengers": int(
                        _nested(payload, "passengers", "unfinished_passengers")
                    ),
                    "output": str(output),
                    "sha256": sha256(output),
                }
            )
    expected = {
        (algorithm, seed, block, scale)
        for algorithm in ALGORITHMS
        for seed in SEEDS
        for block in BLOCKS
        for scale in SCALES
    }
    observed = {
        (row["algorithm"], row["seed"], row["block"], row["scale"])
        for row in rows
    }
    if len(rows) != 135 or observed != expected or len(seen_signatures) != 135:
        raise ValueError("v19 result grid is not exactly 135 unique paired episodes")
    return rows, manifests


def _means(rows: list[dict], algorithm: str) -> dict[str, float]:
    selected = [row for row in rows if row["algorithm"] == algorithm]
    return {
        "primary": statistics.fmean(row["primary"] for row in selected),
        "completion": statistics.fmean(row["completion"] for row in selected),
        "unfinished_passengers": statistics.fmean(
            row["unfinished_passengers"] for row in selected
        ),
    }


def _paired(rows: list[dict], algorithm: str) -> list[dict]:
    lookup = {
        (row["algorithm"], row["seed"], row["block"], row["scale"]): row
        for row in rows
    }
    pairs = []
    for seed in SEEDS:
        for block in BLOCKS:
            for scale in SCALES:
                baseline = lookup[(BASELINE, seed, block, scale)]
                candidate = lookup[(algorithm, seed, block, scale)]
                pairs.append(
                    {
                        "seed": seed,
                        "block": block,
                        "scale": scale,
                        "primary_increase_seconds": (
                            candidate["primary"] - baseline["primary"]
                        ),
                        "completion_gain": (
                            candidate["completion"] - baseline["completion"]
                        ),
                        "unfinished_passenger_change": (
                            candidate["unfinished_passengers"]
                            - baseline["unfinished_passengers"]
                        ),
                    }
                )
    return pairs


def _group_pair_means(pairs: list[dict], key: str) -> dict[str, dict[str, float]]:
    values = sorted({row[key] for row in pairs})
    return {
        str(value): {
            "primary_increase_seconds": statistics.fmean(
                row["primary_increase_seconds"] for row in pairs if row[key] == value
            ),
            "completion_gain": statistics.fmean(
                row["completion_gain"] for row in pairs if row[key] == value
            ),
            "unfinished_passenger_change": statistics.fmean(
                row["unfinished_passenger_change"]
                for row in pairs
                if row[key] == value
            ),
        }
        for value in values
    }


def main() -> int:
    args = parse_args()
    protocol_path, protocol = _load_frozen(
        args.protocol,
        EXPECTED_SHA256["protocol"],
        "transit-h2oplus-v19-completion-safety-protocol-v1",
    )
    registration_path, registration = _load_frozen(
        args.registration,
        EXPECTED_SHA256["registration"],
        "transit-h2oplus-v19-completion-safety-registration-v1",
    )
    environment_path, environment = _load_frozen(
        args.environment,
        EXPECTED_SHA256["environment"],
        "transit-h2oplus-v19-completion-safety-environment-v1",
    )
    methods_path, methods = _load_methods(args.methods, sha256(protocol_path))
    source_freeze_path = args.source_freeze.expanduser().resolve(strict=True)
    source_freeze = json.loads(source_freeze_path.read_text())
    analyzer = source_freeze.get("analyzer", {})
    if (
        source_freeze.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-source-freeze-v1"
        or source_freeze.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("v19_development_evaluation_outcomes_read") is not False
        or source_freeze.get("methods", {}).get("sha256")
        != EXPECTED_SHA256["methods"]
        or source_freeze.get("protocol", {}).get("sha256")
        != EXPECTED_SHA256["protocol"]
        or analyzer.get("sha256") != sha256(Path(__file__).resolve())
    ):
        raise ValueError("v19 source freeze is not bound to this analyzer")
    v18_audit_path, v18_audit = _load_frozen(
        args.v18_audit,
        EXPECTED_SHA256["v18_audit"],
        "transit-h2oplus-v18-empty-tail-reserve-evaluation-audit-v3",
    )
    v18_closeout_path, v18_closeout = _load_frozen(
        args.v18_closeout,
        EXPECTED_SHA256["v18_closeout"],
        "transit-h2oplus-v18-empty-tail-reserve-evaluation-closeout-v3",
    )
    v18_verification_path, v18_verification = _load_frozen(
        args.v18_verification,
        EXPECTED_SHA256["v18_verification"],
        "transit-h2oplus-v18-empty-tail-reserve-evaluation-closeout-verification-v3",
    )
    design_audit_path, design_audit = _load_frozen(
        args.design_audit,
        EXPECTED_SHA256["design_audit"],
        "transit-h2oplus-v19-completion-safety-design-audit-v2",
    )
    design_verification_path, design_verification = _load_frozen(
        args.design_verification,
        EXPECTED_SHA256["design_verification"],
        "transit-h2oplus-v19-completion-safety-design-verification-v2",
    )
    data_audit_path, data_audit = _load_frozen(
        args.data_environment_audit,
        EXPECTED_SHA256["data_environment_audit"],
        "transit-h2oplus-v17-data-environment-convergence-audit-v1",
    )
    data_verification_path, data_verification = _load_frozen(
        args.data_environment_verification,
        EXPECTED_SHA256["data_environment_verification"],
        "transit-h2oplus-v17-data-environment-convergence-audit-verification-v1",
    )
    if (
        protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("v19_development_evaluation_outcomes_read") is not False
        or protocol.get("screen_type") != "paired-mechanism-screen"
        or protocol.get("environment", {}).get("sha256")
        != EXPECTED_SHA256["environment"]
        or registration.get("screen_type") != "paired-mechanism-screen"
        or environment.get("disjointness", {}).get("verified_disjoint") is not True
        or environment.get("environment_cells") != 9
        or v18_audit.get("passed") is not True
        or v18_closeout.get("passed") is not True
        or v18_verification.get("passed") is not True
        or design_audit.get("claim_boundary", {}).get(
            "development_result_available"
        )
        is not False
        or design_verification.get("passed") is not True
        or data_audit.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_verification.get("passed") is not True
    ):
        raise ValueError("v19 protocol, parent closeout, or validity audit drifted")
    rows, manifests = _load_rows(
        args.matrix_root,
        methods_path,
        methods,
        sha256(protocol_path),
        source_freeze,
    )
    means = {algorithm: _means(rows, algorithm) for algorithm in ALGORITHMS}
    candidate_pairs = _paired(rows, CANDIDATE)
    diagnostic_pairs = _paired(rows, DIAGNOSTIC)
    candidate_mean = {
        "primary_increase_seconds": statistics.fmean(
            row["primary_increase_seconds"] for row in candidate_pairs
        ),
        "completion_gain": statistics.fmean(
            row["completion_gain"] for row in candidate_pairs
        ),
        "unfinished_passenger_change": statistics.fmean(
            row["unfinished_passenger_change"] for row in candidate_pairs
        ),
    }
    by_seed = _group_pair_means(candidate_pairs, "seed")
    by_scale = _group_pair_means(candidate_pairs, "scale")
    positive_seeds = sum(row["completion_gain"] > 0.0 for row in by_seed.values())
    positive_scales = sum(row["completion_gain"] > 0.0 for row in by_scale.values())
    gate_spec = protocol["analysis_gates"]
    checks = {
        "all_135_results_valid": len(rows) == 135,
        "all_three_matrix_manifests_complete": len(manifests) == 3,
        "parent_artifact_integrity_passed": v18_audit.get("passed") is True,
        "parent_training_stability_passed": (
            v18_closeout.get("parent_training_stability", {}).get("passed") is True
            or v18_closeout.get("passed") is True
        ),
        "candidate_mean_completion_gain": candidate_mean["completion_gain"]
        >= gate_spec[
            "candidate_mean_completion_gain_over_paired_baseline_at_least"
        ],
        "candidate_mean_primary_noninferiority": candidate_mean[
            "primary_increase_seconds"
        ]
        <= gate_spec[
            "candidate_mean_primary_increase_over_paired_baseline_at_most_seconds"
        ],
        "candidate_completion_gain_positive_all_demand_scales": positive_scales
        >= gate_spec[
            "candidate_positive_completion_gain_demand_scale_count_at_least"
        ],
        "candidate_completion_gain_positive_at_least_four_training_seeds": (
            positive_seeds
            >= gate_spec[
                "candidate_positive_completion_gain_training_seed_count_at_least"
            ]
        ),
    }
    passed = all(checks.values())
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v19 analysis: {output}")
    payload = {
        "schema_version": "transit-h2oplus-v19-completion-safety-analysis-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "formal_held_out_outcomes_read": False,
        "prior_v14_v15_v16_v17_v18_development_outcomes_used_for_design": True,
        "artifact_integrity": {
            "passed": True,
            "result_count": len(rows),
            "matrix_manifests": manifests,
            "parent_v18_audit": {
                "path": str(v18_audit_path),
                "sha256": sha256(v18_audit_path),
            },
        },
        "parent_training_stability": {
            "passed": checks["parent_training_stability_passed"],
            "closeout": {
                "path": str(v18_closeout_path),
                "sha256": sha256(v18_closeout_path),
            },
            "verification": {
                "path": str(v18_verification_path),
                "sha256": sha256(v18_verification_path),
            },
        },
        "data_environment_validity_risks": {
            "material_factors_present": True,
            "fresh_development_environment_used": True,
            "remaining_runtime_shift_from_training": True,
            "adaptive_prior_development_used_for_design": True,
            "audit": {
                "path": str(data_audit_path),
                "sha256": sha256(data_audit_path),
            },
            "verification": {
                "path": str(data_verification_path),
                "sha256": sha256(data_verification_path),
            },
            "v19_design_audit": {
                "path": str(design_audit_path),
                "sha256": sha256(design_audit_path),
            },
            "v19_design_verification": {
                "path": str(design_verification_path),
                "sha256": sha256(design_verification_path),
            },
        },
        "development_scientific_result": {
            "candidate": CANDIDATE,
            "paired_baseline": BASELINE,
            "diagnostic": DIAGNOSTIC,
            "algorithm_means": means,
            "candidate_paired_mean": candidate_mean,
            "candidate_paired_by_training_seed": by_seed,
            "candidate_paired_by_demand_scale": by_scale,
            "candidate_positive_completion_gain_training_seeds": positive_seeds,
            "candidate_positive_completion_gain_demand_scales": positive_scales,
            "diagnostic_paired_mean": {
                "primary_increase_seconds": statistics.fmean(
                    row["primary_increase_seconds"] for row in diagnostic_pairs
                ),
                "completion_gain": statistics.fmean(
                    row["completion_gain"] for row in diagnostic_pairs
                ),
                "unfinished_passenger_change": statistics.fmean(
                    row["unfinished_passenger_change"] for row in diagnostic_pairs
                ),
            },
            "gate_specification": gate_spec,
            "checks": checks,
            "paired_mechanism_screen_passed": passed,
        },
        "eligible_for_full_development_control_matrix": passed,
        "formal_confirmation_eligible": False,
        "formal_confirmation_authorized": False,
        "claim_boundary": (
            "A passing v19 paired mechanism screen permits registration of a fresh "
            "full development control matrix only. It is not a formal held-out "
            "effect and does not create formal confirmation eligibility or authorization."
        ),
        "bindings": {
            "methods": {"path": str(methods_path), "sha256": sha256(methods_path)},
            "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
            "registration": {
                "path": str(registration_path),
                "sha256": sha256(registration_path),
            },
            "environment": {
                "path": str(environment_path),
                "sha256": sha256(environment_path),
            },
            "source_freeze": {
                "path": str(source_freeze_path),
                "sha256": sha256(source_freeze_path),
            },
        },
        "result_files": [
            {"path": row["output"], "sha256": row["sha256"]}
            for row in sorted(rows, key=lambda item: item["output"])
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": sha256(output),
                "result_count": len(rows),
                "paired_mechanism_screen_passed": passed,
                "eligible_for_full_development_control_matrix": passed,
                "formal_confirmation_eligible": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
