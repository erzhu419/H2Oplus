#!/usr/bin/env python3
"""Analyze the preregistered v20 endpoint-only formal confirmation matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256  # noqa: E402
from reproducibility.run_evaluation_matrix_v20_formal_confirmation import (  # noqa: E402
    ALGORITHMS,
    CANDIDATE,
    DIRECT_PARENT,
    LINEAGE_CONTROLS,
    METHODS_SCHEMA,
    RAW_PARENT,
    SEEDS,
    load_formal_methods,
)


EXPECTED = {
    "registration": "c33111cd72b741a801804cd065f42629094d45c0fe03a5b88f9d95dba62149ce",
    "protocol": "10aacced63f4c8a1b7f6f55024f46dfdebbcae6737c6bbafb79d227b2ad44664",
    "methods": "b4bcad09e13cb50c94d5541f2a52340ce99f15b281b5ddef57be38e792d052cd",
    "environment": "50b831923b76a81c773cf7dc19dcc2e10c729ea8fc7b3be6982fc01b7c3cc141",
    "v20_audit": "2e516ab3ea0cff10dc88607d6fb8adf3e3888a822e32a14f84a86953a466f583",
    "v20_closeout": "dbd1da5442219f404f90da6fbc415b0755f7dddd41d28b9b581039d359882132",
    "v20_verification": "08b25f8b229a9fefbfec198f8198aff5db4f9f2680f16d9bf5521985cdfe7f7e",
    "v19_methods": "6d63b7dccf261859db7040d52b93482e17c8360d6db53b902de3f3f569729bb8",
    "data_environment_audit": "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    "data_environment_verification": "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
}
BLOCKS = tuple(f"v20-formal-confirmation-{index:02d}" for index in range(1, 11))
SCALES = (0.75, 1.0, 1.25)
COMPARISON_CONTROLS = (RAW_PARENT, *LINEAGE_CONTROLS)
V19_PROTOCOL_SHA256 = "fa5f629588d35b1435e28bd39931f49d9851811ea3e308d62672f9eacf1f7184"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--methods", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--v20-audit", type=Path, required=True)
    parser.add_argument("--v20-closeout", type=Path, required=True)
    parser.add_argument("--v20-verification", type=Path, required=True)
    parser.add_argument("--data-environment-audit", type=Path, required=True)
    parser.add_argument("--data-environment-verification", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, nargs=10, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_frozen(path: Path, expected: str, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected:
        raise ValueError(f"frozen v20 analysis input drifted: {resolved}")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"frozen v20 analysis schema drifted: {resolved}")
    return resolved, payload


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {label}")
    return result


def nested(payload: dict, *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def ledger_and_service_pass(payload: dict) -> bool:
    semantics = payload.get("bus_service_semantics_audit", {})
    service = payload.get("service", {})
    ledger = payload.get("ledger_audit", {})
    counts = ledger.get("count_source_audit", {})
    passengers = payload.get("passengers", {})
    welfare = payload.get("passenger_welfare", {})
    coverage = welfare.get("chronological_coverage_errors", {})
    return (
        semantics.get("passed") is True
        and service.get("active_trips_at_horizon") == 0
        and service.get("completed_trips") == service.get("departed_trips")
        and service.get("completion_rate") == 1.0
        and ledger.get("legacy_sampled_trip_histories_used_as_ground_truth") is False
        and counts.get("max_in_vehicle_count_error") == 0
        and counts.get("max_waiting_count_error") == 0
        and counts.get("subscription_fallback_queries") == 0
        and passengers.get("legacy_sampled_trip_histories_used_for_welfare") is False
        and passengers.get("active_state_count_balance_error") == 0
        and welfare.get("primary") is True
        and isinstance(coverage, dict)
        and bool(coverage)
        and all(abs(float(value)) <= 1e-9 for value in coverage.values())
        and abs(float(welfare.get("maximum_reward_reconstruction_error", math.inf)))
        <= 1e-9
    )


def source_is_bound(source_hashes: dict, relative: str, expected: str) -> bool:
    return any(
        str(path).endswith(relative) and digest == expected
        for path, digest in source_hashes.items()
    )


def resolve_output(root: Path, raw: str) -> Path:
    value = Path(raw).expanduser()
    if value.is_file():
        return value.resolve()
    local = root / value.name
    if local.is_file():
        return local.resolve()
    matches = list(root.rglob(value.name))
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"cannot resolve v20 output {raw!r} under {root}")


def validate_policy(payload: dict, method: dict, v19_raw: dict[str, dict]) -> None:
    policy = payload.get("policy", {})
    training = policy.get("training_metadata", {})
    mismatches = {
        "sha256": (policy.get("sha256"), method["checkpoint_sha256"]),
        "actor_tensor_sha256": (
            policy.get("actor_tensor_sha256"),
            method["actor_tensor_sha256"],
        ),
        "training_seed": (training.get("training_seed"), method["training_seed"]),
        "run_signature": (
            training.get("run_signature"),
            method["training_run_signature"],
        ),
        "training_checkpoint_sha256": (
            training.get("training_checkpoint_sha256"),
            method["training_checkpoint_sha256"],
        ),
        "training_admission_sha256": (
            training.get("training_admission", {}).get("sha256"),
            method["training_admission_sha256"],
        ),
    }
    mismatches = {key: value for key, value in mismatches.items() if value[0] != value[1]}
    if mismatches:
        raise ValueError(f"v20 result policy provenance drifted: {mismatches}")
    if method["algorithm"] not in {CANDIDATE, DIRECT_PARENT}:
        return
    raw = v19_raw[method["name"]]
    reserve = policy.get("completion_safety_reserve", {})
    variant = method["algorithm"][len("h2oplus-v19-completion-safety-") :]
    if (
        policy.get("artifact_lineage") != "v19-completion-safety-reserve"
        or policy.get("parent_method") != raw["parent_method"]
        or reserve.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-v1"
        or reserve.get("variant") != variant
        or reserve.get("reserve_enabled")
        is not raw["completion_safety_reserve_enabled"]
        or reserve.get("reserve_floor_seconds")
        != raw["completion_safety_reserve_floor_seconds"]
        or reserve.get("activation_start_fraction") != 0.15
        or reserve.get("activation_end_fraction") != 0.75
        or reserve.get("max_onboard_capacity_fraction") != 0.25
        or reserve.get("requires_forward_bus_present") is not True
        or reserve.get("requires_backward_bus_absent") is not True
        or reserve.get("requires_positive_eligible_arrival_rate") is not True
        or reserve.get("v19_protocol_sha256") != V19_PROTOCOL_SHA256
        or reserve.get("checkpoint_selection_by_behavior") is not False
        or reserve.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError(f"v20 v19-policy transform binding drifted: {method['name']}")


def load_rows(
    roots: list[Path],
    methods_path: Path,
    methods: dict[str, dict],
    source_freeze: dict,
) -> tuple[list[dict], list[dict]]:
    matrix_sources = source_freeze.get("matrix_sources", [])
    v19_path = REPRO / "h2oplus_v19_completion_safety_evaluation_methods_v1.json"
    if sha256(v19_path.resolve(strict=True)) != EXPECTED["v19_methods"]:
        raise ValueError("v19 raw method evidence drifted")
    v19_raw = {
        row["name"]: row for row in json.loads(v19_path.read_text())["methods"]
    }
    rows, manifests = [], []
    seen_signatures, seen_blocks = set(), set()
    for raw_root in roots:
        root = raw_root.expanduser().resolve(strict=True)
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        source_hashes = manifest.get("source_sha256", {})
        blocks = {str(job.get("environment_block")) for job in jobs}
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != sha256(methods_path)
            or len(jobs) != 135
            or len(blocks) != 1
            or not isinstance(source_hashes, dict)
            or any(
                not source_is_bound(
                    source_hashes, str(source.get("path", "")), source.get("sha256")
                )
                for source in matrix_sources
            )
        ):
            raise ValueError(
                f"v20 formal matrix is incomplete or source-drifted: {root}"
            )
        block = next(iter(blocks))
        if block not in BLOCKS or block in seen_blocks:
            raise ValueError(f"unexpected or duplicate v20 block: {block}")
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
                raise ValueError("v20 matrix contains an unregistered behavior")
            method = methods[name]
            output = resolve_output(root, job["output"])
            payload = json.loads(output.read_text())
            scale = float(job["scenario"]["demand_scale"])
            signature = job.get("signature")
            runtime = payload.get("runtime", {})
            backend_api = runtime.get("sumo_runtime_fingerprint", {}).get(
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
                or backend_api.get("version_line") != "SUMO 1.25.0"
                or not ledger_and_service_pass(payload)
            ):
                raise ValueError(f"v20 result binding or runtime drifted: {output}")
            seen_signatures.add(signature)
            validate_policy(payload, method, v19_raw)
            rows.append(
                {
                    "algorithm": method["algorithm"],
                    "seed": method["training_seed"],
                    "block": block,
                    "scale": scale,
                    "primary": finite(
                        nested(
                            payload,
                            "passenger_welfare",
                            "accrued_generalized_seconds_per_departed_passenger",
                        ),
                        "primary endpoint",
                    ),
                    "completion": finite(
                        nested(payload, "passengers", "completion_rate"),
                        "completion rate",
                    ),
                    "unfinished_passengers": int(
                        nested(payload, "passengers", "unfinished_passengers")
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
    if len(rows) != 1350 or observed != expected or len(seen_signatures) != 1350:
        raise ValueError(
            "v20 formal grid is not exactly 1350 unique paired episodes"
        )
    return rows, manifests


def means(rows: list[dict], algorithm: str) -> dict[str, float]:
    selected = [row for row in rows if row["algorithm"] == algorithm]
    return {
        "primary": statistics.fmean(row["primary"] for row in selected),
        "completion": statistics.fmean(row["completion"] for row in selected),
        "unfinished_passengers": statistics.fmean(
            row["unfinished_passengers"] for row in selected
        ),
    }


def paired(rows: list[dict], comparison: str) -> list[dict]:
    lookup = {
        (row["algorithm"], row["seed"], row["block"], row["scale"]): row
        for row in rows
    }
    result = []
    for seed in SEEDS:
        for block in BLOCKS:
            for scale in SCALES:
                candidate = lookup[(CANDIDATE, seed, block, scale)]
                control = lookup[(comparison, seed, block, scale)]
                result.append(
                    {
                        "seed": seed,
                        "block": block,
                        "scale": scale,
                        "primary_change_seconds": candidate["primary"]
                        - control["primary"],
                        "completion_change": candidate["completion"]
                        - control["completion"],
                        "unfinished_passenger_change": candidate[
                            "unfinished_passengers"
                        ]
                        - control["unfinished_passengers"],
                    }
                )
    return result


def grouped(pairs: list[dict], key: str) -> dict[str, dict[str, float]]:
    values = sorted({row[key] for row in pairs})
    return {
        str(value): {
            "primary_change_seconds": statistics.fmean(
                row["primary_change_seconds"] for row in pairs if row[key] == value
            ),
            "completion_change": statistics.fmean(
                row["completion_change"] for row in pairs if row[key] == value
            ),
            "unfinished_passenger_change": statistics.fmean(
                row["unfinished_passenger_change"]
                for row in pairs
                if row[key] == value
            ),
        }
        for value in values
    }


def one_sided_sign_pvalue(wins: int, total: int = 10) -> float:
    if wins < 0 or wins > total:
        raise ValueError(f"invalid sign-test count: {wins}/{total}")
    return sum(math.comb(total, count) for count in range(wins, total + 1)) / (
        2**total
    )


def main() -> int:
    args = parse_args()
    registration_path, registration = load_frozen(
        args.registration,
        EXPECTED["registration"],
        "transit-h2oplus-v20-formal-confirmation-registration-v1",
    )
    protocol_path, protocol = load_frozen(
        args.protocol,
        EXPECTED["protocol"],
        "transit-h2oplus-v20-formal-confirmation-protocol-v1",
    )
    environment_path, environment = load_frozen(
        args.environment,
        EXPECTED["environment"],
        "transit-h2oplus-v20-formal-confirmation-environment-v1",
    )
    methods_path = args.methods.expanduser().resolve(strict=True)
    if sha256(methods_path) != EXPECTED["methods"]:
        raise ValueError("v20 methods identity drifted")
    method_rows = load_formal_methods(methods_path, METHODS_SCHEMA)
    methods = {row["name"]: row for row in method_rows}
    source_freeze_path = args.source_freeze.expanduser().resolve(strict=True)
    source_freeze = json.loads(source_freeze_path.read_text())
    if (
        source_freeze.get("schema_version")
        != "transit-h2oplus-v20-formal-confirmation-source-freeze-v1"
        or source_freeze.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("formal_confirmation_outcomes_read") is not False
        or source_freeze.get("formal_confirmation_submission_authorized")
        is not False
        or source_freeze.get("methods", {}).get("sha256") != EXPECTED["methods"]
        or source_freeze.get("protocol", {}).get("sha256") != EXPECTED["protocol"]
        or source_freeze.get("analyzer", {}).get("sha256")
        != sha256(Path(__file__).resolve())
    ):
        raise ValueError("v20 source freeze is not bound to this analyzer")
    v20_audit_path, v20_audit = load_frozen(
        args.v20_audit,
        EXPECTED["v20_audit"],
        "transit-h2oplus-v20-full-control-evaluation-audit-v1",
    )
    v20_closeout_path, v20_closeout = load_frozen(
        args.v20_closeout,
        EXPECTED["v20_closeout"],
        "transit-h2oplus-v20-full-control-evaluation-closeout-v1",
    )
    v20_verification_path, v20_verification = load_frozen(
        args.v20_verification,
        EXPECTED["v20_verification"],
        "transit-h2oplus-v20-full-control-evaluation-closeout-verification-v1",
    )
    data_audit_path, data_audit = load_frozen(
        args.data_environment_audit,
        EXPECTED["data_environment_audit"],
        "transit-h2oplus-v17-data-environment-convergence-audit-v1",
    )
    data_verification_path, data_verification = load_frozen(
        args.data_environment_verification,
        EXPECTED["data_environment_verification"],
        "transit-h2oplus-v17-data-environment-convergence-audit-verification-v1",
    )
    if (
        registration.get("matrix_type") != "endpoint-only-formal-confirmation"
        or registration.get("formal_held_out_outcomes_read") is not False
        or registration.get("formal_confirmation_evaluation_authorized_or_submitted")
        is not False
        or protocol.get("matrix_type") != "endpoint-only-formal-confirmation"
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("analysis", {}).get("submission_authorized") is not False
        or environment.get("environment_cells") != 30
        or environment.get("disjointness", {}).get("verified_disjoint") is not True
        or v20_audit.get("passed") is not True
        or v20_audit.get("artifact_integrity", {}).get("passed") is not True
        or v20_audit.get("parent_training_stability", {}).get("passed") is not True
        or v20_closeout.get("passed") is not True
        or v20_closeout.get("formal_confirmation_eligible") is not True
        or v20_closeout.get("artifact_integrity", {}).get("passed") is not True
        or v20_closeout.get("parent_training_stability", {}).get("passed") is not True
        or v20_closeout.get("formal_confirmation_authorized_or_submitted") is not False
        or v20_closeout.get("formal_held_out_outcomes_read") is not False
        or v20_verification.get("passed") is not True
        or v20_verification.get("artifact_integrity_verified") is not True
        or v20_verification.get("parent_training_stability_verified") is not True
        or v20_verification.get("formal_confirmation_eligible") is not True
        or v20_verification.get("formal_confirmation_authorized_or_submitted")
        is not False
        or v20_verification.get("formal_held_out_outcomes_read") is not False
        or data_audit.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_verification.get("passed") is not True
    ):
        raise ValueError("formal protocol, v20 closeout or validity boundary drifted")

    rows, manifests = load_rows(
        list(args.matrix_root), methods_path, methods, source_freeze
    )
    algorithm_means = {algorithm: means(rows, algorithm) for algorithm in ALGORITHMS}
    direct_pairs = paired(rows, DIRECT_PARENT)
    direct_mean = {
        "primary_change_seconds": statistics.fmean(
            row["primary_change_seconds"] for row in direct_pairs
        ),
        "completion_change": statistics.fmean(
            row["completion_change"] for row in direct_pairs
        ),
        "unfinished_passenger_change": statistics.fmean(
            row["unfinished_passenger_change"] for row in direct_pairs
        ),
    }
    direct_by_block = grouped(direct_pairs, "block")
    direct_by_seed = grouped(direct_pairs, "seed")
    direct_by_scale = grouped(direct_pairs, "scale")
    completion_margin = float(
        protocol["analysis_gates"][
            "candidate_completion_noninferiority_margin_vs_direct_parent"
        ]
    )
    direct_primary_block_wins = sum(
        value["primary_change_seconds"] < 0.0
        for value in direct_by_block.values()
    )
    direct_completion_ni_block_wins = sum(
        value["completion_change"] > -completion_margin
        for value in direct_by_block.values()
    )
    comparisons = {}
    for control in COMPARISON_CONTROLS:
        pairs = paired(rows, control)
        by_block = grouped(pairs, "block")
        by_seed = grouped(pairs, "seed")
        comparisons[control] = {
            "mean_primary_change_seconds": statistics.fmean(
                row["primary_change_seconds"] for row in pairs
            ),
            "mean_completion_change": statistics.fmean(
                row["completion_change"] for row in pairs
            ),
            "primary_block_wins": sum(
                value["primary_change_seconds"] < 0.0 for value in by_block.values()
            ),
            "primary_one_sided_sign_pvalue": one_sided_sign_pvalue(
                sum(
                    value["primary_change_seconds"] < 0.0
                    for value in by_block.values()
                )
            ),
            "by_formal_environment_block": by_block,
            "by_training_seed": by_seed,
            "by_demand_scale": grouped(pairs, "scale"),
        }
    gates = protocol["analysis_gates"]
    checks = {
        "all_1350_results_valid": len(rows) == 1350,
        "all_ten_135_job_matrix_manifests_complete": len(manifests) == 10,
        "parent_artifact_integrity_passed": v20_audit.get("passed") is True
        and v20_audit.get("artifact_integrity", {}).get("passed") is True
        and v20_verification.get("artifact_integrity_verified") is True,
        "parent_training_stability_passed": v20_closeout.get(
            "parent_training_stability", {}
        ).get("passed")
        is True
        and v20_audit.get("parent_training_stability", {}).get("passed") is True
        and v20_verification.get("parent_training_stability_verified") is True,
        "candidate_mean_primary_change_vs_direct_parent_less_than_seconds": (
            direct_mean["primary_change_seconds"]
            < gates[
                "candidate_mean_primary_change_vs_direct_parent_less_than_seconds"
            ]
        ),
        "candidate_primary_block_wins_vs_direct_parent_at_least": (
            direct_primary_block_wins
            >= gates["candidate_primary_block_wins_vs_direct_parent_at_least"]
        ),
        "candidate_mean_completion_noninferior_to_direct_parent": (
            direct_mean["completion_change"] >= -completion_margin
        ),
        "candidate_completion_noninferiority_blocks_vs_direct_parent_at_least": (
            direct_completion_ni_block_wins
            >= gates[
                "candidate_completion_noninferiority_blocks_vs_direct_parent_at_least"
            ]
        ),
        "candidate_primary_better_at_each_demand_scale": all(
            value["primary_change_seconds"] < 0.0
            for value in direct_by_scale.values()
        ),
        "candidate_completion_noninferior_at_each_demand_scale": all(
            value["completion_change"] >= -completion_margin
            for value in direct_by_scale.values()
        ),
        "candidate_mean_primary_lower_than_every_comparison_control": all(
            value["mean_primary_change_seconds"] < 0.0
            for value in comparisons.values()
        ),
        "candidate_completion_noninferior_to_every_comparison_control": all(
            value["mean_completion_change"]
            >= -gates["candidate_completion_noninferiority_margin_per_control"]
            for value in comparisons.values()
        ),
        "candidate_primary_block_wins_vs_sac_at_least": (
            comparisons["sac-online"]["primary_block_wins"]
            >= gates["candidate_primary_block_wins_vs_sac_at_least"]
        ),
    }
    confirmatory_checks = {
        name: checks[name]
        for name in (
            "all_1350_results_valid",
            "all_ten_135_job_matrix_manifests_complete",
            "parent_artifact_integrity_passed",
            "parent_training_stability_passed",
            "candidate_mean_primary_change_vs_direct_parent_less_than_seconds",
            "candidate_primary_block_wins_vs_direct_parent_at_least",
            "candidate_mean_completion_noninferior_to_direct_parent",
            "candidate_completion_noninferiority_blocks_vs_direct_parent_at_least",
        )
    }
    confirmatory_passed = all(confirmatory_checks.values())
    passed = all(checks.values())
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v20 analysis: {output}")
    payload = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-analysis-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "formal-confirmation",
        "matrix_type": "endpoint-only-formal-confirmation",
        "formal_held_out_outcomes_read": True,
        "formal_confirmation_outcomes_analyzed": True,
        "prior_v14_through_v20_development_outcomes_used_for_design": True,
        "artifact_integrity": {
            "passed": checks["all_1350_results_valid"]
            and checks["all_ten_135_job_matrix_manifests_complete"],
            "result_count": len(rows),
            "matrix_manifests": manifests,
            "parent_v20_audit": {
                "path": str(v20_audit_path),
                "sha256": sha256(v20_audit_path),
            },
        },
        "parent_training_stability": {
            "passed": checks["parent_training_stability_passed"],
            "closeout": {
                "path": str(v20_closeout_path),
                "sha256": sha256(v20_closeout_path),
            },
            "verification": {
                "path": str(v20_verification_path),
                "sha256": sha256(v20_verification_path),
            },
        },
        "data_environment_validity_risks": {
            "material_factors_present": True,
            "fresh_formal_confirmation_environment_used": True,
            "remaining_runtime_shift_from_training": True,
            "adaptive_prior_development_used_for_candidate_design": True,
            "audit": {"path": str(data_audit_path), "sha256": sha256(data_audit_path)},
            "verification": {
                "path": str(data_verification_path),
                "sha256": sha256(data_verification_path),
            },
        },
        "formal_confirmation_scientific_result": {
            "candidate": CANDIDATE,
            "direct_behavior_parent": DIRECT_PARENT,
            "comparison_controls": list(COMPARISON_CONTROLS),
            "endpoint_only": True,
            "aulc_computed_or_claimed": False,
            "algorithm_means": algorithm_means,
            "candidate_vs_direct_parent_mean": direct_mean,
            "candidate_vs_direct_parent_by_formal_environment_block": (
                direct_by_block
            ),
            "candidate_vs_direct_parent_by_training_seed": direct_by_seed,
            "candidate_vs_direct_parent_by_demand_scale": direct_by_scale,
            "confirmatory_inference": {
                "independent_units": 10,
                "within_block_pairs_averaged": 15,
                "primary_block_wins": direct_primary_block_wins,
                "primary_one_sided_sign_pvalue": one_sided_sign_pvalue(
                    direct_primary_block_wins
                ),
                "completion_noninferiority_margin": completion_margin,
                "completion_noninferiority_block_wins": (
                    direct_completion_ni_block_wins
                ),
                "completion_one_sided_sign_pvalue": one_sided_sign_pvalue(
                    direct_completion_ni_block_wins
                ),
                "one_sided_alpha": 0.025,
                "ties_count_as_nonwins": True,
                "intersection_union_passed": confirmatory_passed,
                "claim_limited_to_direct_behavior_parent": True,
            },
            "candidate_control_comparisons": comparisons,
            "gate_specification": gates,
            "checks": checks,
            "failed_checks": sorted(name for name, value in checks.items() if not value),
            "all_preregistered_gates_passed": passed,
            "formal_confirmation_passed": passed,
        },
        "formal_confirmation_passed": passed,
        "supports_formal_held_out_effect_claim": passed,
        "claim_boundary": (
            "The confirmatory effect claim, if every preregistered gate passes, "
            "is limited to the fixed candidate versus its direct behavior parent; "
            "lineage-control comparisons are supportive and do not expand it."
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
                "formal_confirmation_passed": passed,
                "confirmatory_intersection_union_passed": confirmatory_passed,
                "formal_held_out_outcomes_read": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
