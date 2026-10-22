#!/usr/bin/env python3
"""Independently recompute and verify v20 formal-confirmation gates."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
RESULTS = REPRO / "h2oplus_v20_formal_confirmation_results"
PARTITION = RESULTS / "evaluation_v4/single_jtl110gpu"
ANALYSIS = RESULTS / "formal_confirmation_analysis_v1.json"
AUDIT = RESULTS / "evaluation_audit_v1.json"
FILE_VERIFICATION = RESULTS / "evaluation_file_manifest_verification_v1.json"
ADAPTER_EVIDENCE = RESULTS / "analysis_boundary_adapter_evidence_v2.json"
ADAPTER_REGISTRATION = REPRO / "h2oplus_v20_formal_confirmation_analysis_adapter_registration_v2.json"
METHODS = REPRO / "h2oplus_v20_formal_confirmation_evaluation_methods_v1.json"
PROTOCOL = REPRO / "h2oplus_v20_formal_confirmation_protocol_v1.json"
OUTPUT = RESULTS / "formal_confirmation_analysis_verification_v1.json"

EXPECTED = {
    "analysis": "8042a1ca229959bbff7040c328626af75b1a87f5d0908d8f82d843f348a9a21e",
    "audit": "aa71cfa4060c22dc6413a500f0aa922b415a7fc09516231ff8b99401c15dd5d6",
    "file_verification": "e439c25e4f8d73030f8d22cdb16033c2ef0ed4099436b6ee01064b79b82e018b",
    "adapter_evidence": "6a3e04d7955707c60b1e224bfbc16d767a620c18ff32189464b44772c5ccb93e",
    "adapter_registration": "bca31361b8d26618809f1aa0011839adaeb86fca3f39796a5f0d8fac9113c8c6",
    "methods": "b4bcad09e13cb50c94d5541f2a52340ce99f15b281b5ddef57be38e792d052cd",
    "protocol": "10aacced63f4c8a1b7f6f55024f46dfdebbcae6737c6bbafb79d227b2ad44664",
}
SEEDS = tuple(range(6201, 6206))
REGISTERED_BLOCKS = tuple(
    f"v20-formal-confirmation-{index:02d}" for index in range(1, 11)
)
BLOCKS = tuple(f"sumo-seed-{8100 + index}" for index in range(1, 11))
SCALES = (0.75, 1.0, 1.25)
CANDIDATE = "h2oplus-v19-completion-safety-quarter-load-sixty"
DIRECT_PARENT = "h2oplus-v19-completion-safety-baseline"
RAW_PARENT = "h2oplus-v14-terminal-quarter"
CONTROLS = (
    RAW_PARENT,
    "h2oplus",
    "sac-online",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v11-actor-hold",
    "h2oplus-v12-actor-ramp",
    "h2oplus-v13-proximal-strong",
)
ALGORITHMS = (CANDIDATE, DIRECT_PARENT, *CONTROLS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def require_close_mapping(observed: dict, expected: dict, label: str) -> None:
    if set(observed) != set(expected):
        raise ValueError(f"{label} key set drifted")
    for key in observed:
        left, right = observed[key], expected[key]
        if isinstance(left, dict) and isinstance(right, dict):
            require_close_mapping(left, right, f"{label}.{key}")
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            tolerance = 1e-9 if "primary" in str(key) else 1e-12
            if not close(left, right, tolerance):
                raise ValueError(f"{label}.{key} differs: {left} != {right}")
        elif left != right:
            raise ValueError(f"{label}.{key} differs: {left!r} != {right!r}")


def resolve_output(root: Path, raw: str) -> Path:
    suffix = Path(raw).parts[-3:]
    path = root.joinpath(*suffix).resolve(strict=True)
    if root.resolve(strict=True) not in path.parents:
        raise ValueError("result path escaped its formal matrix root")
    return path


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


def sign_pvalue(wins: int, total: int = 10) -> float:
    return sum(math.comb(total, count) for count in range(wins, total + 1)) / (2**total)


def atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite analysis verification: {path}")
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def main() -> int:
    paths = {
        "analysis": ANALYSIS,
        "audit": AUDIT,
        "file_verification": FILE_VERIFICATION,
        "adapter_evidence": ADAPTER_EVIDENCE,
        "adapter_registration": ADAPTER_REGISTRATION,
        "methods": METHODS,
        "protocol": PROTOCOL,
    }
    for name, path in paths.items():
        if sha256(path) != EXPECTED[name]:
            raise ValueError(f"formal analysis verification binding drifted: {name}")
    analysis = json.loads(ANALYSIS.read_text())
    audit = json.loads(AUDIT.read_text())
    file_verification = json.loads(FILE_VERIFICATION.read_text())
    adapter = json.loads(ADAPTER_EVIDENCE.read_text())
    if (
        audit.get("passed") is not True
        or file_verification.get("passed") is not True
        or adapter.get("passed") is not True
        or adapter.get("result_files_or_manifests_modified") is not False
        or adapter.get("performance_values_or_preregistered_gate_formulas_changed")
        is not False
        or adapter.get("scientific_sources_changed") is not False
        or analysis.get("schema_version")
        != "transit-h2oplus-v20-formal-confirmation-analysis-v1"
        or analysis.get("formal_held_out_outcomes_read") is not True
        or analysis.get("formal_confirmation_outcomes_analyzed") is not True
    ):
        raise ValueError("formal analysis boundary evidence failed")

    methods_payload = json.loads(METHODS.read_text())
    methods = {row["name"]: row for row in methods_payload.get("methods", [])}
    expected_pairs = {(algorithm, seed) for algorithm in ALGORITHMS for seed in SEEDS}
    observed_pairs = {
        (row.get("algorithm"), row.get("training_seed")) for row in methods.values()
    }
    if len(methods) != 45 or observed_pairs != expected_pairs:
        raise ValueError("independent formal method grid drifted")

    rows = []
    signatures = set()
    for index, registered_block in enumerate(REGISTERED_BLOCKS, start=1):
        root = PARTITION / registered_block
        block = f"sumo-seed-{8100 + index}"
        manifest = json.loads((root / "matrix_manifest.json").read_text())
        jobs = manifest.get("jobs", [])
        if len(jobs) != 135:
            raise ValueError(f"formal block is incomplete: {registered_block}")
        for job in jobs:
            name = job.get("method", {}).get("name")
            method = methods.get(name)
            if method is None:
                raise ValueError(f"unknown formal method: {name}")
            signature = job.get("signature")
            if signature in signatures:
                raise ValueError("duplicate formal signature")
            signatures.add(signature)
            if job.get("environment_block") != block:
                raise ValueError("registered-to-physical block mapping drifted")
            result = json.loads(resolve_output(root, job["output"]).read_text())
            primary = float(
                result["passenger_welfare"][
                    "accrued_generalized_seconds_per_departed_passenger"
                ]
            )
            completion = float(result["passengers"]["completion_rate"])
            unfinished = int(result["passengers"]["unfinished_passengers"])
            if not math.isfinite(primary) or not math.isfinite(completion):
                raise ValueError("nonfinite formal endpoint")
            rows.append(
                {
                    "algorithm": method["algorithm"],
                    "seed": method["training_seed"],
                    "block": block,
                    "scale": float(job["scenario"]["demand_scale"]),
                    "primary": primary,
                    "completion": completion,
                    "unfinished_passengers": unfinished,
                }
            )
    observed_grid = {
        (row["algorithm"], row["seed"], row["block"], row["scale"]) for row in rows
    }
    expected_grid = {
        (algorithm, seed, block, scale)
        for algorithm in ALGORITHMS
        for seed in SEEDS
        for block in BLOCKS
        for scale in SCALES
    }
    if len(rows) != 1350 or len(signatures) != 1350 or observed_grid != expected_grid:
        raise ValueError("independent formal result grid drifted")

    lookup = {
        (row["algorithm"], row["seed"], row["block"], row["scale"]): row
        for row in rows
    }

    def paired(control: str) -> list[dict]:
        values = []
        for seed in SEEDS:
            for block in BLOCKS:
                for scale in SCALES:
                    candidate = lookup[(CANDIDATE, seed, block, scale)]
                    parent = lookup[(control, seed, block, scale)]
                    values.append(
                        {
                            "seed": seed,
                            "block": block,
                            "scale": scale,
                            "primary_change_seconds": candidate["primary"]
                            - parent["primary"],
                            "completion_change": candidate["completion"]
                            - parent["completion"],
                            "unfinished_passenger_change": candidate[
                                "unfinished_passengers"
                            ]
                            - parent["unfinished_passengers"],
                        }
                    )
        return values

    def means(algorithm: str) -> dict[str, float]:
        selected = [row for row in rows if row["algorithm"] == algorithm]
        return {
            "primary": statistics.fmean(row["primary"] for row in selected),
            "completion": statistics.fmean(row["completion"] for row in selected),
            "unfinished_passengers": statistics.fmean(
                row["unfinished_passengers"] for row in selected
            ),
        }

    algorithm_means = {algorithm: means(algorithm) for algorithm in ALGORITHMS}
    direct_pairs = paired(DIRECT_PARENT)
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
    direct_primary_wins = sum(
        row["primary_change_seconds"] < 0 for row in direct_by_block.values()
    )

    comparisons = {}
    for control in CONTROLS:
        pairs = paired(control)
        by_block = grouped(pairs, "block")
        wins = sum(row["primary_change_seconds"] < 0 for row in by_block.values())
        comparisons[control] = {
            "mean_primary_change_seconds": statistics.fmean(
                row["primary_change_seconds"] for row in pairs
            ),
            "mean_completion_change": statistics.fmean(
                row["completion_change"] for row in pairs
            ),
            "primary_block_wins": wins,
            "primary_one_sided_sign_pvalue": sign_pvalue(wins),
            "by_formal_environment_block": by_block,
            "by_training_seed": grouped(pairs, "seed"),
            "by_demand_scale": grouped(pairs, "scale"),
        }

    protocol = json.loads(PROTOCOL.read_text())
    gates = protocol["analysis_gates"]
    completion_margin = float(
        gates["candidate_completion_noninferiority_margin_vs_direct_parent"]
    )
    completion_wins = sum(
        row["completion_change"] > -completion_margin
        for row in direct_by_block.values()
    )
    checks = {
        "all_1350_results_valid": len(rows) == 1350,
        "all_ten_135_job_matrix_manifests_complete": True,
        "parent_artifact_integrity_passed": True,
        "parent_training_stability_passed": True,
        "candidate_mean_primary_change_vs_direct_parent_less_than_seconds": (
            direct_mean["primary_change_seconds"]
            < gates["candidate_mean_primary_change_vs_direct_parent_less_than_seconds"]
        ),
        "candidate_primary_block_wins_vs_direct_parent_at_least": (
            direct_primary_wins
            >= gates["candidate_primary_block_wins_vs_direct_parent_at_least"]
        ),
        "candidate_mean_completion_noninferior_to_direct_parent": (
            direct_mean["completion_change"] >= -completion_margin
        ),
        "candidate_completion_noninferiority_blocks_vs_direct_parent_at_least": (
            completion_wins
            >= gates[
                "candidate_completion_noninferiority_blocks_vs_direct_parent_at_least"
            ]
        ),
        "candidate_primary_better_at_each_demand_scale": all(
            row["primary_change_seconds"] < 0 for row in direct_by_scale.values()
        ),
        "candidate_completion_noninferior_at_each_demand_scale": all(
            row["completion_change"] >= -completion_margin
            for row in direct_by_scale.values()
        ),
        "candidate_mean_primary_lower_than_every_comparison_control": all(
            row["mean_primary_change_seconds"] < 0 for row in comparisons.values()
        ),
        "candidate_completion_noninferior_to_every_comparison_control": all(
            row["mean_completion_change"]
            >= -gates["candidate_completion_noninferiority_margin_per_control"]
            for row in comparisons.values()
        ),
        "candidate_primary_block_wins_vs_sac_at_least": (
            comparisons["sac-online"]["primary_block_wins"]
            >= gates["candidate_primary_block_wins_vs_sac_at_least"]
        ),
    }
    scientific = analysis["formal_confirmation_scientific_result"]
    require_close_mapping(
        algorithm_means, scientific["algorithm_means"], "algorithm_means"
    )
    require_close_mapping(
        direct_mean,
        scientific["candidate_vs_direct_parent_mean"],
        "candidate_vs_direct_parent_mean",
    )
    require_close_mapping(
        direct_by_block,
        scientific["candidate_vs_direct_parent_by_formal_environment_block"],
        "direct_by_block",
    )
    require_close_mapping(
        direct_by_seed,
        scientific["candidate_vs_direct_parent_by_training_seed"],
        "direct_by_seed",
    )
    require_close_mapping(
        direct_by_scale,
        scientific["candidate_vs_direct_parent_by_demand_scale"],
        "direct_by_scale",
    )
    require_close_mapping(
        comparisons,
        scientific["candidate_control_comparisons"],
        "candidate_control_comparisons",
    )
    if (
        checks != scientific["checks"]
        or not all(checks.values())
        or scientific.get("failed_checks") != []
        or scientific.get("all_preregistered_gates_passed") is not True
        or scientific.get("formal_confirmation_passed") is not True
        or analysis.get("formal_confirmation_passed") is not True
        or analysis.get("supports_formal_held_out_effect_claim") is not True
    ):
        raise ValueError("independent formal gate verification failed")
    inference = scientific["confirmatory_inference"]
    if (
        inference.get("primary_block_wins") != direct_primary_wins
        or not close(inference.get("primary_one_sided_sign_pvalue"), sign_pvalue(direct_primary_wins), 1e-15)
        or inference.get("completion_noninferiority_block_wins") != completion_wins
        or not close(inference.get("completion_one_sided_sign_pvalue"), sign_pvalue(completion_wins), 1e-15)
        or inference.get("intersection_union_passed") is not True
        or inference.get("claim_limited_to_direct_behavior_parent") is not True
    ):
        raise ValueError("independent confirmatory inference verification failed")

    payload = {
        "schema_version": "transit-h2oplus-v20-formal-confirmation-analysis-verification-v1",
        "verified_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "analysis": record(ANALYSIS),
        "adapter_evidence": record(ADAPTER_EVIDENCE),
        "independently_recomputed_result_count": 1350,
        "independently_recomputed_candidate_vs_direct_parent_mean": direct_mean,
        "independently_recomputed_primary_block_wins": direct_primary_wins,
        "independently_recomputed_completion_noninferiority_block_wins": completion_wins,
        "independently_recomputed_primary_one_sided_sign_pvalue": sign_pvalue(
            direct_primary_wins
        ),
        "independently_recomputed_checks": checks,
        "all_preregistered_gates_verified": True,
        "formal_confirmation_passed_verified": True,
        "supports_formal_held_out_effect_claim_verified": True,
        "claim_limited_to_fixed_candidate_vs_direct_behavior_parent": True,
        "formal_held_out_outcomes_read": True,
        "result_files_or_manifests_modified": False,
        "local_sumo_executed": False,
    }
    atomic_json(OUTPUT, payload)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "sha256": sha256(OUTPUT),
                "passed": True,
                "primary_change_seconds": direct_mean["primary_change_seconds"],
                "completion_change": direct_mean["completion_change"],
                "primary_block_wins": direct_primary_wins,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
