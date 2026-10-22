#!/usr/bin/env python3
"""Audit and analyze the development-only H2O+ v9 adaptive pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bus_h2o.sumo_provenance import sha256


REPORT_SCHEMA = "transit-h2oplus-v9-adaptive-analysis-v1"
ALGORITHMS = (
    "h2oplus",
    "sac-online",
    "h2oplus-v9-ratio-gated",
    "h2oplus-v9-adaptation",
    "h2oplus-v9-combined",
)
SEEDS = (6201, 6202, 6203)
BUDGETS = (0, 5_000, 10_000, 25_000)
SCALES = (0.85, 1.0, 1.15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, action="append", required=True)
    parser.add_argument("--methods-manifest", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-closeout", type=Path, required=True)
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def nested(payload: dict, *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {label}")
    return result


def resolve_output(root: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    matches = list(root.rglob(candidate.name))
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"cannot resolve matrix output {raw!r} under {root}")


def load_methods(path: Path) -> tuple[dict[str, dict], list[dict]]:
    manifest = path.expanduser().resolve(strict=True)
    payload = json.loads(manifest.read_text())
    methods = payload.get("methods", [])
    aliases = payload.get("aliases", [])
    if (
        payload.get("schema_version") != "transit-evaluation-methods-v9-adaptive-v1"
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("physical_method_count") != 42
        or payload.get("method_budget_alias_count") != 60
        or len(methods) != 42
        or len(aliases) != 60
    ):
        raise ValueError("v9 methods manifest drifted")
    by_name = {item["name"]: item for item in methods}
    if len(by_name) != 42:
        raise ValueError("duplicate physical methods")
    logical = set()
    for alias in aliases:
        key = (alias["algorithm"], alias["training_seed"], alias["online_events"])
        if (
            alias.get("name") in logical
            or alias.get("physical_method") not in by_name
            or alias.get("actor_tensor_sha256")
            != by_name[alias["physical_method"]]["actor_tensor_sha256"]
            or alias["algorithm"] not in ALGORITHMS
            or alias["training_seed"] not in SEEDS
            or alias["online_events"] not in BUDGETS
        ):
            raise ValueError(f"invalid logical method alias: {alias}")
        logical.add(alias["name"])
    expected = {(algorithm, seed, budget) for algorithm in ALGORITHMS for seed in SEEDS for budget in BUDGETS}
    actual = {(item["algorithm"], item["training_seed"], item["online_events"]) for item in aliases}
    if actual != expected or len(logical) != 60:
        raise ValueError("logical method grid drifted")
    return by_name, aliases


def load_physical_rows(roots: list[Path], methods: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    if len(roots) != 2:
        raise ValueError("v9 analysis requires exactly two merged block roots")
    rows, evidence, seen = [], [], set()
    for raw_root in roots:
        root = raw_root.expanduser().resolve(strict=True)
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        if manifest.get("schema_version") != "transit-evaluation-matrix-v7" or len(jobs) != 126:
            raise ValueError(f"v9 matrix block is incomplete: {root}")
        evidence.append({"path": str(manifest_path), "sha256": sha256(manifest_path), "jobs": 126})
        for job in jobs:
            method = job["method"]
            name = method["name"]
            frozen = methods.get(name)
            if frozen is None:
                raise ValueError(f"unregistered physical method: {name}")
            output = resolve_output(root, job["output"])
            payload = json.loads(output.read_text())
            training = payload.get("policy", {}).get("training_metadata", {})
            if (
                payload.get("schema_version") != "transit-evaluation-v7"
                or payload.get("evaluation_job", {}).get("signature") != job["signature"]
                or payload.get("method") != name
                or payload.get("source_sha256") != manifest.get("source_sha256")
                or training.get("study_stage") != "development"
                or training.get("formal_held_out_outcomes_read") is not False
                or training.get("run_signature") != frozen["training_run_signature"]
                or training.get("training_checkpoint_sha256") != frozen["training_checkpoint_sha256"]
                or training.get("training_admission", {}).get("sha256")
                != frozen["training_admission_sha256"]
                or payload.get("policy", {}).get("actor_tensor_sha256")
                != frozen["actor_tensor_sha256"]
            ):
                raise ValueError(f"v9 result provenance failed: {output}")
            coverage = payload.get("passenger_welfare", {}).get("chronological_coverage_errors", {})
            if not coverage or any(abs(float(value)) > 1e-9 for value in coverage.values()):
                raise ValueError(f"v9 welfare ledger failed: {output}")
            block = str(job["environment_block"])
            scale = float(job["scenario"]["demand_scale"])
            key = (name, block, scale)
            if key in seen:
                raise ValueError(f"duplicate physical result: {key}")
            seen.add(key)
            rows.append(
                {
                    "physical_method": name,
                    "block": block,
                    "scale": scale,
                    "primary": finite(
                        nested(payload, "passenger_welfare", "accrued_generalized_seconds_per_departed_passenger"),
                        "primary endpoint",
                    ),
                    "completion": finite(nested(payload, "passengers", "completion_rate"), "completion rate"),
                    "output": str(output),
                    "sha256": sha256(output),
                }
            )
    if len(rows) != 252:
        raise ValueError(f"v9 analysis requires 252 physical rows, got {len(rows)}")
    return rows, evidence


def expand_aliases(rows: list[dict], aliases: list[dict]) -> list[dict]:
    by_key = {(row["physical_method"], row["block"], row["scale"]): row for row in rows}
    expanded = []
    for alias in aliases:
        matching = [row for key, row in by_key.items() if key[0] == alias["physical_method"]]
        if len(matching) != 6:
            raise ValueError(f"physical actor lacks six development cells: {alias['physical_method']}")
        for row in matching:
            expanded.append(
                {
                    **row,
                    "algorithm": alias["algorithm"],
                    "seed": alias["training_seed"],
                    "events": alias["online_events"],
                    "logical_method": alias["name"],
                }
            )
    if len(expanded) != 360:
        raise ValueError("logical alias expansion must produce 360 rows")
    return expanded


def cell_means(rows: list[dict], metric: str) -> dict[tuple[str, int, int], float]:
    grouped: dict[tuple[str, int, int], list[float]] = {}
    for row in rows:
        grouped.setdefault((row["algorithm"], row["seed"], row["events"]), []).append(row[metric])
    if any(len(values) != 6 for values in grouped.values()) or len(grouped) != 60:
        raise ValueError("logical development-cell coverage drifted")
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def aulc(means: dict, algorithm: str, seed: int) -> float:
    values = [means[(algorithm, seed, budget)] for budget in BUDGETS]
    return float(np.trapz(values, np.asarray(BUDGETS, dtype=np.float64)) / 25_000.0)


def relative_improvement(candidate: float, reference: float) -> float:
    return float((reference - candidate) / reference)


def scale_aulc(rows: list[dict], algorithm: str, scale: float) -> float:
    points = []
    for budget in BUDGETS:
        values = [
            row["primary"]
            for row in rows
            if row["algorithm"] == algorithm
            and row["events"] == budget
            and math.isclose(row["scale"], scale)
        ]
        if len(values) != 6:
            raise ValueError("demand-scale AULC coverage drifted")
        points.append(float(np.mean(values)))
    return float(np.trapz(points, np.asarray(BUDGETS, dtype=np.float64)) / 25_000.0)


def main() -> int:
    args = parse_args()
    registration_path = args.registration.expanduser().resolve(strict=True)
    protocol_path = args.protocol.expanduser().resolve(strict=True)
    closeout_path = args.training_closeout.expanduser().resolve(strict=True)
    source_path = args.source_freeze.expanduser().resolve(strict=True)
    registration = json.loads(registration_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    closeout = json.loads(closeout_path.read_text())
    source_freeze = json.loads(source_path.read_text())
    if (
        registration.get("schema_version") != "transit-h2oplus-v9-adaptive-registration-v2"
        or protocol.get("schema_version") != "transit-h2oplus-v9-adaptive-protocol-v2"
        or closeout.get("schema_version") != "transit-h2oplus-v9-adaptive-training-closeout-v1"
        or source_freeze.get("schema_version") != "transit-h2oplus-v9-adaptive-evaluation-source-freeze-v1"
        or any(item.get("formal_held_out_outcomes_read") is not False for item in (registration, protocol, closeout, source_freeze))
    ):
        raise ValueError("v9 analysis control scope drifted")
    methods, aliases = load_methods(args.methods_manifest)
    physical_rows, matrix_evidence = load_physical_rows(args.matrix_root, methods)
    rows = expand_aliases(physical_rows, aliases)
    primary = cell_means(rows, "primary")
    completion = cell_means(rows, "completion")
    curves = {
        algorithm: np.asarray([aulc(primary, algorithm, seed) for seed in SEEDS])
        for algorithm in ALGORITHMS
    }
    completion_curves = {
        algorithm: np.asarray([aulc(completion, algorithm, seed) for seed in SEEDS])
        for algorithm in ALGORITHMS
    }
    combined = curves["h2oplus-v9-combined"]
    fixed = curves["h2oplus"]
    sac = curves["sac-online"]
    thresholds = registration["screening_gates"]
    early = {}
    for budget in (10_000, 25_000):
        values = {
            algorithm: float(np.mean([primary[(algorithm, seed, budget)] for seed in SEEDS]))
            for algorithm in ALGORITHMS
        }
        early[str(budget)] = {
            **values,
            "combined_lower_than_fixed_h2o_and_sac": (
                values["h2oplus-v9-combined"] < values["h2oplus"]
                and values["h2oplus-v9-combined"] < values["sac-online"]
            ),
        }
    scale_rows = []
    for scale in SCALES:
        row = {algorithm: scale_aulc(rows, algorithm, scale) for algorithm in ALGORITHMS}
        row.update(
            {
                "demand_scale": scale,
                "combined_relative_degradation_over_sac": (
                    row["h2oplus-v9-combined"] - row["sac-online"]
                ) / row["sac-online"],
            }
        )
        scale_rows.append(row)
    intervention_attribution = bool(closeout.get("intervention_attribution_available"))
    completion_margin = float(thresholds["completion_rate_noninferiority_margin"])
    checks = {
        "combined_relative_aulc_over_fixed_h2o": relative_improvement(combined.mean(), fixed.mean())
        >= float(thresholds["combined_relative_aulc_improvement_over_fixed_h2o_at_least"]),
        "combined_relative_aulc_over_sac": relative_improvement(combined.mean(), sac.mean())
        >= float(thresholds["combined_relative_aulc_improvement_over_sac_at_least"]),
        "combined_seed_wins_over_fixed_h2o": int(np.sum(combined < fixed))
        >= int(thresholds["combined_seed_wins_over_fixed_h2o"]),
        "combined_seed_wins_over_sac": int(np.sum(combined < sac))
        >= int(thresholds["combined_seed_wins_over_sac"]),
        "combined_mean_lower_at_10000_and_25000": all(
            early[str(budget)]["combined_lower_than_fixed_h2o_and_sac"] for budget in (10_000, 25_000)
        ),
        "completion_rate_noninferiority": (
            completion_curves["h2oplus-v9-combined"].mean()
            >= max(completion_curves["h2oplus"].mean(), completion_curves["sac-online"].mean()) - completion_margin
        ),
        "demand_scale_robustness": all(
            row["combined_relative_degradation_over_sac"]
            <= float(thresholds["no_demand_scale_relative_aulc_degradation_over_sac_greater_than"])
            for row in scale_rows
        ),
        "intervention_attribution": intervention_attribution
        if thresholds.get("intervention_attribution_required") is True
        else True,
    }
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {
            "passed": True,
            "physical_evaluation_rows": len(physical_rows),
            "logical_alias_rows": len(rows),
            "physical_actors": len(methods),
            "logical_method_budget_aliases": len(aliases),
            "matrix_blocks": len(matrix_evidence),
        },
        "development_scientific_result": {
            "screening_gate_passed": all(checks.values()),
            "checks": checks,
            "mean_aulc": {algorithm: float(values.mean()) for algorithm, values in curves.items()},
            "per_seed_aulc": {
                algorithm: [
                    {"seed": seed, "aulc": float(values[index])}
                    for index, seed in enumerate(SEEDS)
                ]
                for algorithm, values in curves.items()
            },
            "combined_relative_aulc_improvement_over_fixed_h2o": relative_improvement(combined.mean(), fixed.mean()),
            "combined_relative_aulc_improvement_over_sac": relative_improvement(combined.mean(), sac.mean()),
            "combined_seed_wins_over_fixed_h2o": int(np.sum(combined < fixed)),
            "combined_seed_wins_over_sac": int(np.sum(combined < sac)),
            "early_budgets": early,
            "completion_mean_aulc": {
                algorithm: float(values.mean()) for algorithm, values in completion_curves.items()
            },
            "demand_scale_aulc": scale_rows,
            "factorial_mechanism_diagnostics": {
                "ratio_only_relative_aulc_improvement_over_fixed": relative_improvement(curves["h2oplus-v9-ratio-gated"].mean(), fixed.mean()),
                "adaptation_only_relative_aulc_improvement_over_fixed": relative_improvement(curves["h2oplus-v9-adaptation"].mean(), fixed.mean()),
                "combined_relative_aulc_improvement_over_adaptation": relative_improvement(combined.mean(), curves["h2oplus-v9-adaptation"].mean()),
                "ratio_intervention_ever_applied": closeout.get("ratio_intervention_ever_applied"),
                "intervention_attribution_available": intervention_attribution,
                "training_scientific_stability_passed": closeout.get("scientific_stability_passed"),
            },
        },
        "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
        "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
        "training_closeout": {"path": str(closeout_path), "sha256": sha256(closeout_path)},
        "source_freeze": {"path": str(source_path), "sha256": sha256(source_path)},
        "methods_manifest": {
            "path": str(args.methods_manifest.expanduser().resolve()),
            "sha256": sha256(args.methods_manifest.expanduser().resolve()),
        },
        "matrix_evidence": matrix_evidence,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
        "formal_confirmation_stage_authorized_or_submitted": False,
    }
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    print(json.dumps(report["development_scientific_result"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
