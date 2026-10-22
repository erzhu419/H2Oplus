#!/usr/bin/env python3
"""Audit and analyze the development-only H2O+ few-shot comparison."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bus_h2o.sumo_provenance import sha256


REPORT_SCHEMA = "transit-h2oplus-v8-fewshot-analysis-v1"
METHOD = re.compile(
    r"^(?P<algorithm>h2oplus|sac-online)-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SEEDS = list(range(6201, 6206))
PRIMARY_BUDGETS = [0, 5_000, 10_000, 25_000, 50_000, 100_000]
UNIQUE_BUDGETS = {
    "h2oplus": [0, 10_000, 25_000, 50_000, 100_000],
    "sac-online": [0, 10_000, 25_000, 50_000, 55_000, 60_000, 75_000, 100_000, 150_000],
}
LEARNER_STEP_PAIRS = [
    (5_000, 55_000, 50_000),
    (10_000, 60_000, 55_000),
    (25_000, 75_000, 70_000),
    (50_000, 100_000, 95_000),
    (100_000, 150_000, 145_000),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, action="append", required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _nested(payload: dict, *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def _finite(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {label}")
    return result


def _resolve_output(root: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    local = root / candidate.name
    matches = list(root.rglob(candidate.name))
    if local.is_file():
        return local.resolve()
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"cannot resolve matrix output {raw!r} under {root}")


def _load_rows(roots: list[Path], protocol_hash: str) -> tuple[list[dict], list[dict]]:
    if len(roots) != 3:
        raise ValueError("few-shot analysis requires exactly three merged block roots")
    rows = []
    manifests = []
    seen = set()
    for raw_root in roots:
        root = raw_root.expanduser().resolve()
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or len(jobs) != 210
        ):
            raise ValueError(f"few-shot matrix block is incomplete: {root}")
        manifests.append(
            {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
                "jobs": len(jobs),
            }
        )
        for job in jobs:
            method = job["method"]
            match = METHOD.fullmatch(method["name"])
            if match is None:
                raise ValueError("matrix contains an unregistered method name")
            algorithm = match.group("algorithm")
            seed = int(match.group("seed"))
            events = int(match.group("events"))
            if seed not in SEEDS or events not in UNIQUE_BUDGETS[algorithm]:
                raise ValueError("matrix method lies outside the frozen budget grid")
            output = _resolve_output(root, job["output"])
            payload = json.loads(output.read_text())
            training = payload.get("policy", {}).get("training_metadata", {})
            if (
                payload.get("schema_version") != "transit-evaluation-v7"
                or payload.get("evaluation_job", {}).get("signature") != job["signature"]
                or payload.get("method") != method["name"]
                or training.get("study_stage") != "development"
                or training.get("formal_held_out_outcomes_read") is not False
                or training.get("fewshot_online_events_nominal") != events
                or training.get("fewshot_protocol", {}).get("sha256") != protocol_hash
                or training.get("run_signature") != method["training_run_signature"]
                or training.get("training_checkpoint_sha256")
                != method["training_checkpoint_sha256"]
                or training.get("training_admission", {}).get("sha256")
                != method["training_admission_sha256"]
            ):
                raise ValueError(f"few-shot result provenance failed: {output}")
            if any(
                abs(float(value)) > 1e-9
                for value in payload.get("passenger_welfare", {})
                .get("chronological_coverage_errors", {})
                .values()
            ):
                raise ValueError(f"few-shot welfare ledger failed: {output}")
            scenario = job["scenario"]
            block = str(job["environment_block"])
            scale = float(scenario["demand_scale"])
            key = (algorithm, seed, events, block, scale)
            if key in seen:
                raise ValueError(f"duplicate few-shot result: {key}")
            seen.add(key)
            rows.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "events": events,
                    "actual_events": int(training["fewshot_online_events_actual"]),
                    "learner_update_steps": int(training["fewshot_learner_update_steps"]),
                    "ratio_optimizer_steps": int(training["fewshot_ratio_optimizer_steps"]),
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
                    "output": str(output),
                    "sha256": sha256(output),
                }
            )
    if len(rows) != 630:
        raise ValueError(f"few-shot analysis requires 630 unique rows, got {len(rows)}")
    return rows, manifests


def _verify_training(root: Path, protocol_hash: str) -> list[dict]:
    evidence = []
    for algorithm in ("h2oplus", "sac-online"):
        expected = set(UNIQUE_BUDGETS[algorithm]) | {5_000}
        for seed in SEEDS:
            run = root / f"{algorithm}_seed{seed}"
            export = json.loads((run / "fewshot_export_summary.json").read_text())
            policy_manifest = json.loads(
                (run / "fewshot_policies/policy_manifest.json").read_text()
            )
            entries = policy_manifest.get("entries", [])
            by_events = {entry["online_events_nominal"]: entry for entry in entries}
            if (
                export.get("passed") is not True
                or export.get("protocol_sha256") != protocol_hash
                or policy_manifest.get("protocol_sha256") != protocol_hash
                or set(by_events) != expected
                or by_events[0]["actor_tensor_sha256"]
                != by_events[5_000]["actor_tensor_sha256"]
            ):
                raise ValueError(f"training milestone audit failed: {run}")
            evidence.append(
                {
                    "run": str(run),
                    "method": algorithm,
                    "seed": seed,
                    "export_summary_sha256": sha256(
                        run / "fewshot_export_summary.json"
                    ),
                    "policy_manifest_sha256": sha256(
                        run / "fewshot_policies/policy_manifest.json"
                    ),
                    "zero_and_5000_actor_identical": True,
                }
            )
    return evidence


def _cell_means(rows: list[dict]) -> dict[tuple[str, int, int], dict[str, float]]:
    grouped: dict[tuple[str, int, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["algorithm"], row["seed"], row["events"]), []).append(row)
    result = {}
    for key, values in grouped.items():
        if len(values) != 9:
            raise ValueError(f"few-shot policy does not cover nine development cells: {key}")
        result[key] = {
            "primary": float(np.mean([item["primary"] for item in values])),
            "completion": float(np.mean([item["completion"] for item in values])),
        }
    return result


def _value(means: dict, algorithm: str, seed: int, events: int, metric: str) -> float:
    source_events = 0 if events == 5_000 else events
    return float(means[(algorithm, seed, source_events)][metric])


def _aulc(means: dict, algorithm: str, seed: int, metric: str) -> float:
    values = [_value(means, algorithm, seed, events, metric) for events in PRIMARY_BUDGETS]
    return float(np.trapz(values, np.asarray(PRIMARY_BUDGETS, dtype=np.float64)) / 100_000.0)


def exact_sign_flip_p(differences: np.ndarray) -> float:
    observed = float(differences.mean())
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        values.append(float(np.mean(differences * np.asarray(signs))))
    return float(sum(value >= observed - 1e-12 for value in values) / len(values))


def _scale_aulc(rows: list[dict], algorithm: str, scale: float) -> float:
    points = []
    for events in PRIMARY_BUDGETS:
        source_events = 0 if events == 5_000 else events
        values = [
            row["primary"]
            for row in rows
            if row["algorithm"] == algorithm
            and row["events"] == source_events
            and math.isclose(row["scale"], scale)
        ]
        if len(values) != 15:
            raise ValueError("demand-scale AULC coverage drifted")
        points.append(float(np.mean(values)))
    return float(np.trapz(points, np.asarray(PRIMARY_BUDGETS)) / 100_000.0)


def main() -> int:
    args = parse_args()
    registration_path = args.registration.expanduser().resolve(strict=True)
    protocol_path = args.protocol.expanduser().resolve(strict=True)
    registration = json.loads(registration_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if (
        registration.get("schema_version")
        != "transit-h2oplus-v8-fewshot-registration-v1"
        or protocol.get("schema_version") != "transit-h2oplus-v8-fewshot-protocol-v1"
        or registration.get("formal_held_out_outcomes_read") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError("few-shot registration or protocol scope drifted")
    protocol_hash = sha256(protocol_path)
    training_evidence = _verify_training(
        args.training_root.expanduser().resolve(), protocol_hash
    )
    rows, matrix_evidence = _load_rows(args.matrix_root, protocol_hash)
    means = _cell_means(rows)
    h2o_aulc = np.asarray(
        [_aulc(means, "h2oplus", seed, "primary") for seed in SEEDS]
    )
    sac_aulc = np.asarray(
        [_aulc(means, "sac-online", seed, "primary") for seed in SEEDS]
    )
    differences = sac_aulc - h2o_aulc
    relative = float((sac_aulc.mean() - h2o_aulc.mean()) / sac_aulc.mean())
    thresholds = registration["fewshot_confirmation_gate"]
    seed_wins = int(np.sum(h2o_aulc < sac_aulc))
    p_value = exact_sign_flip_p(differences)
    early = {}
    for events in (10_000, 25_000):
        h2o = float(np.mean([_value(means, "h2oplus", seed, events, "primary") for seed in SEEDS]))
        sac = float(np.mean([_value(means, "sac-online", seed, events, "primary") for seed in SEEDS]))
        early[str(events)] = {
            "h2oplus": h2o,
            "sac_online": sac,
            "h2oplus_lower": h2o < sac,
            "relative_improvement": (sac - h2o) / sac,
        }
    completion_delta = float(
        np.mean([_aulc(means, "h2oplus", seed, "completion") for seed in SEEDS])
        - np.mean([_aulc(means, "sac-online", seed, "completion") for seed in SEEDS])
    )
    scale_rows = []
    for scale in (0.85, 1.0, 1.15):
        h2o = _scale_aulc(rows, "h2oplus", scale)
        sac = _scale_aulc(rows, "sac-online", scale)
        scale_rows.append(
            {
                "demand_scale": scale,
                "h2oplus_aulc": h2o,
                "sac_online_aulc": sac,
                "h2oplus_relative_degradation": (h2o - sac) / sac,
            }
        )
    final_h2o = float(np.mean([_value(means, "h2oplus", seed, 100_000, "primary") for seed in SEEDS]))
    final_sac = float(np.mean([_value(means, "sac-online", seed, 100_000, "primary") for seed in SEEDS]))
    final_degradation = (final_h2o - final_sac) / final_sac
    checks = {
        "practical_aulc_improvement": relative
        >= float(thresholds["mean_relative_aulc_improvement_at_least"]),
        "seed_robustness": seed_wins
        >= int(thresholds["training_seeds_with_lower_aulc_at_least"]),
        "exact_paired_sign_flip": p_value
        < float(thresholds["exact_one_sided_paired_sign_flip_p_less_than"]),
        "mean_at_10000_lower": early["10000"]["h2oplus_lower"],
        "mean_at_25000_lower": early["25000"]["h2oplus_lower"],
        "completion_noninferiority": completion_delta
        >= -float(thresholds["completion_rate_noninferiority_margin"]),
        "demand_scale_robustness": all(
            item["h2oplus_relative_degradation"]
            <= float(thresholds["no_demand_scale_aulc_relative_degradation_greater_than"])
            for item in scale_rows
        ),
        "final_100000_noncatastrophic": final_degradation
        <= float(thresholds["final_100000_event_relative_degradation_not_greater_than"]),
    }
    compute_matched = []
    for h2o_events, sac_events, learner_steps in LEARNER_STEP_PAIRS:
        h2o = float(np.mean([_value(means, "h2oplus", seed, h2o_events, "primary") for seed in SEEDS]))
        sac = float(np.mean([_value(means, "sac-online", seed, sac_events, "primary") for seed in SEEDS]))
        compute_matched.append(
            {
                "learner_update_steps": learner_steps,
                "equal_total_optimizer_steps": False,
                "h2oplus_online_events": h2o_events,
                "sac_online_events": sac_events,
                "h2oplus_mean": h2o,
                "sac_online_mean": sac,
                "h2oplus_relative_improvement": (sac - h2o) / sac,
                "supports_few_shot_claim": False,
            }
        )
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {
            "passed": True,
            "training_runs": len(training_evidence),
            "evaluation_rows": len(rows),
            "matrix_blocks": len(matrix_evidence),
        },
        "development_scientific_result": {
            "fewshot_gate_passed": all(checks.values()),
            "checks": checks,
            "h2oplus_mean_aulc": float(h2o_aulc.mean()),
            "sac_online_mean_aulc": float(sac_aulc.mean()),
            "h2oplus_relative_aulc_improvement": relative,
            "seed_wins": seed_wins,
            "exact_one_sided_sign_flip_p": p_value,
            "per_seed": [
                {
                    "seed": seed,
                    "h2oplus_aulc": float(h2o_aulc[index]),
                    "sac_online_aulc": float(sac_aulc[index]),
                    "h2oplus_lower": bool(h2o_aulc[index] < sac_aulc[index]),
                }
                for index, seed in enumerate(SEEDS)
            ],
            "early_budgets": early,
            "completion_aulc_delta": completion_delta,
            "demand_scale_aulc": scale_rows,
            "final_100000_relative_degradation": final_degradation,
            "compute_matched_secondary": compute_matched,
        },
        "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
        "protocol": {"path": str(protocol_path), "sha256": protocol_hash},
        "training_evidence": training_evidence,
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
