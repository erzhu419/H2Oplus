#!/usr/bin/env python3
"""Audit and analyze the preregistered v17 tail-service development screen."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility import analyze_h2oplus_v14_terminal_taper as v14


REPORT_SCHEMA = "transit-h2oplus-v17-tail-service-reserve-analysis-v1"
METHODS_SCHEMA = "transit-evaluation-methods-v17-tail-service-reserve-v1"
REGISTRATION_SCHEMA = "transit-h2oplus-v17-tail-service-reserve-registration-v1"
PROTOCOL_SCHEMA = "transit-h2oplus-v17-tail-service-reserve-protocol-v1"
SOURCE_FREEZE_SCHEMA = "transit-h2oplus-v17-tail-service-reserve-evaluation-source-freeze-v1"
EXPECTED_SHA256 = {
    "registration": "d5465e677e29971eb1c6e88a926387bacd172de737f669f6fd84426bb7c68364",
    "protocol": "5f7bcde7c90f165b3ae91215f69de0dfea1f204615d0639637eeef5f51a28790",
    "methods": "f2383ced11263bdd78287f7bc48f61be13ef1f181868827f555b9c06e9038ec5",
    "v16_analysis": "4a88f48946c2ec478ca47274df39e0ccdf3535c1c21ad41141142d529617f75e",
    "v16_closeout": "ef7d456e6c71bb2fbcbd176642c01acf123f8e709cefffd2b8d4ae26a850ddf5",
    "v16_verification": "54b25d3c9307e9f926319b7f9ae543e6747d4639ce9b8af762e123bade9f4c85",
    "v14_methods": "560a91eaca3e279cb986daedf6978c8275ce4c937859d60cd2eed37e27e52c3b",
    "v14_analysis": "937a5300988311f80a84dc1220dfee3f0f96f2b6f83691d4e309c38d6bbe6856",
    "v14_closeout": "95fef5b3a5206ce14a3da164d9f2948b0dc3b5d61ee1d1b34c52c431d1cd1d32",
    "v14_verification": "0e20ccc1e7855e4feb5860e5cb334cd60237f28032a948b7325037ad0aa61dc0",
}
METHOD = re.compile(
    r"^(?P<algorithm>h2oplus-v17-tail-service-reserve-(?P<variant>twenty|thirty|forty))"
    r"-seed(?P<seed>62\d{2})-events000025000$"
)
SEEDS = tuple(range(6201, 6206))
BUDGETS = (0, 5_000, 10_000, 25_000)
SCALES = (0.85, 1.0, 1.15)
BLOCKS = tuple(f"v12-staged-release-development-0{index}" for index in range(1, 4))
PARENT = "h2oplus-v14-terminal-quarter"
CANDIDATE = "h2oplus-v17-tail-service-reserve-thirty"
DIAGNOSTICS = (
    "h2oplus-v17-tail-service-reserve-twenty",
    "h2oplus-v17-tail-service-reserve-forty",
)
V17_ALGORITHMS = (*DIAGNOSTICS[:1], CANDIDATE, *DIAGNOSTICS[1:])
CONTROLS = v14.CONTROLS
TAIL_SERVICE_FLOORS = {"twenty": 20.0, "thirty": 30.0, "forty": 40.0}
PARENT_BASE_HOLD_SCALE = 0.125
PARENT_HEADWAY_BALANCE_GAIN = 1.0
PARENT_RESERVE_HORIZON_FRACTION = 0.75
ACTIVATION_START_FRACTION = 0.5
ACTIVATION_END_FRACTION = 0.75
MAX_ONBOARD_CAPACITY_FRACTION = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v17-matrix-root", type=Path, action="append", required=True)
    parser.add_argument("--v14-matrix-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--methods-manifest",
        type=Path,
        default=ROOT / "reproducibility/h2oplus_v17_tail_service_reserve_evaluation_methods_v1.json",
    )
    parser.add_argument(
        "--registration",
        type=Path,
        default=ROOT / "reproducibility/h2oplus_v17_tail_service_reserve_registration_v1.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "reproducibility/h2oplus_v17_tail_service_reserve_protocol_v1.json",
    )
    parser.add_argument("--source-freeze", type=Path, required=True)
    parser.add_argument(
        "--v16-analysis",
        type=Path,
        default=ROOT / "reproducibility/h2oplus_v16_headway_reserve_results/headway_reserve_analysis_v1.json",
    )
    parser.add_argument(
        "--v16-closeout",
        type=Path,
        default=ROOT / "reproducibility/h2oplus_v16_headway_reserve_results/evaluation_closeout_v1.json",
    )
    parser.add_argument(
        "--v16-verification",
        type=Path,
        default=ROOT / "reproducibility/h2oplus_v16_headway_reserve_results/evaluation_closeout_verification_v1.json",
    )
    parser.add_argument(
        "--v14-methods",
        type=Path,
        default=ROOT / "reproducibility/h2oplus_v14_terminal_taper_evaluation_methods_v1.json",
    )
    parser.add_argument(
        "--v14-analysis",
        type=Path,
        default=ROOT / "reproducibility/h2oplus_v14_terminal_taper_results/terminal_taper_analysis_v1.json",
    )
    parser.add_argument(
        "--v14-closeout",
        type=Path,
        default=ROOT / "reproducibility/h2oplus_v14_terminal_taper_results/evaluation_closeout_v1.json",
    )
    parser.add_argument(
        "--v14-verification",
        type=Path,
        default=ROOT / "reproducibility/h2oplus_v14_terminal_taper_results/evaluation_closeout_verification_v1.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_control(path: Path, expected_sha: str, schema: str) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected_sha:
        raise ValueError(f"frozen identity drifted: {resolved}")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"frozen schema drifted: {resolved}")
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
    raise FileNotFoundError(f"cannot resolve v17 matrix output {raw!r} under {root}")


def _load_methods(path: Path, protocol_hash: str) -> tuple[Path, dict, dict[str, dict]]:
    resolved, payload = _load_control(path, EXPECTED_SHA256["methods"], METHODS_SCHEMA)
    methods = payload.get("methods", [])
    aliases = payload.get("aliases", [])
    if (
        payload.get("study_stage") != "development"
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("v17_development_evaluation_outcomes_read") is not False
        or payload.get("prior_v14_v15_v16_development_outcomes_used_for_design")
        is not True
        or payload.get("predesignated_candidate") != CANDIDATE
        or payload.get("physical_method_count") != 15
        or payload.get("logical_method_count") != 15
        or payload.get("expected_new_remote_sumo_episodes") != 135
        or payload.get("v17_protocol_sha256") != protocol_hash
        or len(methods) != 15
        or len(aliases) != 15
    ):
        raise ValueError("v17 methods manifest scope drifted")
    physical: dict[str, dict] = {}
    behavior_keys: set[
        tuple[int, str, float, float, float, float, float, float, float]
    ] = set()
    for row in methods:
        match = METHOD.fullmatch(row.get("name", ""))
        variant = match.group("variant") if match else ""
        floor_seconds = TAIL_SERVICE_FLOORS.get(variant)
        key = (
            row.get("training_seed"),
            row.get("actor_tensor_sha256"),
            row.get("parent_base_hold_scale"),
            row.get("parent_headway_balance_gain"),
            row.get("parent_reserve_horizon_fraction"),
            floor_seconds,
            row.get("activation_start_fraction"),
            row.get("activation_end_fraction"),
            row.get("max_onboard_capacity_fraction"),
        )
        if (
            match is None
            or row["name"] in physical
            or key in behavior_keys
            or row.get("algorithm") != match.group("algorithm")
            or row.get("training_seed") != int(match.group("seed"))
            or row.get("online_events") != 25_000
            or row.get("artifact_lineage") != "v17-tail-service-reserve"
            or row.get("parent_artifact_lineage") != "v16-headway-reserve"
            or row.get("parent_method")
            != (
                "h2oplus-v16-headway-reserve-three-quarter-"
                f"seed{row.get('training_seed')}-events000025000"
            )
            or row.get("parent_base_hold_scale") != PARENT_BASE_HOLD_SCALE
            or row.get("parent_headway_balance_gain") != PARENT_HEADWAY_BALANCE_GAIN
            or row.get("parent_reserve_horizon_fraction")
            != PARENT_RESERVE_HORIZON_FRACTION
            or row.get("tail_service_floor_seconds") != floor_seconds
            or row.get("activation_start_fraction") != ACTIVATION_START_FRACTION
            or row.get("activation_end_fraction") != ACTIVATION_END_FRACTION
            or row.get("max_onboard_capacity_fraction")
            != MAX_ONBOARD_CAPACITY_FRACTION
            or row.get("predesignated_candidate") is not (row["algorithm"] == CANDIDATE)
            or row.get("v17_protocol_sha256") != protocol_hash
        ):
            raise ValueError(f"malformed v17 method: {row!r}")
        checkpoint = Path(row["checkpoint"])
        if not checkpoint.is_absolute():
            checkpoint = resolved.parent / checkpoint
        checkpoint = checkpoint.resolve(strict=True)
        if sha256(checkpoint) != row.get("checkpoint_sha256"):
            raise ValueError(f"v17 parent checkpoint drifted: {row['name']}")
        physical[row["name"]] = row
        behavior_keys.add(key)
    alias_names = {row.get("name") for row in aliases}
    if alias_names != set(physical):
        raise ValueError("v17 aliases do not exactly cover physical behaviors")
    return resolved, payload, physical


def _scientific_sources_bound(source_hashes: dict, source_freeze: dict) -> bool:
    for source in source_freeze.get("matrix_sources", []):
        suffix = str(source.get("path", ""))
        expected = source.get("sha256")
        if not any(str(path).endswith(suffix) and digest == expected for path, digest in source_hashes.items()):
            return False
    return True


def _validate_result_policy(payload: dict, method: dict, protocol_hashes: dict[str, str]) -> None:
    policy = payload.get("policy", {})
    reserve = policy.get("tail_service_reserve", {})
    expected = {
        "artifact_lineage": "v17-tail-service-reserve",
        "parent_artifact_lineage": "v16-headway-reserve",
        "parent_method": method["parent_method"],
        "sha256": method["checkpoint_sha256"],
        "actor_tensor_sha256": method["actor_tensor_sha256"],
        "temperature_tensor_sha256": method["temperature_tensor_sha256"],
    }
    mismatches = {key: (policy.get(key), value) for key, value in expected.items() if policy.get(key) != value}
    variant = method["algorithm"][len("h2oplus-v17-tail-service-reserve-") :]
    if (
        mismatches
        or reserve.get("schema_version") != "transit-h2oplus-v17-tail-service-reserve-v1"
        or reserve.get("variant") != variant
        or reserve.get("tail_service_floor_seconds")
        != method["tail_service_floor_seconds"]
        or reserve.get("activation_start_fraction")
        != method["activation_start_fraction"]
        or reserve.get("activation_end_fraction")
        != method["activation_end_fraction"]
        or reserve.get("max_onboard_capacity_fraction")
        != method["max_onboard_capacity_fraction"]
        or reserve.get("requires_forward_bus_present") is not True
        or reserve.get("requires_backward_bus_absent") is not True
        or reserve.get("requires_positive_eligible_arrival_rate") is not True
        or reserve.get("parent_base_hold_scale")
        != method["parent_base_hold_scale"]
        or reserve.get("parent_headway_balance_gain")
        != method["parent_headway_balance_gain"]
        or reserve.get("parent_reserve_horizon_fraction")
        != method["parent_reserve_horizon_fraction"]
        or reserve.get("unit") != "physical_hold_seconds"
        or reserve.get("activation_budget_online_events") != 25_000
        or reserve.get("parent_action_preserved_outside_guard") is not True
        or reserve.get("guarded_hold_may_exceed_parent") is not True
        or reserve.get("physical_max_hold_preserved") is not True
        or reserve.get("predesignated_candidate") is not (method["algorithm"] == CANDIDATE)
        or reserve.get("checkpoint_selection_by_behavior") is not False
        or reserve.get("v17_protocol_sha256") != protocol_hashes["v17"]
        or reserve.get("formal_held_out_outcomes_read") is not False
    ):
        raise ValueError(f"v17 result tail-service provenance drifted: {mismatches}")
    parent = dict(method)
    parent.update(
        {
            "name": f"{PARENT}-seed{method['training_seed']}-events000025000",
            "algorithm": PARENT,
            "artifact_lineage": "v14-terminal-taper",
            "source_protocol_sha256": protocol_hashes["v14-terminal-taper"],
        }
    )
    v14._validate_training_metadata(payload, parent, protocol_hashes)


def _load_v17_rows(
    roots: list[Path],
    methods_path: Path,
    physical: dict[str, dict],
    protocol_hashes: dict[str, str],
    source_freeze: dict,
) -> tuple[list[dict], list[dict]]:
    if len(roots) != 3:
        raise ValueError("v17 analysis requires exactly three merged development blocks")
    rows, evidence, seen_signatures, blocks = [], [], set(), set()
    for raw_root in roots:
        root = raw_root.expanduser().resolve(strict=True)
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        source_hashes = manifest.get("source_sha256")
        block_names = {str(job.get("environment_block")) for job in jobs}
        if (
            manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != sha256(methods_path)
            or len(jobs) != 45
            or len(block_names) != 1
            or not isinstance(source_hashes, dict)
            or not _scientific_sources_bound(source_hashes, source_freeze)
        ):
            raise ValueError(f"v17 matrix block is incomplete or source-drifted: {root}")
        block = next(iter(block_names))
        if block not in BLOCKS or block in blocks:
            raise ValueError(f"unexpected or duplicate v17 block: {block}")
        blocks.add(block)
        evidence.append(
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
            if name not in physical:
                raise ValueError("v17 matrix contains an unregistered physical behavior")
            method = physical[name]
            output = _resolve_output(root, job["output"])
            payload = json.loads(output.read_text())
            demand_scale = float(job["scenario"]["demand_scale"])
            signature = job.get("signature")
            if (
                payload.get("schema_version") != "transit-evaluation-v7"
                or payload.get("evaluation_job", {}).get("signature") != signature
                or payload.get("method") != name
                or payload.get("source_sha256") != source_hashes
                or demand_scale not in SCALES
                or signature in seen_signatures
            ):
                raise ValueError(f"v17 physical result binding failed: {output}")
            seen_signatures.add(signature)
            _validate_result_policy(payload, method, protocol_hashes)
            coverage_errors = payload.get("passenger_welfare", {}).get("chronological_coverage_errors", {})
            if any(abs(float(value)) > 1e-9 for value in coverage_errors.values()):
                raise ValueError(f"v17 welfare ledger failed: {output}")
            rows.append(
                {
                    "algorithm": method["algorithm"],
                    "seed": method["training_seed"],
                    "events": 25_000,
                    "physical_method": name,
                    "actor_tensor_sha256": method["actor_tensor_sha256"],
                    "parent_base_hold_scale": method["parent_base_hold_scale"],
                    "parent_headway_balance_gain": method[
                        "parent_headway_balance_gain"
                    ],
                    "parent_reserve_horizon_fraction": method[
                        "parent_reserve_horizon_fraction"
                    ],
                    "tail_service_floor_seconds": method[
                        "tail_service_floor_seconds"
                    ],
                    "activation_start_fraction": method[
                        "activation_start_fraction"
                    ],
                    "activation_end_fraction": method["activation_end_fraction"],
                    "max_onboard_capacity_fraction": method[
                        "max_onboard_capacity_fraction"
                    ],
                    "block": block,
                    "scale": demand_scale,
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
    expected = {
        (algorithm, seed, block, scale)
        for algorithm in V17_ALGORITHMS
        for seed in SEEDS
        for block in BLOCKS
        for scale in SCALES
    }
    observed = {(row["algorithm"], row["seed"], row["block"], row["scale"]) for row in rows}
    if blocks != set(BLOCKS) or len(rows) != 135 or observed != expected:
        raise ValueError(f"v17 result grid is incomplete: {len(rows)}")
    return rows, evidence


def _v17_cell_means(v14_means: dict, v17_rows: list[dict]) -> dict:
    means = {
        key: value
        for key, value in v14_means.items()
        if key[0] in CONTROLS
    }
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in v17_rows:
        grouped.setdefault((row["algorithm"], row["seed"]), []).append(row)
    for algorithm in V17_ALGORITHMS:
        for seed in SEEDS:
            for events in BUDGETS[:-1]:
                means[(algorithm, seed, events)] = dict(v14_means[(PARENT, seed, events)])
            values = grouped.get((algorithm, seed), [])
            if len(values) != 9:
                raise ValueError(f"v17 policy lacks nine development cells: {algorithm}, {seed}")
            means[(algorithm, seed, 25_000)] = {
                "primary": float(np.mean([row["primary"] for row in values])),
                "completion": float(np.mean([row["completion"] for row in values])),
            }
    expected_cells = (len(CONTROLS) + len(V17_ALGORITHMS)) * len(SEEDS) * len(BUDGETS)
    if len(means) != expected_cells:
        raise ValueError(f"combined v17 logical grid drifted: {len(means)}")
    return means


def _aulcs(means: dict, algorithm: str, metric: str) -> np.ndarray:
    return np.asarray([v14._aulc(means, algorithm, seed, metric) for seed in SEEDS])


def _comparison(means: dict, candidate: str, other: str) -> dict:
    candidate_values = _aulcs(means, candidate, "primary")
    other_values = _aulcs(means, other, "primary")
    return {
        "candidate": candidate,
        "other": other,
        "candidate_mean_aulc": float(candidate_values.mean()),
        "other_mean_aulc": float(other_values.mean()),
        "relative_improvement": float((other_values.mean() - candidate_values.mean()) / other_values.mean()),
        "seed_wins": int(np.sum(candidate_values < other_values)),
        "mean_lower": bool(candidate_values.mean() < other_values.mean()),
        "per_seed": [
            {
                "seed": seed,
                "candidate_aulc": float(candidate_values[index]),
                "other_aulc": float(other_values[index]),
                "candidate_lower": bool(candidate_values[index] < other_values[index]),
            }
            for index, seed in enumerate(SEEDS)
        ],
    }


def _scale_aulc(v14_rows: list[dict], v17_rows: list[dict], algorithm: str, scale: float) -> float:
    points = []
    for events in BUDGETS:
        source = v14_rows if events < 25_000 else v17_rows
        source_algorithm = PARENT if events < 25_000 else algorithm
        values = [
            row["primary"]
            for row in source
            if row["algorithm"] == source_algorithm
            and row["events"] == events
            and math.isclose(row["scale"], scale)
        ]
        if len(values) != 15:
            raise ValueError("v17 demand-scale AULC coverage drifted")
        points.append(float(np.mean(values)))
    return float(np.trapz(points, np.asarray(BUDGETS, dtype=np.float64)) / 25_000.0)


def main() -> int:
    args = parse_args()
    registration_path, registration = _load_control(
        args.registration, EXPECTED_SHA256["registration"], REGISTRATION_SCHEMA
    )
    protocol_path, protocol = _load_control(
        args.protocol, EXPECTED_SHA256["protocol"], PROTOCOL_SCHEMA
    )
    source_freeze_path = args.source_freeze.expanduser().resolve(strict=True)
    source_freeze = json.loads(source_freeze_path.read_text())
    if source_freeze.get("schema_version") != SOURCE_FREEZE_SCHEMA:
        raise ValueError("v17 source freeze schema drifted")
    v16_analysis_path, v16_analysis = _load_control(
        args.v16_analysis,
        EXPECTED_SHA256["v16_analysis"],
        "transit-h2oplus-v16-headway-reserve-analysis-v1",
    )
    v16_closeout_path, v16_closeout = _load_control(
        args.v16_closeout,
        EXPECTED_SHA256["v16_closeout"],
        "transit-h2oplus-v16-headway-reserve-evaluation-closeout-v1",
    )
    v16_verification_path, v16_verification = _load_control(
        args.v16_verification,
        EXPECTED_SHA256["v16_verification"],
        "transit-h2oplus-v16-headway-reserve-evaluation-closeout-verification-v1",
    )
    v14_analysis_path, v14_analysis = _load_control(
        args.v14_analysis,
        EXPECTED_SHA256["v14_analysis"],
        "transit-h2oplus-v14-terminal-taper-analysis-v1",
    )
    v14_closeout_path, v14_closeout = _load_control(
        args.v14_closeout,
        EXPECTED_SHA256["v14_closeout"],
        "transit-h2oplus-v14-terminal-taper-evaluation-closeout-v1",
    )
    v14_verification_path, v14_verification = _load_control(
        args.v14_verification,
        EXPECTED_SHA256["v14_verification"],
        "transit-h2oplus-v14-terminal-taper-evaluation-closeout-verification-v1",
    )
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or registration.get("v17_development_evaluation_outcomes_read") is not False
        or registration.get(
            "prior_v14_v15_v16_development_outcomes_used_for_design"
        )
        is not True
        or registration.get("candidate", {}).get("algorithm") != CANDIDATE
        or registration.get("candidate", {}).get("tail_service_floor_seconds")
        != 30.0
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("v17_development_evaluation_outcomes_read") is not False
        or source_freeze.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("v17_evaluation_outcomes_read") is not False
        or source_freeze.get(
            "prior_v14_v15_v16_development_outcomes_used_for_design"
        )
        is not True
        or source_freeze.get("physical_behavior_count") != 15
        or source_freeze.get("expected_new_remote_sumo_episodes") != 135
        or source_freeze.get("expected_matrix_source_file_count") != 53
        or source_freeze.get("matrix_source_count") != 14
        or source_freeze.get("methods_manifest", {}).get("sha256") != EXPECTED_SHA256["methods"]
        or source_freeze.get("protocol", {}).get("sha256") != EXPECTED_SHA256["protocol"]
        or v16_analysis.get("artifact_integrity", {}).get("passed") is not True
        or v16_analysis.get("parent_training_scientific_stability", {}).get("passed") is not True
        or v16_analysis.get("formal_confirmation_eligible") is not False
        or v16_closeout.get("passed") is not True
        or v16_closeout.get("artifact_integrity", {}).get("passed") is not True
        or v16_verification.get("passed") is not True
        or v14_analysis.get("formal_confirmation_eligible") is not False
        or v14_closeout.get("passed") is not True
        or v14_verification.get("passed") is not True
        or any(
            payload.get("formal_held_out_outcomes_read") is not False
            for payload in (
                v16_analysis,
                v16_closeout,
                v16_verification,
                v14_analysis,
                v14_closeout,
                v14_verification,
            )
        )
    ):
        raise ValueError("v17 immutable analysis controls drifted")

    methods_path, methods_payload, physical = _load_methods(
        args.methods_manifest, sha256(protocol_path)
    )
    v14_methods_path, v14_methods_payload, v14_aliases = v14._load_methods(args.v14_methods)
    if sha256(v14_methods_path) != EXPECTED_SHA256["v14_methods"]:
        raise ValueError("v14 methods identity drifted")
    v14_roots = args.v14_matrix_root or [
        ROOT / "reproducibility/h2oplus_v14_terminal_taper_results/evaluation_canonical_v1" / block
        for block in BLOCKS
    ]
    v14_rows, v14_matrix_evidence = v14._load_rows(
        v14_roots, v14_methods_path, v14_methods_payload, v14_aliases
    )
    v14_means = v14._cell_means(v14_rows)
    protocol_hashes = {
        "v10-delayed": methods_payload["v10_protocol_sha256"],
        "v11-actor-hold": methods_payload["v11_protocol_sha256"],
        "v12-staged-release": methods_payload["v12_protocol_sha256"],
        "v13-proximal-release": methods_payload["v13_protocol_sha256"],
        "v14-terminal-taper": methods_payload["v14_protocol_sha256"],
        "v15-hold-calibration": methods_payload["v15_protocol_sha256"],
        "v16-headway-reserve": methods_payload["v16_protocol_sha256"],
        "v17": methods_payload["v17_protocol_sha256"],
    }
    v17_rows, v17_matrix_evidence = _load_v17_rows(
        args.v17_matrix_root,
        methods_path,
        physical,
        protocol_hashes,
        source_freeze,
    )
    means = _v17_cell_means(v14_means, v17_rows)
    thresholds = protocol["analysis_gates"]
    comparisons = {control: _comparison(means, CANDIDATE, control) for control in CONTROLS}
    diagnostic_comparisons = {
        variant: {control: _comparison(means, variant, control) for control in CONTROLS}
        for variant in DIAGNOSTICS
    }
    all_algorithms = (*CONTROLS, *V17_ALGORITHMS)
    budget_means = {
        algorithm: float(np.mean([means[(algorithm, seed, 25_000)]["primary"] for seed in SEEDS]))
        for algorithm in all_algorithms
    }
    candidate_completion = float(_aulcs(means, CANDIDATE, "completion").mean())
    completion = {
        control: {
            "candidate_aulc": candidate_completion,
            "control_aulc": float(_aulcs(means, control, "completion").mean()),
            "delta": candidate_completion - float(_aulcs(means, control, "completion").mean()),
        }
        for control in CONTROLS
    }
    scale_rows = []
    for scale in SCALES:
        candidate_aulc = _scale_aulc(v14_rows, v17_rows, CANDIDATE, scale)
        sac_aulc = v14._scale_aulc(v14_rows, "sac-online", scale)
        scale_rows.append(
            {
                "demand_scale": scale,
                "candidate_aulc": candidate_aulc,
                "sac_online_aulc": sac_aulc,
                "relative_degradation": (candidate_aulc - sac_aulc) / sac_aulc,
            }
        )

    relative_thresholds = {
        "h2oplus": "candidate_relative_aulc_improvement_over_fixed_h2o_at_least",
        "sac-online": "candidate_relative_aulc_improvement_over_sac_at_least",
        "h2oplus-v10-delayed-actor": "candidate_relative_aulc_improvement_over_v10_delayed_actor_at_least",
        "h2oplus-v11-actor-hold": "candidate_relative_aulc_improvement_over_v11_at_least",
        "h2oplus-v12-actor-ramp": "candidate_relative_aulc_improvement_over_v12_actor_ramp_at_least",
        "h2oplus-v13-proximal-strong": "candidate_relative_aulc_improvement_over_v13_strong_at_least",
    }
    win_thresholds = {
        "h2oplus": "candidate_seed_wins_over_fixed_h2o_at_least",
        "sac-online": "candidate_seed_wins_over_sac_at_least",
        "h2oplus-v10-delayed-actor": "candidate_seed_wins_over_v10_delayed_actor_at_least",
        "h2oplus-v11-actor-hold": "candidate_seed_wins_over_v11_at_least",
        "h2oplus-v12-actor-ramp": "candidate_seed_wins_over_v12_actor_ramp_at_least",
        "h2oplus-v13-proximal-strong": "candidate_seed_wins_over_v13_strong_at_least",
    }
    checks = {
        "sealed_v16_parent_artifact_integrity": v16_closeout.get("artifact_integrity", {}).get("passed") is True,
        "sealed_v14_parent_training_stability": v14_closeout.get("training_scientific_stability", {}).get("passed") is True,
        "candidate_mean_lower_than_all_controls_at_25000": all(
            budget_means[CANDIDATE] < budget_means[control] for control in CONTROLS
        ),
        "relative_aulc_improvement_over_all_controls": all(
            comparisons[control]["relative_improvement"] >= float(thresholds[key])
            for control, key in relative_thresholds.items()
        ),
        "seed_wins_over_all_controls": all(
            comparisons[control]["seed_wins"] >= int(thresholds[key])
            for control, key in win_thresholds.items()
        ),
        "completion_rate_noninferiority_against_all_controls": all(
            item["delta"] >= -float(thresholds["completion_rate_noninferiority_margin"])
            for item in completion.values()
        ),
        "demand_scale_robustness_against_sac": all(
            item["relative_degradation"]
            <= float(thresholds["no_demand_scale_relative_aulc_degradation_over_sac_greater_than"])
            for item in scale_rows
        ),
    }
    gate_passed = all(checks.values())
    report = {
        "schema_version": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_integrity": {
            "passed": True,
            "new_v17_physical_results": len(v17_rows),
            "reused_frozen_v14_physical_results": 630,
            "v17_physical_behaviors": 15,
            "v17_matrix_blocks": len(v17_matrix_evidence),
            "v14_matrix_blocks": len(v14_matrix_evidence),
        },
        "parent_training_scientific_stability": {
            "passed": checks["sealed_v14_parent_training_stability"],
            "new_training_runs": 0,
            "parent": "sealed v14 terminal-quarter 25k actors",
        },
        "development_scientific_result": {
            "screening_gate_passed": gate_passed,
            "predesignated_candidate": CANDIDATE,
            "checks": checks,
            "mean_primary_at_25000": budget_means,
            "control_comparisons": comparisons,
            "completion_aulc_against_controls": completion,
            "demand_scale_aulc_against_sac": scale_rows,
            "all_algorithm_mean_aulc": {
                algorithm: float(_aulcs(means, algorithm, "primary").mean())
                for algorithm in all_algorithms
            },
        },
        "diagnostic_results_not_used_as_selection_gates": {
            "algorithms": list(DIAGNOSTICS),
            "control_comparisons": diagnostic_comparisons,
        },
        "design_reuse_disclosure": {
            "prior_v14_v15_v16_development_outcomes_used_for_v17_design": True,
            "v17_outcomes_used_before_freeze": False,
            "independent_confirmation": False,
        },
        "evidence": {
            "registration": {"path": str(registration_path), "sha256": sha256(registration_path)},
            "protocol": {"path": str(protocol_path), "sha256": sha256(protocol_path)},
            "methods": {"path": str(methods_path), "sha256": sha256(methods_path)},
            "source_freeze": {"path": str(source_freeze_path), "sha256": sha256(source_freeze_path)},
            "v16_analysis": {"path": str(v16_analysis_path), "sha256": sha256(v16_analysis_path)},
            "v16_closeout": {"path": str(v16_closeout_path), "sha256": sha256(v16_closeout_path)},
            "v16_verification": {"path": str(v16_verification_path), "sha256": sha256(v16_verification_path)},
            "v14_analysis": {"path": str(v14_analysis_path), "sha256": sha256(v14_analysis_path)},
            "v14_closeout": {"path": str(v14_closeout_path), "sha256": sha256(v14_closeout_path)},
            "v14_verification": {"path": str(v14_verification_path), "sha256": sha256(v14_verification_path)},
            "v17_matrix": v17_matrix_evidence,
        },
        "formal_confirmation_eligible": gate_passed,
        "formal_confirmation_eligibility_does_not_authorize_submission": True,
        "formal_confirmation_stage_authorized_or_submitted": False,
        "formal_held_out_outcomes_read": False,
        "supports_formal_held_out_model_effect_claim": False,
    }
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v17 analysis: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["development_scientific_result"], indent=2, sort_keys=True))
    return 0 if gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
