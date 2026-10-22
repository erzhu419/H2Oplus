#!/usr/bin/env python3
"""Build a no-SUMO audit for the v19 completion-safety design."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import (  # noqa: E402
    DEFAULT_CONFIG_V7,
    LOCAL_EVENT_STATE_INDEX,
    RAW_PREFIX_NAMES,
    RAW_SLOT_WIDTH,
    RAW_STATE_INDEX,
    policy_action_to_hold_seconds_v7,
    raw_state_to_canonical_v7,
)
from bus_h2o.sumo_provenance import sha256  # noqa: E402
from SimpleSAC import evaluate_transit_v17_tail_service_reserve as v17  # noqa: E402
from SimpleSAC.transit_v7.trainer import load_actor_checkpoint_v7  # noqa: E402


REPRO = ROOT / "reproducibility"
DATASET = ROOT / "bus_h2o/datasets_v7/formal_v2_full/transit_smdp_v7_formal_v2.h5"
METHODS = REPRO / "h2oplus_v18_empty_tail_reserve_evaluation_methods_v1.json"
EVALUATOR = ROOT / "SimpleSAC/evaluate_transit_v19_completion_safety_reserve.py"
MATRIX = REPRO / "run_evaluation_matrix_v19_completion_safety.py"
TEST_SOURCE = REPRO / "test_h2oplus_v19_completion_safety.py"
V1_AUDIT = REPRO / "h2oplus_v19_completion_safety_design_audit_v1.json"
DEFAULT_OUTPUT = REPRO / "h2oplus_v19_completion_safety_design_audit_v2.json"
SAMPLE_SEED = 20_260_825
SAMPLE_ROWS = 30_000

FROZEN_INPUTS = {
    "v18_methods": (
        METHODS,
        "f87fec33f993e12fa49edad4264e34acce7ffde5dfa55c571197d424039c24e4",
    ),
    "v11_protocol": (
        REPRO / "h2oplus_v11_actor_hold_protocol_v2.json",
        "c5b162541c6ce2476bb0f57f6dc936de13b85b5e1e6ede77b533a5cf9a4dfced",
    ),
    "v11_analysis": (
        REPRO / "h2oplus_v11_actor_hold_results/actor_hold_analysis_v1.json",
        "e3f6df3d075774dec3e662cbf2b6ef848e5f361f57be45fff7cf159768fa04e1",
    ),
    "v14_analysis": (
        REPRO / "h2oplus_v14_terminal_taper_results/terminal_taper_analysis_v1.json",
        "937a5300988311f80a84dc1220dfee3f0f96f2b6f83691d4e309c38d6bbe6856",
    ),
    "v15_analysis": (
        REPRO / "h2oplus_v15_hold_calibration_results/hold_calibration_analysis_v1.json",
        "d122aa79f6ee2cd5010e241607347921ed9bdc80bbed6ba41c3105e601ac9d9c",
    ),
    "v16_analysis": (
        REPRO / "h2oplus_v16_headway_reserve_results/headway_reserve_analysis_v1.json",
        "4a88f48946c2ec478ca47274df39e0ccdf3535c1c21ad41141142d529617f75e",
    ),
    "v17_analysis": (
        REPRO / "h2oplus_v17_tail_service_reserve_results/tail_service_reserve_analysis_v2.json",
        "84e80b7f08d16f6e356f09c56ff75d1bbdf322578641df4ede9e73f8d4ff4f78",
    ),
    "v18_analysis": (
        REPRO / "h2oplus_v18_empty_tail_reserve_results/empty_tail_reserve_analysis_v3.json",
        "0fa89711f239b4eb798ad34d10dea79e00fd77a3274314a73d7d2e2bff745287",
    ),
    "data_environment_audit": (
        REPRO / "h2oplus_v17_data_environment_convergence_audit_v1.json",
        "ff5210e82913c456314ba1e47a1f3400e0e5bdb58fce2f14d8e9bc1cdd0c95e9",
    ),
    "data_environment_verification": (
        REPRO / "h2oplus_v17_data_environment_convergence_verification_v1.json",
        "925fa9ace1af907bd50042718ba9d7492a892a8e8a2a863b567091540af62390",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    return {
        "path": str(resolved.relative_to(ROOT)),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def load_frozen() -> tuple[dict[str, dict], dict[str, dict]]:
    payloads = {}
    bindings = {}
    for name, (path, expected) in FROZEN_INPUTS.items():
        actual = sha256(path.resolve(strict=True))
        if actual != expected:
            raise ValueError(f"frozen input drifted: {name}: {actual}")
        payload = json.loads(path.read_text())
        if payload.get("formal_held_out_outcomes_read") not in (None, False):
            raise ValueError(f"formal outcome boundary drifted: {name}")
        payloads[name] = payload
        bindings[name] = descriptor(path)
    return payloads, bindings


def scientific_result(payload: dict) -> dict:
    return payload.get("development_scientific_result", payload)


def paired_response(
    root: Path,
    baseline: str,
    variants: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    by_cell: dict[tuple, dict[str, np.ndarray]] = {}
    for path in root.glob("**/sumo_seed*.json"):
        payload = json.loads(path.read_text())
        algorithm = payload["method"].split("-seed")[0]
        if algorithm not in (baseline, *variants):
            continue
        training = payload["policy"]["training_metadata"]
        scenario = payload["scenario"]
        key = (
            int(training["training_seed"]),
            int(payload["sumo_seed"]),
            int(scenario["selection_seed"]),
            float(scenario["demand_scale"]),
        )
        by_cell.setdefault(key, {})[algorithm] = np.asarray(
            (
                payload["operations"]["hold_seconds"]["mean"],
                payload["passengers"]["completion_rate"],
                payload["passenger_welfare"][
                    "accrued_generalized_seconds_per_departed_passenger"
                ],
                payload["passengers"]["unfinished_passengers"],
            ),
            dtype=np.float64,
        )
    result = {}
    for variant in variants:
        rows = [
            methods[variant] - methods[baseline]
            for methods in by_cell.values()
            if baseline in methods and variant in methods
        ]
        values = np.asarray(rows, dtype=np.float64)
        if values.shape != (45, 4) or np.any(values[:, 0] <= 0.0):
            raise ValueError(f"paired development evidence incomplete: {variant}")
        hold = values[:, 0]
        result[variant] = {
            "matched_cells": len(values),
            "mean_added_hold_seconds_per_decision": float(np.mean(hold)),
            "mean_completion_gain": float(np.mean(values[:, 1])),
            "mean_primary_change_seconds": float(np.mean(values[:, 2])),
            "mean_unfinished_passenger_change": float(np.mean(values[:, 3])),
            "ols_completion_gain_per_added_hold_second": float(
                np.sum(hold * values[:, 1]) / np.sum(hold**2)
            ),
            "ols_primary_change_per_added_hold_second": float(
                np.sum(hold * values[:, 2]) / np.sum(hold**2)
            ),
            "positive_completion_cells": int(np.sum(values[:, 1] > 0.0)),
        }
    return result


def dataset_audit() -> tuple[dict, np.ndarray, np.ndarray]:
    with h5py.File(DATASET, "r") as handle:
        rows = len(handle["raw_observations"])
        rng = np.random.default_rng(SAMPLE_SEED)
        sample_indices = np.sort(rng.choice(rows, size=SAMPLE_ROWS, replace=False))
        raw = handle["raw_observations"][sample_indices]
        station_max = np.asarray(
            json.loads(handle.attrs["station_max_by_line"]), dtype=np.float32
        )
        episode_ids = handle["episode_id"][:]
        terminal = handle["terminals"][:].astype(bool)
        cells = set(
            zip(
                handle["sumo_seed"][:].tolist(),
                handle["demand_realization_seed"][:].tolist(),
                handle["demand_scale"][:].tolist(),
            )
        )
        terminal_rows = np.flatnonzero(terminal)
        source_ids = handle["source_shard_id"][terminal_rows]
        terminal_raw = handle["raw_next_observations"][terminal_rows]
        source_shards = json.loads(handle.attrs["source_shards"])

    exact_differences = []
    for source_id, state in zip(source_ids, terminal_raw):
        source = Path(source_shards[int(source_id)]["path"])
        summary = json.loads(Path(str(source) + ".json").read_text())
        exact_unfinished = float(summary["passengers"]["unfinished_passengers"])
        packed_active = float(
            state[RAW_STATE_INDEX["system_waiting_passengers"]]
            + state[RAW_STATE_INDEX["system_in_vehicle_passengers"]]
        )
        exact_differences.append(exact_unfinished - packed_active)
    difference = np.asarray(exact_differences, dtype=np.float64)
    audit = {
        "descriptor": descriptor(DATASET),
        "rows": rows,
        "episodes": int(len(np.unique(episode_ids))),
        "terminal_rows": int(np.sum(terminal)),
        "independent_environment_cells": len(cells),
        "mean_rows_per_episode": float(rows / len(np.unique(episode_ids))),
        "endpoint_label_gap": {
            "meaning": (
                "exact unfinished passengers minus packed terminal waiting and "
                "in-vehicle counts; the difference is other active journey state"
            ),
            "mean_passengers": float(np.mean(difference)),
            "max_passengers": float(np.max(difference)),
            "gate_equivalent_passengers_at_mean_departures": 45.25,
        },
    }
    return audit, raw, station_max


def offline_reach(raw: np.ndarray, station_max: np.ndarray) -> dict:
    canonical = raw_state_to_canonical_v7(raw, station_max, DEFAULT_CONFIG_V7)
    current = np.rint(raw[:, RAW_STATE_INDEX["current_decision_index"]]).astype(int)
    start = len(RAW_PREFIX_NAMES) + current * RAW_SLOT_WIDTH
    index = np.arange(len(raw))

    def local(name: str) -> np.ndarray:
        return raw[index, start + 1 + LOCAL_EVENT_STATE_INDEX[name]]

    capacity = local("bus_capacity")
    onboard = local("onboard_passengers")
    arrival_rate = local("eligible_arrival_rate_per_second")
    forward_present = np.rint(local("forward_bus_present")).astype(int)
    backward_present = np.rint(local("backward_bus_present")).astype(int)
    progress = np.clip(
        raw[:, RAW_STATE_INDEX["decision_time_seconds"]]
        / DEFAULT_CONFIG_V7.horizon_seconds,
        0.0,
        1.0,
    )
    guard = (
        (progress >= 0.15)
        & (progress < 0.75)
        & (forward_present == 1)
        & (backward_present == 0)
        & (arrival_rate > 0.0)
        & (capacity > 0.0)
        & (onboard / capacity <= 0.25)
    )

    methods = json.loads(METHODS.read_text())["methods"]
    methods = [
        method
        for method in methods
        if method["algorithm"] == "h2oplus-v18-empty-tail-reserve-baseline"
    ]
    if len(methods) != 5:
        raise ValueError("v19 reach audit requires five sealed parent actors")
    rows = []
    for method in methods:
        checkpoint = (REPRO / method["checkpoint"]).resolve(strict=True)
        if sha256(checkpoint) != method["checkpoint_sha256"]:
            raise ValueError("sealed v14 parent checkpoint drifted")
        actor, _ = load_actor_checkpoint_v7(
            checkpoint, "cpu", require_policy_artifact=True
        )
        actions = []
        with torch.inference_mode():
            for batch in np.array_split(canonical, 30):
                action, _ = actor(
                    torch.as_tensor(batch, dtype=torch.float32),
                    deterministic=True,
                )
                actions.append(action.cpu().numpy().reshape(-1))
        action_values = np.concatenate(actions)
        parent_hold = np.asarray(
            [
                policy_action_to_hold_seconds_v7(
                    float(
                        v17.tail_service_hold_action(
                            state,
                            np.asarray([action], dtype=np.float32),
                            20.0,
                            DEFAULT_CONFIG_V7,
                        )[0]
                    ),
                    DEFAULT_CONFIG_V7,
                )
                for state, action in zip(raw, action_values)
            ],
            dtype=np.float64,
        )
        candidate_hold = np.where(guard, np.maximum(parent_hold, 60.0), parent_hold)
        diagnostic_hold = np.where(guard, np.maximum(parent_hold, 50.0), parent_hold)
        rows.append(
            {
                "training_seed": method["training_seed"],
                "guard_fraction": float(np.mean(guard)),
                "parent_mean_hold_seconds": float(np.mean(parent_hold)),
                "candidate_mean_hold_seconds": float(np.mean(candidate_hold)),
                "candidate_mean_added_hold_seconds": float(
                    np.mean(candidate_hold - parent_hold)
                ),
                "diagnostic_mean_added_hold_seconds": float(
                    np.mean(diagnostic_hold - parent_hold)
                ),
                "candidate_increased_decision_fraction": float(
                    np.mean(candidate_hold > parent_hold + 1e-7)
                ),
            }
        )
    fields = (
        "guard_fraction",
        "parent_mean_hold_seconds",
        "candidate_mean_hold_seconds",
        "candidate_mean_added_hold_seconds",
        "diagnostic_mean_added_hold_seconds",
        "candidate_increased_decision_fraction",
    )
    return {
        "sample_seed": SAMPLE_SEED,
        "sample_rows": SAMPLE_ROWS,
        "per_parent_actor": rows,
        "five_actor_means": {
            field: float(np.mean([row[field] for row in rows])) for field in fields
        },
    }


def build_payload() -> dict:
    frozen, bindings = load_frozen()
    v11 = scientific_result(frozen["v11_analysis"])
    v14 = scientific_result(frozen["v14_analysis"])
    v15 = scientific_result(frozen["v15_analysis"])
    v16 = scientific_result(frozen["v16_analysis"])
    v17_result = scientific_result(frozen["v17_analysis"])
    v18 = scientific_result(frozen["v18_analysis"])
    data_risks = {row["id"]: row for row in frozen["data_environment_audit"]["risk_register"]}

    responses = {
        "v15_global_scale": paired_response(
            REPRO / "h2oplus_v15_hold_calibration_results/evaluation_partition_v1",
            "h2oplus-v15-hold-eighth",
            ("h2oplus-v15-hold-quarter", "h2oplus-v15-hold-three-eighths"),
        ),
        "v17_targeted_floor": paired_response(
            REPRO / "h2oplus_v17_tail_service_reserve_results/evaluation_partition_v2",
            "h2oplus-v17-tail-service-reserve-twenty",
            (
                "h2oplus-v17-tail-service-reserve-thirty",
                "h2oplus-v17-tail-service-reserve-forty",
            ),
        ),
        "v18_targeted_occupancy": paired_response(
            REPRO / "h2oplus_v18_empty_tail_reserve_results/evaluation_partition_v3",
            "h2oplus-v18-empty-tail-reserve-baseline",
            (
                "h2oplus-v18-empty-tail-reserve-empty-forty",
                "h2oplus-v18-empty-tail-reserve-one-seat-forty",
            ),
        ),
    }
    dataset, raw, station_max = dataset_audit()
    reach = offline_reach(raw, station_max)
    added = reach["five_actor_means"]["candidate_mean_added_hold_seconds"]
    global_slopes = [
        row["ols_completion_gain_per_added_hold_second"]
        for row in responses["v15_global_scale"].values()
    ]
    targeted_slopes = [
        row["ols_completion_gain_per_added_hold_second"]
        for group in ("v17_targeted_floor", "v18_targeted_occupancy")
        for row in responses[group].values()
    ]
    conservative_projection = min(global_slopes) * added
    targeted_projection = min(targeted_slopes) * added
    if not conservative_projection < 0.003 < targeted_projection:
        raise ValueError("v19 reach uncertainty no longer brackets the paired gate")

    v11_margin = frozen["v11_protocol"]["analysis_gates"][
        "completion_rate_noninferiority_margin"
    ]
    current_margin = 0.003
    v11_completion = v11["completion_aulc"]
    why = {
        "verified_model_context": "the earlier v11 development was produced in a GPT-5.5 session",
        "v11_training_intervention": (
            "freeze actor and temperature through 10k while critic, value and "
            "ratio updates continue, then use online-only states for actor updates"
        ),
        "v11_primary_relative_improvement": {
            name: row["relative_improvement"]
            for name, row in v11["comparisons"].items()
        },
        "v11_completion_margin": v11_margin,
        "current_completion_margin": current_margin,
        "v11_completion_deltas": {
            name: row["delta"] for name, row in v11_completion.items()
        },
        "v11_would_pass_current_completion_margin": all(
            row["delta"] >= -current_margin for row in v11_completion.values()
        ),
        "later_primary_endpoint_examples": {
            "v11": v11["budget_results"]["25000"]["means"][
                "h2oplus-v11-actor-hold"
            ],
            "v15_eighth": v15["mean_primary_at_25000"][
                "h2oplus-v15-hold-eighth"
            ],
            "v16_three_quarter": v16["mean_primary_at_25000"][
                "h2oplus-v16-headway-reserve-three-quarter"
            ],
            "v17_twenty": v17_result["mean_primary_at_25000"][
                "h2oplus-v17-tail-service-reserve-twenty"
            ],
            "v18_one_seat": v18["algorithm_means"][
                "h2oplus-v18-empty-tail-reserve-one-seat-forty"
            ]["primary"],
        },
        "conclusion": (
            "GPT-5.5 found a real training-trajectory repair for the welfare "
            "objective, but its v11 completion result relied on a 0.005 margin "
            "and would fail the current 0.003 margin. Later policies are better "
            "on primary welfare; the unresolved defect is completion safety, not "
            "a general loss of optimizer competence."
        ),
    }
    if why["v11_would_pass_current_completion_margin"] is not False:
        raise ValueError("v11/current gate comparison drifted")

    return {
        "schema_version": "transit-h2oplus-v19-completion-safety-design-audit-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development-design",
        "formal_held_out_outcomes_read": False,
        "local_sumo_executed": False,
        "local_training_executed": False,
        "prior_v14_v15_v16_v17_v18_development_outcomes_used_for_design": True,
        "frozen_inputs": bindings,
        "evaluator": descriptor(EVALUATOR),
        "matrix": descriptor(MATRIX),
        "superseded_design_audit": descriptor(V1_AUDIT),
        "binding_correction": (
            "v2 adds the v18 methods manifest and v19 matrix source to explicit "
            "file-level bindings before any v19 evaluation outcome"
        ),
        "implementation_verification": {
            "test_source": descriptor(TEST_SOURCE),
            "runner": "python3 -m pytest -q -s",
            "tests_passed": 5,
            "tests_failed": 0,
            "sumo_import_or_execution_required": False,
            "training_required": False,
        },
        "why_gpt55_looked_better": why,
        "current_problem": {
            "primary": "objective-constraint mismatch",
            "secondary": [
                "millions of transitions collapse to 30 independent exogenous cells",
                "exact unfinished count includes journey states absent from the packed terminal count",
                "positive large-hold action support is sparse",
                "training SUMO 1.27.1 differs from development SUMO 1.25.0",
                "v14-v18 outcomes adaptively informed successive designs",
            ],
            "v14_completion_delta_vs_h2oplus": v14[
                "completion_aulc_against_controls"
            ]["h2oplus"]["delta"],
            "v17_completion_delta_vs_h2oplus": v17_result[
                "completion_aulc_against_controls"
            ]["h2oplus"]["delta"],
            "v18_candidate_completion_gain": v18["candidate_paired_mean"][
                "completion_gain"
            ],
            "v18_diagnostic_completion_gain": v18["diagnostic_paired_mean"][
                "completion_gain"
            ],
            "existing_risk_ids": sorted(data_risks),
        },
        "dataset_audit": dataset,
        "observed_paired_response": responses,
        "offline_mechanism_reach": reach,
        "fixed_successor_design": {
            "parent": "sealed v17 twenty-second 25k behavior",
            "paired_baseline": "baseline",
            "predesignated_candidate": "quarter-load-sixty",
            "diagnostic": "quarter-load-fifty",
            "activation_fraction": [0.15, 0.75],
            "max_onboard_capacity_fraction": 0.25,
            "requires_forward_bus_present": True,
            "requires_backward_bus_absent": True,
            "requires_positive_eligible_arrival_rate": True,
            "candidate_floor_seconds": 60.0,
            "diagnostic_floor_seconds": 50.0,
            "paired_completion_gain_gate": 0.003,
            "paired_primary_increase_gate_seconds": 3.0,
            "checkpoint_selection_by_behavior": False,
            "new_training_runs": 0,
            "fresh_development_cells_required": True,
        },
        "reach_uncertainty": {
            "mean_added_hold_seconds_per_decision": added,
            "conservative_global_response_projection": conservative_projection,
            "targeted_response_projection": targeted_projection,
            "paired_completion_gain_gate": 0.003,
            "interpretation": (
                "mechanism reach is sufficient only under the observed targeted "
                "response regime; it is insufficient under the global-scale "
                "response regime, so a fresh paired development screen is "
                "necessary and success is not assumed"
            ),
        },
        "claim_boundary": {
            "core_implementation_verified_without_sumo": True,
            "development_result_available": False,
            "formal_confirmation_eligible": False,
            "formal_confirmation_authorized": False,
            "formal_effect_claim_supported": False,
        },
    }


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite v19 design evidence: {output}")
    payload = build_payload()
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_bytes(encoded)
    os.replace(partial, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "formal_held_out_outcomes_read": False,
                "local_sumo_executed": False,
                "local_training_executed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
