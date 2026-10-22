#!/usr/bin/env python3
"""Evaluate a frozen v17 parent with a bounded completion-safety reserve."""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import (  # noqa: E402
    RAW_EVENT_STATE_INDEX,
    RAW_STATE_INDEX,
    current_event_v7,
    hold_seconds_to_policy_action_v7,
    policy_action_to_hold_seconds_v7,
    raw_state_to_canonical_v7,
)
from bus_h2o.sumo_provenance import sha256  # noqa: E402
from SimpleSAC import evaluate_transit_v17_tail_service_reserve as v17  # noqa: E402
from SimpleSAC.transit_v7.trainer import (  # noqa: E402
    POLICY_ARTIFACT_SCHEMA,
    load_actor_checkpoint_v7,
)


V19_PROTOCOL_SHA_ENV = "H2OPLUS_V19_COMPLETION_SAFETY_PROTOCOL_SHA256"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus-v19-completion-safety-"
    r"(?P<variant>baseline|quarter-load-fifty|quarter-load-sixty))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
VARIANT_CONFIG = {
    "baseline": {
        "enabled": False,
        "floor_seconds": 20.0,
        "max_onboard_fraction": 0.25,
    },
    "quarter-load-fifty": {
        "enabled": True,
        "floor_seconds": 50.0,
        "max_onboard_fraction": 0.25,
    },
    "quarter-load-sixty": {
        "enabled": True,
        "floor_seconds": 60.0,
        "max_onboard_fraction": 0.25,
    },
}
PARENT_TAIL_SERVICE_FLOOR_SECONDS = 20.0
ACTIVATION_START_FRACTION = 0.15
ACTIVATION_END_FRACTION = 0.75


def completion_safety_action(
    raw_state: np.ndarray,
    policy_action: np.ndarray | float,
    variant: str,
    config,
) -> np.ndarray:
    """Restore service only for registered low-load tail-bus states."""

    registered = VARIANT_CONFIG.get(variant)
    if registered is None:
        raise ValueError(f"unregistered v19 completion-safety variant: {variant!r}")
    action = np.asarray(policy_action, dtype=np.float32)
    if action.shape not in ((), (1,)) or not np.isfinite(action).all():
        raise ValueError("v19 policy action must be one finite scalar")
    parent_action = v17.tail_service_hold_action(
        raw_state,
        action,
        PARENT_TAIL_SERVICE_FLOOR_SECONDS,
        config,
    )
    if not registered["enabled"]:
        return parent_action.copy()

    parent_hold = float(policy_action_to_hold_seconds_v7(parent_action[0], config))
    state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    horizon_progress = float(
        np.clip(
            state[RAW_STATE_INDEX["decision_time_seconds"]]
            / config.horizon_seconds,
            0.0,
            1.0,
        )
    )
    event = current_event_v7(raw_state)
    capacity = float(event[RAW_EVENT_STATE_INDEX["bus_capacity"]])
    onboard = float(event[RAW_EVENT_STATE_INDEX["onboard_passengers"]])
    arrival_rate = float(
        event[RAW_EVENT_STATE_INDEX["eligible_arrival_rate_per_second"]]
    )
    forward_present = int(
        round(float(event[RAW_EVENT_STATE_INDEX["forward_bus_present"]]))
    )
    backward_present = int(
        round(float(event[RAW_EVENT_STATE_INDEX["backward_bus_present"]]))
    )
    if not all(math.isfinite(value) for value in (capacity, onboard, arrival_rate)):
        raise ValueError("v19 completion-safety guard received nonfinite state")

    low_load = (
        capacity > 0.0
        and onboard / capacity <= float(registered["max_onboard_fraction"])
    )
    guard_active = (
        ACTIVATION_START_FRACTION <= horizon_progress < ACTIVATION_END_FRACTION
        and forward_present == 1
        and backward_present == 0
        and arrival_rate > 0.0
        and low_load
    )
    calibrated_hold = parent_hold
    if guard_active:
        calibrated_hold = max(parent_hold, float(registered["floor_seconds"]))
    calibrated_hold = min(float(config.max_hold_seconds), calibrated_hold)
    if not 0.0 <= calibrated_hold <= float(config.max_hold_seconds) + 1e-9:
        raise ValueError("v19 transformed hold escaped the physical action bound")
    calibrated_action = hold_seconds_to_policy_action_v7(calibrated_hold, config)
    return np.asarray([calibrated_action], dtype=np.float32)


def evaluation_source_paths(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v17.evaluation_source_paths(sumo_dir, edge_xml))
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "reproducibility/run_evaluation_matrix_v19_completion_safety.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def build_development_policy(args, config, station_max, dataset_hash):
    match = METHOD_NAME.fullmatch(args.method_name)
    if match is None:
        return v17.build_development_policy(args, config, station_max, dataset_hash)
    algorithm = match.group("algorithm")
    variant = match.group("variant")
    seed = int(match.group("seed"))
    events = int(match.group("events"))
    registered = VARIANT_CONFIG[variant]
    parent_method = (
        f"h2oplus-v17-tail-service-reserve-twenty-seed{seed}-events000025000"
    )
    protocol_hash = os.environ.get(V19_PROTOCOL_SHA_ENV, "")
    if (
        events != 25_000
        or args.checkpoint is None
        or args.expected_algorithm != algorithm
        or args.expected_training_seed != seed
        or args.expected_v19_parent_method_name != parent_method
        or not math.isclose(
            float(args.expected_v19_parent_floor_seconds),
            PARENT_TAIL_SERVICE_FLOOR_SECONDS,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or bool(args.expected_v19_reserve_enabled) is not registered["enabled"]
        or not math.isclose(
            float(args.expected_v19_reserve_floor_seconds),
            float(registered["floor_seconds"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(args.expected_v19_activation_start_fraction),
            ACTIVATION_START_FRACTION,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(args.expected_v19_activation_end_fraction),
            ACTIVATION_END_FRACTION,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(args.expected_v19_max_onboard_fraction),
            float(registered["max_onboard_fraction"]),
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not SHA256.fullmatch(protocol_hash)
    ):
        raise ValueError("v19 matrix identity or protocol binding drifted")

    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    actor, extra = load_actor_checkpoint_v7(
        checkpoint, args.device, require_policy_artifact=True
    )
    common = {
        "dataset_sha256": dataset_hash,
        "contract_config": config.to_dict(),
        "mdp_version": "v7",
        "policy_artifact_schema": POLICY_ARTIFACT_SCHEMA,
        "training_seed": seed,
        "run_signature": args.expected_training_run_signature,
        "training_checkpoint_sha256": args.expected_training_checkpoint_sha256,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {
        key: {"artifact": extra.get(key), "expected": value}
        for key, value in common.items()
        if extra.get(key) != value
    }
    if mismatches:
        raise ValueError(f"v19 parent common provenance mismatch: {mismatches}")
    v17.v16.v15.v14._validate_v14(extra, "terminal-quarter", events, args)
    if (
        v17.v16.v15.v14.v13.v12.v11.actor_digest(actor.state_dict())
        != args.expected_actor_tensor_sha256
    ):
        raise ValueError("v19 parent actor tensor differs from methods manifest")

    def checkpoint_policy(raw_state: np.ndarray) -> np.ndarray:
        canonical = raw_state_to_canonical_v7(raw_state, station_max, config)
        with torch.inference_mode():
            tensor = torch.as_tensor(
                canonical[None, :], dtype=torch.float32, device=args.device
            )
            action, _ = actor(tensor, deterministic=True)
        return completion_safety_action(
            raw_state,
            action.cpu().numpy()[0],
            variant,
            config,
        )

    return checkpoint_policy, {
        "type": "development-v19-completion-safety-policy",
        "artifact_schema": POLICY_ARTIFACT_SCHEMA,
        "artifact_lineage": "v19-completion-safety-reserve",
        "parent_artifact_lineage": "v17-tail-service-reserve",
        "parent_method": parent_method,
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
        "actor_tensor_sha256": args.expected_actor_tensor_sha256,
        "temperature_tensor_sha256": args.expected_temperature_tensor_sha256,
        "completion_safety_reserve": {
            "schema_version": "transit-h2oplus-v19-completion-safety-v1",
            "variant": variant,
            "parent_tail_service_floor_seconds": (
                PARENT_TAIL_SERVICE_FLOOR_SECONDS
            ),
            "reserve_enabled": registered["enabled"],
            "reserve_floor_seconds": registered["floor_seconds"],
            "activation_start_fraction": ACTIVATION_START_FRACTION,
            "activation_end_fraction": ACTIVATION_END_FRACTION,
            "max_onboard_capacity_fraction": registered[
                "max_onboard_fraction"
            ],
            "requires_forward_bus_present": True,
            "requires_backward_bus_absent": True,
            "requires_positive_eligible_arrival_rate": True,
            "unit": "physical_hold_seconds",
            "activation_budget_online_events": 25_000,
            "parent_action_preserved_outside_guard": True,
            "guarded_hold_may_exceed_parent": True,
            "physical_max_hold_preserved": True,
            "predesignated_candidate": variant == "quarter-load-sixty",
            "checkpoint_selection_by_behavior": False,
            "v19_protocol_sha256": protocol_hash,
            "formal_held_out_outcomes_read": False,
        },
        "selection_rule": (
            "sealed v17 twenty-second 25k parent fixed before v19; baseline, "
            "quarter-load-sixty candidate and quarter-load-fifty diagnostic "
            "fixed before any v19 outcome; v14-v18 development outcomes informed "
            "the design; no checkpoint was selected by behavior"
        ),
        "training_metadata": extra,
    }


base = v17.v16.v15.v14.v13.v12.v11.base


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--expected-v19-parent-floor-seconds", type=float, required=True
    )
    parser.add_argument(
        "--expected-v19-reserve-enabled", type=int, choices=(0, 1), required=True
    )
    parser.add_argument(
        "--expected-v19-reserve-floor-seconds", type=float, required=True
    )
    parser.add_argument(
        "--expected-v19-activation-start-fraction", type=float, required=True
    )
    parser.add_argument(
        "--expected-v19-activation-end-fraction", type=float, required=True
    )
    parser.add_argument(
        "--expected-v19-max-onboard-fraction", type=float, required=True
    )
    parser.add_argument("--expected-v19-parent-method-name", required=True)
    expected, remaining = parser.parse_known_args()

    def final_boundary_build(args, config, station_max, dataset_hash):
        args.expected_v19_parent_floor_seconds = (
            expected.expected_v19_parent_floor_seconds
        )
        args.expected_v19_reserve_enabled = bool(
            expected.expected_v19_reserve_enabled
        )
        args.expected_v19_reserve_floor_seconds = (
            expected.expected_v19_reserve_floor_seconds
        )
        args.expected_v19_activation_start_fraction = (
            expected.expected_v19_activation_start_fraction
        )
        args.expected_v19_activation_end_fraction = (
            expected.expected_v19_activation_end_fraction
        )
        args.expected_v19_max_onboard_fraction = (
            expected.expected_v19_max_onboard_fraction
        )
        args.expected_v19_parent_method_name = (
            expected.expected_v19_parent_method_name
        )
        return build_development_policy(args, config, station_max, dataset_hash)

    base.build_policy = final_boundary_build
    base.evaluation_source_paths = evaluation_source_paths
    sys.argv = [sys.argv[0], *remaining]
    v17.main()


if __name__ == "__main__":
    main()
