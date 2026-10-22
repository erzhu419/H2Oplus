#!/usr/bin/env python3
"""Evaluate one frozen v14 actor with a preregistered early headway reserve."""

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

from bus_h2o.mdp_contract_v4 import RAW_EVENT_STATE_DIM as RAW_EVENT_STATE_DIM_V4
from bus_h2o.mdp_contract_v7 import (
    RAW_STATE_INDEX,
    current_event_v7,
    hold_seconds_to_policy_action_v7,
    policy_action_to_hold_seconds_v7,
    raw_state_to_canonical_v7,
)
from bus_h2o.sumo_provenance import sha256
from SimpleSAC import evaluate_transit_v15_hold_calibration as v15
from SimpleSAC.transit_v4.policies import headway_balance_hold_seconds_v4
from SimpleSAC.transit_v7.trainer import POLICY_ARTIFACT_SCHEMA, load_actor_checkpoint_v7


V16_PROTOCOL_SHA_ENV = "H2OPLUS_V16_HEADWAY_RESERVE_PROTOCOL_SHA256"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus-v16-headway-reserve-"
    r"(?P<variant>half|three-quarter|full))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
HORIZON_CUTOFFS = {"half": 0.5, "three-quarter": 0.75, "full": 1.0}
BASE_HOLD_SCALE = 0.125
HEADWAY_BALANCE_GAIN = 1.0


def reserved_hold_action(
    raw_state: np.ndarray,
    policy_action: np.ndarray | float,
    reserve_horizon_fraction: float,
    config,
) -> np.ndarray:
    """Apply the bounded early reserve in physical hold seconds."""

    if reserve_horizon_fraction not in HORIZON_CUTOFFS.values():
        raise ValueError(f"unregistered v16 horizon cutoff: {reserve_horizon_fraction!r}")
    action = np.asarray(policy_action, dtype=np.float32)
    if action.shape not in ((), (1,)) or not np.isfinite(action).all():
        raise ValueError("v16 policy action must be one finite scalar")
    parent_hold = float(
        policy_action_to_hold_seconds_v7(float(action.reshape(-1)[0]), config)
    )
    base_hold = parent_hold * BASE_HOLD_SCALE
    state = np.asarray(raw_state, dtype=np.float32).reshape(-1)
    horizon_progress = float(
        np.clip(
            state[RAW_STATE_INDEX["decision_time_seconds"]] / config.horizon_seconds,
            0.0,
            1.0,
        )
    )
    calibrated_hold = base_hold
    if horizon_progress < reserve_horizon_fraction and parent_hold > 0.0:
        event = current_event_v7(raw_state)[None, :RAW_EVENT_STATE_DIM_V4]
        reserve = float(
            headway_balance_hold_seconds_v4(
                event,
                HEADWAY_BALANCE_GAIN,
                config,
            )[0]
        )
        calibrated_hold = min(parent_hold, max(base_hold, reserve))
    if not 0.0 <= calibrated_hold <= parent_hold + 1e-9:
        raise ValueError("v16 transformed hold escaped its parent-action bound")
    calibrated_action = hold_seconds_to_policy_action_v7(calibrated_hold, config)
    return np.asarray([calibrated_action], dtype=np.float32)


def evaluation_source_paths(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v15.evaluation_source_paths(sumo_dir, edge_xml))
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "reproducibility/run_evaluation_matrix_v16_headway_reserve.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def build_development_policy(args, config, station_max, dataset_hash):
    match = METHOD_NAME.fullmatch(args.method_name)
    if match is None:
        return v15.build_development_policy(args, config, station_max, dataset_hash)
    algorithm = match.group("algorithm")
    variant = match.group("variant")
    seed = int(match.group("seed"))
    events = int(match.group("events"))
    cutoff = HORIZON_CUTOFFS[variant]
    parent_method = f"h2oplus-v15-hold-eighth-seed{seed}-events000025000"
    protocol_hash = os.environ.get(V16_PROTOCOL_SHA_ENV, "")
    if (
        events != 25_000
        or args.checkpoint is None
        or args.expected_algorithm != algorithm
        or args.expected_training_seed != seed
        or args.expected_parent_method_name != parent_method
        or not math.isclose(
            float(args.expected_base_hold_scale),
            BASE_HOLD_SCALE,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(args.expected_headway_balance_gain),
            HEADWAY_BALANCE_GAIN,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not math.isclose(
            float(args.expected_reserve_horizon_fraction),
            cutoff,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        or not SHA256.fullmatch(protocol_hash)
    ):
        raise ValueError("v16 matrix identity or protocol binding drifted")

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
        raise ValueError(f"v16 parent common provenance mismatch: {mismatches}")

    v15.v14._validate_v14(extra, "terminal-quarter", events, args)
    if (
        v15.v14.v13.v12.v11.actor_digest(actor.state_dict())
        != args.expected_actor_tensor_sha256
    ):
        raise ValueError("v16 parent actor tensor differs from methods manifest")

    def checkpoint_policy(raw_state: np.ndarray) -> np.ndarray:
        canonical = raw_state_to_canonical_v7(raw_state, station_max, config)
        with torch.inference_mode():
            tensor = torch.as_tensor(
                canonical[None, :], dtype=torch.float32, device=args.device
            )
            action, _ = actor(tensor, deterministic=True)
        return reserved_hold_action(
            raw_state,
            action.cpu().numpy()[0],
            cutoff,
            config,
        )

    return checkpoint_policy, {
        "type": "development-v16-headway-reserve-policy",
        "artifact_schema": POLICY_ARTIFACT_SCHEMA,
        "artifact_lineage": "v16-headway-reserve",
        "parent_artifact_lineage": "v15-hold-calibration",
        "parent_method": parent_method,
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
        "actor_tensor_sha256": args.expected_actor_tensor_sha256,
        "temperature_tensor_sha256": args.expected_temperature_tensor_sha256,
        "headway_reserve": {
            "schema_version": "transit-h2oplus-v16-headway-reserve-v1",
            "variant": variant,
            "base_hold_scale": BASE_HOLD_SCALE,
            "headway_balance_gain": HEADWAY_BALANCE_GAIN,
            "reserve_horizon_fraction": cutoff,
            "unit": "physical_hold_seconds",
            "activation_budget_online_events": 25_000,
            "zero_hold_preserved_exactly": True,
            "transformed_hold_never_exceeds_parent": True,
            "predesignated_candidate": variant == "three-quarter",
            "checkpoint_selection_by_behavior": False,
            "v16_protocol_sha256": protocol_hash,
            "formal_held_out_outcomes_read": False,
        },
        "selection_rule": (
            "sealed v15 hold-eighth 25k parent fixed before v16; base scale, "
            "headway gain, and horizon cutoff fixed before any v16 outcomes; "
            "physical behavior keyed by actor tensor SHA256 and full transform"
        ),
        "training_metadata": extra,
    }


v15.v14.v13.v12.v11.base.build_policy = build_development_policy
v15.v14.v13.v12.v11.base.evaluation_source_paths = evaluation_source_paths


def main() -> None:
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--expected-temperature-tensor-sha256", default="")
    extra_parser.add_argument("--expected-base-hold-scale", type=float, required=True)
    extra_parser.add_argument(
        "--expected-headway-balance-gain", type=float, required=True
    )
    extra_parser.add_argument(
        "--expected-reserve-horizon-fraction", type=float, required=True
    )
    extra_parser.add_argument("--expected-parent-method-name", required=True)
    extra, remaining = extra_parser.parse_known_args()
    original_parse_args = v15.v14.v13.v12.v11.base.parse_args

    def parse_args():
        args = original_parse_args()
        args.expected_temperature_tensor_sha256 = extra.expected_temperature_tensor_sha256
        args.expected_base_hold_scale = extra.expected_base_hold_scale
        args.expected_headway_balance_gain = extra.expected_headway_balance_gain
        args.expected_reserve_horizon_fraction = extra.expected_reserve_horizon_fraction
        args.expected_parent_method_name = extra.expected_parent_method_name
        return args

    v15.v14.v13.v12.v11.base.parse_args = parse_args
    sys.argv = [sys.argv[0], *remaining]
    v15.v14.v13.v12.v11.main()


if __name__ == "__main__":
    main()
