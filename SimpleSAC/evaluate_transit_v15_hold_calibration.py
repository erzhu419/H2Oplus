#!/usr/bin/env python3
"""Evaluate one frozen v14 terminal-quarter actor with fixed hold calibration."""

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

from bus_h2o.mdp_contract_v7 import (
    hold_seconds_to_policy_action_v7,
    policy_action_to_hold_seconds_v7,
    raw_state_to_canonical_v7,
)
from bus_h2o.sumo_provenance import sha256
from SimpleSAC import evaluate_transit_v14_terminal_taper as v14
from SimpleSAC.transit_v7.trainer import POLICY_ARTIFACT_SCHEMA, load_actor_checkpoint_v7


V15_PROTOCOL_SHA_ENV = "H2OPLUS_V15_HOLD_CALIBRATION_PROTOCOL_SHA256"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus-v15-hold-(?P<variant>eighth|quarter|three-eighths))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
HOLD_SCALES = {
    "eighth": 0.125,
    "quarter": 0.25,
    "three-eighths": 0.375,
}


def calibrated_hold_action(
    policy_action: np.ndarray | float,
    hold_scale: float,
    config,
) -> np.ndarray:
    """Scale physical hold seconds while preserving exact zero-hold semantics."""

    if hold_scale not in HOLD_SCALES.values():
        raise ValueError(f"unregistered v15 hold scale: {hold_scale!r}")
    action = np.asarray(policy_action, dtype=np.float32)
    if action.shape not in ((), (1,)) or not np.isfinite(action).all():
        raise ValueError("v15 policy action must be one finite scalar")
    hold_seconds = float(policy_action_to_hold_seconds_v7(float(action.reshape(-1)[0]), config))
    calibrated_seconds = hold_seconds * hold_scale
    calibrated_action = hold_seconds_to_policy_action_v7(calibrated_seconds, config)
    return np.asarray([calibrated_action], dtype=np.float32)


def evaluation_source_paths(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v14.evaluation_source_paths(sumo_dir, edge_xml))
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "reproducibility/run_evaluation_matrix_v15_hold_calibration.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def build_development_policy(args, config, station_max, dataset_hash):
    match = METHOD_NAME.fullmatch(args.method_name)
    if match is None:
        return v14.build_development_policy(args, config, station_max, dataset_hash)
    algorithm = match.group("algorithm")
    variant = match.group("variant")
    seed = int(match.group("seed"))
    events = int(match.group("events"))
    hold_scale = HOLD_SCALES[variant]
    parent_method = f"h2oplus-v14-terminal-quarter-seed{seed}-events000025000"
    protocol_hash = os.environ.get(V15_PROTOCOL_SHA_ENV, "")
    if (
        events != 25_000
        or args.checkpoint is None
        or args.expected_algorithm != algorithm
        or args.expected_training_seed != seed
        or args.expected_parent_method_name != parent_method
        or not math.isclose(
            float(args.expected_hold_scale), hold_scale, rel_tol=0.0, abs_tol=0.0
        )
        or not SHA256.fullmatch(protocol_hash)
    ):
        raise ValueError("v15 matrix identity or protocol binding drifted")

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
        raise ValueError(f"v15 parent common provenance mismatch: {mismatches}")

    v14._validate_v14(extra, "terminal-quarter", events, args)
    if v14.v13.v12.v11.actor_digest(actor.state_dict()) != args.expected_actor_tensor_sha256:
        raise ValueError("v15 parent actor tensor differs from methods manifest")

    def checkpoint_policy(raw_state: np.ndarray) -> np.ndarray:
        canonical = raw_state_to_canonical_v7(raw_state, station_max, config)
        with torch.inference_mode():
            tensor = torch.as_tensor(
                canonical[None, :], dtype=torch.float32, device=args.device
            )
            action, _ = actor(tensor, deterministic=True)
        return calibrated_hold_action(action.cpu().numpy()[0], hold_scale, config)

    return checkpoint_policy, {
        "type": "development-v15-hold-calibration-policy",
        "artifact_schema": POLICY_ARTIFACT_SCHEMA,
        "artifact_lineage": "v15-hold-calibration",
        "parent_artifact_lineage": "v14-terminal-taper",
        "parent_method": parent_method,
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
        "actor_tensor_sha256": args.expected_actor_tensor_sha256,
        "temperature_tensor_sha256": args.expected_temperature_tensor_sha256,
        "hold_calibration": {
            "schema_version": "transit-h2oplus-v15-hold-calibration-v1",
            "variant": variant,
            "hold_scale": hold_scale,
            "unit": "physical_hold_seconds",
            "activation_budget_online_events": 25_000,
            "zero_hold_preserved_exactly": True,
            "predesignated_candidate": variant == "quarter",
            "checkpoint_selection_by_behavior": False,
            "v15_protocol_sha256": protocol_hash,
            "formal_held_out_outcomes_read": False,
        },
        "selection_rule": (
            "sealed v14 terminal-quarter 25k parent fixed before v15; hold scale "
            "predesignated before any v15 outcomes; physical behavior keyed by actor "
            "tensor SHA256 and hold scale"
        ),
        "training_metadata": extra,
    }


v14.v13.v12.v11.base.build_policy = build_development_policy
v14.v13.v12.v11.base.evaluation_source_paths = evaluation_source_paths


def main() -> None:
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--expected-temperature-tensor-sha256", default="")
    extra_parser.add_argument("--expected-hold-scale", type=float, required=True)
    extra_parser.add_argument("--expected-parent-method-name", required=True)
    extra, remaining = extra_parser.parse_known_args()
    original_parse_args = v14.v13.v12.v11.base.parse_args

    def parse_args():
        args = original_parse_args()
        args.expected_temperature_tensor_sha256 = extra.expected_temperature_tensor_sha256
        args.expected_hold_scale = extra.expected_hold_scale
        args.expected_parent_method_name = extra.expected_parent_method_name
        return args

    v14.v13.v12.v11.base.parse_args = parse_args
    sys.argv = [sys.argv[0], *remaining]
    v14.v13.v12.v11.main()


if __name__ == "__main__":
    main()
