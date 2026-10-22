#!/usr/bin/env python3
"""Evaluate one frozen v8/v10/v11/v12 actor for v12 development."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import raw_state_to_canonical_v7
from bus_h2o.sumo_provenance import sha256
from SimpleSAC import evaluate_transit_v11_actor_hold as v11
from SimpleSAC.transit_v7.trainer import POLICY_ARTIFACT_SCHEMA, load_actor_checkpoint_v7


V12_ADMISSION_SCHEMA = "transit-training-admission-v12-staged-release-development-v2"
V12_PROTOCOL_SHA_ENV = "H2OPLUS_V12_STAGED_RELEASE_PROTOCOL_SHA256"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus-v12-(?P<variant>abrupt-control|actor-ramp|"
    r"temperature-ramp|actor-anchor|staged-combined))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)


def evaluation_source_paths(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v11._base_evaluation_source_paths(sumo_dir, edge_xml))
    paths.update(
        {
            ROOT / "SimpleSAC/evaluate_transit_v11_actor_hold.py",
            Path(__file__).resolve(),
            ROOT / "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
            ROOT / "reproducibility/run_evaluation_matrix_v12_staged_release.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def _validate_v12(extra: dict, variant: str, events: int, args) -> None:
    protocol_hash = os.environ.get(V12_PROTOCOL_SHA_ENV, "")
    expected = {
        "method": "h2oplus",
        "variant": variant,
        "stage": "branch",
        "phase": "complete",
        "v12_online_events_nominal": events,
        "v12_learner_update_steps": args.expected_learner_update_steps,
        "v12_ratio_optimizer_steps": args.expected_ratio_optimizer_steps,
        "v12_actor_tensor_sha256": args.expected_actor_tensor_sha256,
        "v12_temperature_tensor_sha256": (
            args.expected_temperature_tensor_sha256
        ),
    }
    mismatches = {
        key: {"artifact": extra.get(key), "expected": value}
        for key, value in expected.items()
        if extra.get(key) != value
    }
    flags = {
        "abrupt-control": (False, False, False),
        "actor-ramp": (True, False, False),
        "temperature-ramp": (False, True, False),
        "actor-anchor": (False, False, True),
        "staged-combined": (True, True, True),
    }[variant]
    admission = extra.get("training_admission", {})
    intervention = extra.get("v12_intervention", {})
    if (
        mismatches
        or events != 25_000
        or admission.get("schema_version") != V12_ADMISSION_SCHEMA
        or admission.get("sha256") != args.expected_training_admission_sha256
        or extra.get("v12_protocol", {}).get("sha256") != protocol_hash
        or not SHA256.fullmatch(protocol_hash)
        or intervention.get("schema_version")
        != "transit-h2oplus-v12-staged-release-variant-v1"
        or intervention.get("variant") != variant
        or intervention.get("actor_learning_rate_ramp") is not flags[0]
        or intervention.get("temperature_learning_rate_ramp") is not flags[1]
        or intervention.get("actor_state_anchor") is not flags[2]
        or intervention.get("unit_ratio_weights") is not True
        or intervention.get("checkpoint_selection_by_behavior") is not False
    ):
        raise ValueError(f"v12 staged-release policy mismatch: {mismatches}")


def build_development_policy(args, config, station_max, dataset_hash):
    match = METHOD_NAME.fullmatch(args.method_name)
    if match is None:
        return v11.build_development_policy(args, config, station_max, dataset_hash)
    algorithm = match.group("algorithm")
    variant = match.group("variant")
    seed = int(match.group("seed"))
    events = int(match.group("events"))
    if args.checkpoint is None:
        raise ValueError("v12 physical method requires a checkpoint")
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
        raise ValueError(f"v12 common provenance mismatch: {mismatches}")
    if args.expected_algorithm != algorithm or args.expected_training_seed != seed:
        raise ValueError("matrix identity differs from v12 checkpoint identity")
    _validate_v12(extra, variant, events, args)
    if v11.actor_digest(actor.state_dict()) != args.expected_actor_tensor_sha256:
        raise ValueError("v12 actor tensor differs from methods manifest")

    def checkpoint_policy(raw_state: np.ndarray) -> np.ndarray:
        canonical = raw_state_to_canonical_v7(raw_state, station_max, config)
        with torch.inference_mode():
            tensor = torch.as_tensor(
                canonical[None, :], dtype=torch.float32, device=args.device
            )
            action, _ = actor(tensor, deterministic=True)
        return action.cpu().numpy()[0]

    return checkpoint_policy, {
        "type": "development-v12-staged-release-policy-artifact",
        "artifact_schema": POLICY_ARTIFACT_SCHEMA,
        "artifact_lineage": "v12-staged-release",
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
        "actor_tensor_sha256": args.expected_actor_tensor_sha256,
        "temperature_tensor_sha256": args.expected_temperature_tensor_sha256,
        "selection_rule": (
            "all preregistered common-budget aliases frozen before evaluation; "
            "physical actors deduplicated only by exact tensor SHA256"
        ),
        "training_metadata": extra,
    }


v11.base.build_policy = build_development_policy
v11.base.evaluation_source_paths = evaluation_source_paths


def main() -> None:
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--expected-temperature-tensor-sha256", default="")
    extra, remaining = extra_parser.parse_known_args()
    original_parse_args = v11.base.parse_args

    def parse_args():
        args = original_parse_args()
        args.expected_temperature_tensor_sha256 = (
            extra.expected_temperature_tensor_sha256
        )
        return args

    v11.base.parse_args = parse_args
    sys.argv = [sys.argv[0], *remaining]
    v11.main()


if __name__ == "__main__":
    main()
