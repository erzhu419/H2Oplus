#!/usr/bin/env python3
"""Evaluate one frozen v8/v10/v11/v12/v13 actor for v13 development."""

from __future__ import annotations

import argparse
import json
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

from bus_h2o.mdp_contract_v7 import raw_state_to_canonical_v7
from bus_h2o.sumo_provenance import sha256
from SimpleSAC import evaluate_transit_v12_staged_release as v12
from SimpleSAC.transit_v7.trainer import POLICY_ARTIFACT_SCHEMA, load_actor_checkpoint_v7


V13_ADMISSION_SCHEMA = "transit-training-admission-v13-proximal-release-development-v2"
V13_PROTOCOL_SHA_ENV = "H2OPLUS_V13_PROXIMAL_RELEASE_PROTOCOL_SHA256"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus-v13-(?P<variant>proximal-(?:weak|medium|strong)))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
PROXIMAL_RATES = {
    "proximal-weak": 1e-5,
    "proximal-medium": 2.5e-5,
    "proximal-strong": 5e-5,
}


def evaluation_source_paths(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v12.evaluation_source_paths(sumo_dir, edge_xml))
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "reproducibility/run_evaluation_matrix_v13_proximal_release.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def _validate_v13(extra: dict, variant: str, events: int, args) -> None:
    protocol_hash = os.environ.get(V13_PROTOCOL_SHA_ENV, "")
    rate = PROXIMAL_RATES[variant]
    expected = {
        "method": "h2oplus",
        "variant": variant,
        "stage": "branch",
        "phase": "complete",
        "v13_online_events_nominal": events,
        "v13_learner_update_steps": args.expected_learner_update_steps,
        "v13_ratio_optimizer_steps": args.expected_ratio_optimizer_steps,
        "v13_actor_tensor_sha256": args.expected_actor_tensor_sha256,
        "v13_temperature_tensor_sha256": args.expected_temperature_tensor_sha256,
    }
    mismatches = {
        key: {"artifact": extra.get(key), "expected": value}
        for key, value in expected.items()
        if extra.get(key) != value
    }
    admission = extra.get("training_admission", {})
    selected = admission.get("selected_arguments", {})
    handoff = selected.get("source_handoff", {})
    intervention = extra.get("v13_intervention", {})
    expected_retention = (1.0 - rate) ** 15_000
    expected_handoff_events = 9_999 if args.expected_training_seed == 6205 else 10_000
    if (
        mismatches
        or events != 25_000
        or admission.get("schema_version") != V13_ADMISSION_SCHEMA
        or admission.get("sha256") != args.expected_training_admission_sha256
        or extra.get("v13_protocol", {}).get("sha256") != protocol_hash
        or not SHA256.fullmatch(protocol_hash)
        or handoff.get("online_events_actual") != expected_handoff_events
        or not SHA256.fullmatch(str(handoff.get("manifest_sha256", "")))
        or not SHA256.fullmatch(str(handoff.get("checkpoint_state_sha256", "")))
        or intervention.get("schema_version")
        != "transit-h2oplus-v13-proximal-release-variant-v1"
        or intervention.get("variant") != variant
        or intervention.get("warmup_events") != 5_000
        or intervention.get("actor_and_temperature_hold_through_online_events") != 10_000
        or intervention.get("actor_learning_rate_ramp") != "0.1 to 1.0 over 10k-15k"
        or intervention.get("temperature_learning_rate_after_release")
        != "full registered rate"
        or intervention.get("actor_state_source_after_release") != "online-only"
        or intervention.get("actor_anchor")
        != "exact seed-matched sealed v12 10k actor tensors"
        or intervention.get("proximal_update")
        != "theta <- (1-rate)*theta + rate*theta_anchor"
        or intervention.get("proximal_rate_per_actor_update") != rate
        or intervention.get("registered_post_10k_actor_updates") != 15_000
        or not math.isclose(
            float(intervention.get("no_gradient_anchor_retention_at_25k", math.nan)),
            expected_retention,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or intervention.get("predesignated_candidate")
        is not (variant == "proximal-medium")
        or intervention.get("offline_critic_replay_schedule")
        != [[0, 0.5], [25_000, 0.5]]
        or intervention.get("online_update_order") != "v7-actor-before-critic"
        or intervention.get("unit_ratio_weights") is not True
        or intervention.get("ratio_classifiers_continue_for_diagnostics") is not True
        or intervention.get("checkpoint_selection_by_behavior") is not False
    ):
        raise ValueError(f"v13 proximal-release policy mismatch: {mismatches}")


def build_development_policy(args, config, station_max, dataset_hash):
    match = METHOD_NAME.fullmatch(args.method_name)
    if match is None:
        return v12.build_development_policy(args, config, station_max, dataset_hash)
    algorithm = match.group("algorithm")
    variant = match.group("variant")
    seed = int(match.group("seed"))
    events = int(match.group("events"))
    if args.checkpoint is None:
        raise ValueError("v13 physical method requires a checkpoint")
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
        raise ValueError(f"v13 common provenance mismatch: {mismatches}")
    if args.expected_algorithm != algorithm or args.expected_training_seed != seed:
        raise ValueError("matrix identity differs from v13 checkpoint identity")
    _validate_v13(extra, variant, events, args)
    if v12.v11.actor_digest(actor.state_dict()) != args.expected_actor_tensor_sha256:
        raise ValueError("v13 actor tensor differs from methods manifest")

    def checkpoint_policy(raw_state: np.ndarray) -> np.ndarray:
        canonical = raw_state_to_canonical_v7(raw_state, station_max, config)
        with torch.inference_mode():
            tensor = torch.as_tensor(
                canonical[None, :], dtype=torch.float32, device=args.device
            )
            action, _ = actor(tensor, deterministic=True)
        return action.cpu().numpy()[0]

    return checkpoint_policy, {
        "type": "development-v13-proximal-release-policy-artifact",
        "artifact_schema": POLICY_ARTIFACT_SCHEMA,
        "artifact_lineage": "v13-proximal-release",
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


v12.v11.base.build_policy = build_development_policy
v12.v11.base.evaluation_source_paths = evaluation_source_paths


def main() -> None:
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--expected-temperature-tensor-sha256", default="")
    extra, remaining = extra_parser.parse_known_args()
    original_parse_args = v12.v11.base.parse_args

    def parse_args():
        args = original_parse_args()
        args.expected_temperature_tensor_sha256 = extra.expected_temperature_tensor_sha256
        return args

    v12.v11.base.parse_args = parse_args
    sys.argv = [sys.argv[0], *remaining]
    v12.v11.main()


if __name__ == "__main__":
    main()
