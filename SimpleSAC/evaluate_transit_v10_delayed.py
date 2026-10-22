#!/usr/bin/env python3
"""Evaluate one frozen v8/v10 development actor for the v10 pilot."""

from __future__ import annotations

import argparse
import hashlib
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
from SimpleSAC import evaluate_transit_v7 as base
from SimpleSAC.transit_v7.trainer import POLICY_ARTIFACT_SCHEMA, load_actor_checkpoint_v7


V8_ADMISSION_SCHEMA = "transit-training-admission-v8-fewshot-development-v1"
V10_ADMISSION_SCHEMA = "transit-training-admission-v10-delayed-development-v4"
V8_PROTOCOL_SHA_ENV = "H2OPLUS_V8_FEWSHOT_PROTOCOL_SHA256"
V10_PROTOCOL_SHA_ENV = "H2OPLUS_V10_DELAYED_PROTOCOL_SHA256"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus|sac-online|h2oplus-v10-(?:shared-prefix|unit-control|"
    r"delayed-replay|delayed-actor|delayed-order|delayed-combined))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)


_base_evaluation_source_paths = base.evaluation_source_paths


def evaluation_source_paths(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(_base_evaluation_source_paths(sumo_dir, edge_xml))
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "reproducibility/run_evaluation_matrix_v10_delayed.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def actor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
        digest.update(b"\n")
    return digest.hexdigest()


def _v8_expected_phase(algorithm: str, events: int) -> str:
    if events == 0:
        return "offline_initializer_frozen" if algorithm == "h2oplus" else "initialized"
    return "online_training"


def _validate_v8(extra: dict, algorithm: str, events: int, args) -> None:
    protocol_hash = os.environ.get(V8_PROTOCOL_SHA_ENV, "")
    expected = {
        "method": algorithm,
        "phase": _v8_expected_phase(algorithm, events),
        "fewshot_online_events_nominal": events,
        "fewshot_learner_update_steps": args.expected_learner_update_steps,
        "fewshot_ratio_optimizer_steps": args.expected_ratio_optimizer_steps,
    }
    mismatches = {
        key: {"artifact": extra.get(key), "expected": value}
        for key, value in expected.items()
        if extra.get(key) != value
    }
    admission = extra.get("training_admission", {})
    if (
        mismatches
        or admission.get("schema_version") != V8_ADMISSION_SCHEMA
        or admission.get("sha256") != args.expected_training_admission_sha256
        or extra.get("fewshot_protocol", {}).get("sha256") != protocol_hash
        or not SHA256.fullmatch(protocol_hash)
        or extra.get("fewshot_actor_tensor_sha256")
        != args.expected_actor_tensor_sha256
    ):
        raise ValueError(f"v8 control policy provenance mismatch: {mismatches}")


def _validate_v10(extra: dict, algorithm: str, events: int, args) -> None:
    variant = algorithm[len("h2oplus-v10-") :]
    protocol_hash = os.environ.get(V10_PROTOCOL_SHA_ENV, "")
    expected_stage = "prefix" if variant == "shared-prefix" else "branch"
    expected = {
        "method": "h2oplus",
        "variant": variant,
        "stage": expected_stage,
        "phase": "complete",
        "v10_online_events_nominal": events,
        "v10_learner_update_steps": args.expected_learner_update_steps,
        "v10_ratio_optimizer_steps": args.expected_ratio_optimizer_steps,
    }
    mismatches = {
        key: {"artifact": extra.get(key), "expected": value}
        for key, value in expected.items()
        if extra.get(key) != value
    }
    admission = extra.get("training_admission", {})
    intervention = extra.get("v10_intervention", {})
    if (
        mismatches
        or admission.get("schema_version") != V10_ADMISSION_SCHEMA
        or admission.get("sha256") != args.expected_training_admission_sha256
        or extra.get("v10_protocol", {}).get("sha256") != protocol_hash
        or not SHA256.fullmatch(protocol_hash)
        or extra.get("v10_actor_tensor_sha256") != args.expected_actor_tensor_sha256
        or intervention.get("variant") != variant
        or intervention.get("unit_ratio_weights") is not True
    ):
        raise ValueError(f"v10 delayed policy provenance mismatch: {mismatches}")


def build_development_policy(args, config, station_max, dataset_hash):
    if args.checkpoint is None:
        return base.build_policy(args, config, station_max, dataset_hash)
    match = METHOD_NAME.fullmatch(args.method_name)
    if match is None:
        raise ValueError("v10 development checkpoint method name is malformed")
    algorithm = match.group("algorithm")
    seed = int(match.group("seed"))
    events = int(match.group("events"))
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
        raise ValueError(f"development policy common provenance mismatch: {mismatches}")
    if args.expected_algorithm != algorithm or args.expected_training_seed != seed:
        raise ValueError("matrix identity differs from encoded checkpoint identity")
    if algorithm.startswith("h2oplus-v10-"):
        _validate_v10(extra, algorithm, events, args)
        lineage = "v10-delayed"
    else:
        _validate_v8(extra, algorithm, events, args)
        lineage = "v8-fewshot-control"
    if actor_digest(actor.state_dict()) != args.expected_actor_tensor_sha256:
        raise ValueError("actor tensor digest differs from the frozen methods manifest")

    def checkpoint_policy(raw_state: np.ndarray) -> np.ndarray:
        canonical = raw_state_to_canonical_v7(raw_state, station_max, config)
        with torch.inference_mode():
            tensor = torch.as_tensor(
                canonical[None, :], dtype=torch.float32, device=args.device
            )
            action, _ = actor(tensor, deterministic=True)
        return action.cpu().numpy()[0]

    return checkpoint_policy, {
        "type": "development-v10-delayed-policy-artifact",
        "artifact_schema": POLICY_ARTIFACT_SCHEMA,
        "artifact_lineage": lineage,
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
        "actor_tensor_sha256": args.expected_actor_tensor_sha256,
        "selection_rule": (
            "all preregistered common-budget aliases frozen before evaluation; "
            "physical actors deduplicated only by exact tensor SHA256"
        ),
        "training_metadata": extra,
    }


base.build_policy = build_development_policy
base.evaluation_source_paths = evaluation_source_paths


def main() -> None:
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--expected-actor-tensor-sha256", required=True)
    extra_parser.add_argument("--expected-learner-update-steps", type=int, required=True)
    extra_parser.add_argument("--expected-ratio-optimizer-steps", type=int, required=True)
    extra, remaining = extra_parser.parse_known_args()
    original_parse_args = base.parse_args

    def parse_args():
        args = original_parse_args()
        for key, value in vars(extra).items():
            setattr(args, key, value)
        return args

    base.parse_args = parse_args
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
