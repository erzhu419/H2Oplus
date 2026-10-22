#!/usr/bin/env python3
"""Evaluate a preregistered v8 development few-shot actor in SUMO."""

from __future__ import annotations

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

from bus_h2o.sumo_provenance import sha256
from bus_h2o.mdp_contract_v7 import raw_state_to_canonical_v7
from SimpleSAC import evaluate_transit_v7 as base
from SimpleSAC.transit_v7.trainer import (
    POLICY_ARTIFACT_SCHEMA,
    load_actor_checkpoint_v7,
)


ADMISSION_SCHEMA = "transit-training-admission-v8-fewshot-development-v1"
PROTOCOL_SHA_ENV = "H2OPLUS_V8_FEWSHOT_PROTOCOL_SHA256"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus|sac-online)-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)


_base_evaluation_source_paths = base.evaluation_source_paths


def evaluation_source_paths(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    """Match the matrix-side source contract for v8 few-shot results."""
    paths = set(_base_evaluation_source_paths(sumo_dir, edge_xml))
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "reproducibility" / "run_evaluation_matrix_v8_fewshot.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def _actor_digest(state: dict) -> str:
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


def _expected_phase(algorithm: str, events: int) -> str:
    if events == 0:
        return "offline_initializer_frozen" if algorithm == "h2oplus" else "initialized"
    if (algorithm, events) in {("h2oplus", 100_000), ("sac-online", 150_000)}:
        return "complete"
    return "online_training"


def build_development_policy(args, config, station_max, dataset_hash):
    if args.checkpoint is None:
        return base.build_policy(args, config, station_max, dataset_hash)
    match = METHOD_NAME.fullmatch(args.method_name)
    if match is None:
        raise ValueError("few-shot checkpoint method name does not encode its budget")
    algorithm = match.group("algorithm")
    seed = int(match.group("seed"))
    events = int(match.group("events"))
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    actor, extra = load_actor_checkpoint_v7(
        checkpoint, args.device, require_policy_artifact=True
    )
    expected = {
        "dataset_sha256": dataset_hash,
        "contract_config": config.to_dict(),
        "mdp_version": "v7",
        "policy_artifact_schema": POLICY_ARTIFACT_SCHEMA,
        "method": algorithm,
        "training_seed": seed,
        "run_signature": args.expected_training_run_signature,
        "training_checkpoint_sha256": args.expected_training_checkpoint_sha256,
        "phase": _expected_phase(algorithm, events),
        "fewshot_online_events_nominal": events,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {
        name: {"checkpoint": extra.get(name), "evaluation": value}
        for name, value in expected.items()
        if extra.get(name) != value
    }
    if mismatches:
        raise ValueError(f"few-shot policy provenance mismatch: {mismatches}")
    if args.expected_algorithm != algorithm or args.expected_training_seed != seed:
        raise ValueError("matrix method identity differs from the encoded checkpoint")
    admission = extra.get("training_admission")
    if not (
        isinstance(admission, dict)
        and admission.get("schema_version") == ADMISSION_SCHEMA
        and admission.get("study_stage") == "development"
        and isinstance(admission.get("sha256"), str)
        and admission.get("sha256") == args.expected_training_admission_sha256
    ):
        raise ValueError("few-shot policy lacks its development admission")
    protocol = extra.get("fewshot_protocol")
    expected_protocol_sha256 = os.environ.get(PROTOCOL_SHA_ENV, "")
    if not (
        isinstance(protocol, dict)
        and SHA256.fullmatch(expected_protocol_sha256)
        and protocol.get("sha256") == expected_protocol_sha256
    ):
        raise ValueError("few-shot policy lacks its frozen protocol")
    digest = _actor_digest(actor.state_dict())
    if digest != extra.get("fewshot_actor_tensor_sha256"):
        raise ValueError("few-shot actor tensor digest drifted")

    def checkpoint_policy(raw_state: np.ndarray) -> np.ndarray:
        canonical = raw_state_to_canonical_v7(raw_state, station_max, config)
        with torch.inference_mode():
            tensor = torch.as_tensor(
                canonical[None, :], dtype=torch.float32, device=args.device
            )
            action, _ = actor(tensor, deterministic=True)
        return action.cpu().numpy()[0]

    return checkpoint_policy, {
        "type": "development-fewshot-policy-artifact",
        "artifact_schema": POLICY_ARTIFACT_SCHEMA,
        "path": str(checkpoint),
        "sha256": sha256(checkpoint),
        "selection_rule": (
            "all preregistered online-event milestones; no development or formal "
            "evaluation outcome selects a checkpoint"
        ),
        "training_metadata": extra,
    }


base.build_policy = build_development_policy
base.evaluation_source_paths = evaluation_source_paths


if __name__ == "__main__":
    base.main()
