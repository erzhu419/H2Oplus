#!/usr/bin/env python3
"""Correct the v9 policy-count arithmetic and build the physical methods map."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


COMMON_EVENTS = (0, 5_000, 10_000, 25_000)
SEEDS = (6201, 6202, 6203)
ALGORITHMS = (
    "h2oplus",
    "sac-online",
    "h2oplus-v9-ratio-gated",
    "h2oplus-v9-adaptation",
    "h2oplus-v9-combined",
)
CORRECTION_SCHEMA = "transit-h2oplus-v9-adaptive-evaluation-count-correction-v1"
METHODS_SCHEMA = "transit-evaluation-methods-v9-adaptive-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": sha256(resolved)}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def actor_tensor_sha256(path: Path, lineage: str) -> str:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    extra = artifact.get("extra", {})
    key = "fewshot_actor_tensor_sha256" if lineage == "v8-fewshot" else "v9_actor_tensor_sha256"
    value = extra.get(key)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"actor tensor hash missing from {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-closeout", type=Path, required=True)
    parser.add_argument("--training-verification", type=Path, required=True)
    parser.add_argument("--v9-policies", type=Path, required=True)
    parser.add_argument("--v8-methods", type=Path, required=True)
    parser.add_argument("--v8-closeout", type=Path, required=True)
    parser.add_argument("--v8-verification", type=Path, required=True)
    parser.add_argument("--correction-output", type=Path, required=True)
    parser.add_argument("--methods-output", type=Path, required=True)
    for name in (
        "registration", "protocol", "training_closeout", "training_verification",
        "v9_policies", "v8_methods", "v8_closeout", "v8_verification",
    ):
        parser.add_argument(f"--expected-{name.replace('_', '-')}-sha256", required=True)
    return parser.parse_args()


def bind(path: Path, expected: str) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected:
        raise ValueError(f"identity drift: {resolved}")
    return resolved, json.loads(resolved.read_text())


def main() -> int:
    args = parse_args()
    registration_path, registration = bind(args.registration, args.expected_registration_sha256)
    protocol_path, protocol = bind(args.protocol, args.expected_protocol_sha256)
    closeout_path, closeout = bind(args.training_closeout, args.expected_training_closeout_sha256)
    verification_path, verification = bind(
        args.training_verification, args.expected_training_verification_sha256
    )
    v9_policies_path, v9_payload = bind(args.v9_policies, args.expected_v9_policies_sha256)
    v8_methods_path, v8_payload = bind(args.v8_methods, args.expected_v8_methods_sha256)
    v8_closeout_path, v8_closeout = bind(args.v8_closeout, args.expected_v8_closeout_sha256)
    v8_verification_path, v8_verification = bind(
        args.v8_verification, args.expected_v8_verification_sha256
    )
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("formal_held_out_outcomes_read") is not False
        or verification.get("passed") is not True
        or v8_closeout.get("artifact_integrity_passed") is not True
        or v8_verification.get("passed") is not True
        or registration.get("development_evaluation", {}).get("common_online_event_budgets")
        != list(COMMON_EVENTS)
        or registration.get("development_evaluation", {}).get("unique_policies_after_training") != 45
        or registration.get("development_evaluation", {}).get("unique_remote_sumo_episodes") != 270
    ):
        raise ValueError("evaluation count-correction inputs drifted")

    source_rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for raw in v8_payload.get("methods", []):
        algorithm = raw.get("algorithm")
        seed = int(raw.get("training_seed", -1))
        events = int(raw.get("online_events", -1))
        if algorithm in {"h2oplus", "sac-online"} and seed in SEEDS and events in {0, 10_000, 25_000}:
            checkpoint = resolve(v8_methods_path, raw["checkpoint"])
            row = dict(raw)
            row.update({
                "checkpoint": checkpoint,
                "artifact_lineage": "v8-fewshot",
                "actor_tensor_sha256": actor_tensor_sha256(checkpoint, "v8-fewshot"),
                "source_protocol_sha256": v8_payload["fewshot_protocol_sha256"],
            })
            source_rows[(algorithm, seed, events)] = row
    for raw in v9_payload.get("policies", []):
        algorithm = raw.get("algorithm")
        seed = int(raw.get("training_seed", -1))
        events = int(raw.get("online_events", -1))
        if algorithm in ALGORITHMS and seed in SEEDS and events in COMMON_EVENTS:
            checkpoint = resolve(v9_policies_path, raw["checkpoint"])
            row = dict(raw)
            row.update({
                "checkpoint": checkpoint,
                "artifact_lineage": "v9-adaptive",
                "source_protocol_sha256": args.expected_protocol_sha256,
            })
            if actor_tensor_sha256(checkpoint, "v9-adaptive") != row["actor_tensor_sha256"]:
                raise ValueError(f"v9 actor tensor identity drifted: {checkpoint}")
            source_rows[(algorithm, seed, events)] = row

    aliases = []
    physical_by_hash: dict[tuple[int, str], dict[str, Any]] = {}
    identity_evidence = []
    for seed in SEEDS:
        # The v8 manifests deliberately omit 5k because it is tensor-identical to 0.
        for algorithm in ("h2oplus", "sac-online"):
            source_rows[(algorithm, seed, 5_000)] = dict(source_rows[(algorithm, seed, 0)])
        for algorithm in ALGORITHMS:
            for events in COMMON_EVENTS:
                source = source_rows[(algorithm, seed, events)]
                tensor_hash = source["actor_tensor_sha256"]
                key = (seed, tensor_hash)
                physical = physical_by_hash.get(key)
                if physical is None:
                    physical = dict(source)
                    physical["name"] = source["name"]
                    physical["physical_online_events"] = int(source["online_events"])
                    physical["checkpoint"] = source["checkpoint"]
                    physical_by_hash[key] = physical
                aliases.append({
                    "name": f"{algorithm}-seed{seed}-events{events:09d}",
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "online_events": events,
                    "physical_method": physical["name"],
                    "actor_tensor_sha256": tensor_hash,
                })
        seed_aliases = [item for item in aliases if item["training_seed"] == seed]
        seed_physical = [item for (item_seed, _), item in physical_by_hash.items() if item_seed == seed]
        identity_evidence.append({
            "training_seed": seed,
            "method_budget_aliases": len(seed_aliases),
            "unique_actor_tensors": len(seed_physical),
            "duplicate_groups": sorted(
                [
                    sorted(item["name"] for item in seed_aliases if item["actor_tensor_sha256"] == tensor_hash)
                    for item_seed, tensor_hash in physical_by_hash
                    if item_seed == seed
                    and sum(
                        item["actor_tensor_sha256"] == tensor_hash for item in seed_aliases
                    ) > 1
                ]
            ),
        })

    physical = sorted(physical_by_hash.values(), key=lambda item: item["name"])
    if len(aliases) != 60 or len(physical) != 42:
        raise ValueError(f"corrected policy inventory drifted: aliases={len(aliases)} physical={len(physical)}")
    if any(item["method_budget_aliases"] != 20 or item["unique_actor_tensors"] != 14 for item in identity_evidence):
        raise ValueError("per-seed actor identity arithmetic drifted")

    correction_path = args.correction_output.expanduser().resolve()
    correction = {
        "schema_version": CORRECTION_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "pre-evaluation actor-identity arithmetic correction only",
        "reason": (
            "the frozen registration counted 45 policies, but exact tensor identities across "
            "the five method curves yield 60 budget aliases and 42 distinct actors"
        ),
        "performance_or_evaluation_outcomes_read": False,
        "formal_held_out_outcomes_read": False,
        "training_or_policy_artifacts_changed": False,
        "registered_unique_policies": 45,
        "registered_remote_sumo_episodes": 270,
        "corrected_method_budget_aliases": 60,
        "corrected_unique_actor_tensors": 42,
        "development_blocks": 2,
        "demand_scales": 3,
        "corrected_remote_sumo_episodes": 252,
        "alias_reuse_rule": (
            "one deterministic SUMO result is reused only for aliases whose compact actor "
            "tensor SHA256 is exactly equal within the same training seed"
        ),
        "identity_evidence": identity_evidence,
        "bindings": {
            "registration": descriptor(registration_path),
            "protocol": descriptor(protocol_path),
            "training_closeout": descriptor(closeout_path),
            "training_verification": descriptor(verification_path),
            "v9_training_policies": descriptor(v9_policies_path),
            "v8_methods": descriptor(v8_methods_path),
            "v8_training_closeout": descriptor(v8_closeout_path),
            "v8_training_verification": descriptor(v8_verification_path),
        },
    }
    atomic_json(correction_path, correction)

    methods_path = args.methods_output.expanduser().resolve()
    methods_payload = {
        "schema_version": METHODS_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "selection_rule": (
            "all common-budget method aliases fixed before evaluation; physical evaluation "
            "deduplicated only by exact actor tensor SHA256"
        ),
        "formal_held_out_outcomes_read": False,
        "v9_protocol_sha256": args.expected_protocol_sha256,
        "count_correction": descriptor(correction_path),
        "method_budget_alias_count": len(aliases),
        "physical_method_count": len(physical),
        "aliases": sorted(aliases, key=lambda item: item["name"]),
        "methods": [],
    }
    for raw in physical:
        row = {
            key: raw[key]
            for key in (
                "name", "algorithm", "training_seed", "type", "training_run_signature",
                "training_admission_sha256", "training_checkpoint_sha256",
                "actual_online_events", "learner_update_steps", "ratio_optimizer_steps",
                "actor_tensor_sha256", "artifact_lineage", "source_protocol_sha256",
            )
            if key in raw
        }
        row["online_events"] = int(raw["physical_online_events"])
        row["checkpoint"] = os.path.relpath(raw["checkpoint"], methods_path.parent)
        methods_payload["methods"].append(row)
    atomic_json(methods_path, methods_payload)

    print(json.dumps({
        "count_correction_sha256": sha256(correction_path),
        "methods_sha256": sha256(methods_path),
        "aliases": len(aliases),
        "physical_methods": len(physical),
        "remote_sumo_episodes": len(physical) * 2 * 3,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
