#!/usr/bin/env python3
"""Freeze the logical-to-physical actor map for v12 development evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
STUDY_ROOT = REPRO / "h2oplus_v12_staged_release_results"
SEEDS = tuple(range(6201, 6206))
EVENTS = (0, 5_000, 10_000, 25_000)
VARIANTS = (
    "abrupt-control",
    "actor-ramp",
    "temperature-ramp",
    "actor-anchor",
    "staged-combined",
)
SCHEMA = "transit-evaluation-methods-v12-staged-release-v1"
SELECTION_RULE = (
    "all preregistered common-budget aliases fixed before evaluation; physical "
    "evaluation deduplicated only by exact actor tensor SHA256 within a seed"
)
V11_METHODS_SHA256 = (
    "6716c56c73e9d491431b1ab019727f9bfd18a2242cb7500826cf7ccc0628b260"
)
V12_PROTOCOL_SHA256 = (
    "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c"
)
V12_CLOSEOUT_SHA256 = (
    "998f47488b50b2cd40f9a6ddd5a399712bb9208cb1ef883cad1181f800385b66"
)
V12_VERIFICATION_SHA256 = (
    "6835e3c545be22c25bd9e3c7eae7ef987bb1864e027c4fef9849788f744868c8"
)
V12_PREFIX_CLOSEOUT_SHA256 = (
    "8bb303dbe038089422ba88f28bf06cd47a672b71f929d1c14306679b943d0ef1"
)
V12_PREFIX_VERIFICATION_SHA256 = (
    "b13b7f0827326f52c6c9f512a7385e71e5abf40a8f15df0e70d5ec26126242e7"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def descriptor(path: Path, owner: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": (
            os.path.relpath(resolved, owner.parent) if owner is not None else str(resolved)
        ),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def load_bound(path: Path, expected: str, schema: str) -> tuple[Path, dict[str, Any]]:
    resolved = path.expanduser().resolve(strict=True)
    if sha256(resolved) != expected:
        raise ValueError(f"frozen identity drifted: {resolved}")
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != schema:
        raise ValueError(f"frozen schema drifted: {resolved}")
    return resolved, payload


def resolve(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def load_v12_policy(
    path: Path,
    *,
    seed: int,
    events: int,
    variant: str,
    stage: str,
    admission_sha256: str,
) -> tuple[dict[str, Any], str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("artifact_schema") != "transit-policy-artifact-v7-v1":
        raise ValueError(f"v12 policy schema drifted: {path}")
    extra = payload.get("extra", {})
    actor_sha = actor_digest(payload.get("actor", {}))
    expected_phase = {
        0: "offline_initializer_loaded",
        5_000: "online_warmup",
        10_000: "complete",
        25_000: "complete",
    }[events]
    expected_steps = {
        0: (50_000, 0),
        5_000: (50_000, 0),
        10_000: (55_000, 2_250),
        25_000: (70_000, 6_000),
    }[events]
    expected = {
        "method": "h2oplus",
        "variant": variant,
        "stage": stage,
        "phase": expected_phase,
        "training_seed": seed,
        "v12_online_events_nominal": events,
        "v12_learner_update_steps": expected_steps[0],
        "v12_ratio_optimizer_steps": expected_steps[1],
        "v12_actor_tensor_sha256": actor_sha,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {
        key: (extra.get(key), value)
        for key, value in expected.items()
        if extra.get(key) != value
    }
    admission = extra.get("training_admission", {})
    intervention = extra.get("v12_intervention", {})
    flags = {
        "abrupt-control": (False, False, False),
        "actor-ramp": (True, False, False),
        "temperature-ramp": (False, True, False),
        "actor-anchor": (False, False, True),
        "staged-combined": (True, True, True),
        "shared-prefix": (False, False, False),
    }[variant]
    if (
        mismatches
        or admission.get("schema_version")
        != "transit-training-admission-v12-staged-release-development-v2"
        or admission.get("sha256") != admission_sha256
        or extra.get("v12_protocol", {}).get("sha256") != V12_PROTOCOL_SHA256
        or intervention.get("schema_version")
        != "transit-h2oplus-v12-staged-release-variant-v1"
        or intervention.get("variant") != variant
        or intervention.get("actor_learning_rate_ramp") is not flags[0]
        or intervention.get("temperature_learning_rate_ramp") is not flags[1]
        or intervention.get("actor_state_anchor") is not flags[2]
        or intervention.get("unit_ratio_weights") is not True
        or intervention.get("checkpoint_selection_by_behavior") is not False
        or not isinstance(extra.get("v12_temperature_tensor_sha256"), str)
        or len(extra["v12_temperature_tensor_sha256"]) != 64
    ):
        raise ValueError(f"v12 policy provenance drifted: {path}: {mismatches}")
    return extra, actor_sha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v11-methods",
        type=Path,
        default=REPRO / "h2oplus_v11_actor_hold_evaluation_methods_v1.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_protocol_v2.json",
    )
    parser.add_argument(
        "--prefix-closeout",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_prefix_closeout_v2.json",
    )
    parser.add_argument(
        "--prefix-verification",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_prefix_closeout_verification_v2.json",
    )
    parser.add_argument(
        "--branch-closeout",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_branch_closeout_v2.json",
    )
    parser.add_argument(
        "--branch-verification",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_branch_closeout_verification_v2.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v12_staged_release_evaluation_methods_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v11_path, v11 = load_bound(
        args.v11_methods,
        V11_METHODS_SHA256,
        "transit-evaluation-methods-v11-actor-hold-v1",
    )
    protocol_path, protocol = load_bound(
        args.protocol,
        V12_PROTOCOL_SHA256,
        "transit-h2oplus-v12-staged-release-protocol-v2",
    )
    prefix_closeout_path, prefix_closeout = load_bound(
        args.prefix_closeout,
        V12_PREFIX_CLOSEOUT_SHA256,
        "transit-h2oplus-v12-staged-release-prefix-closeout-v2",
    )
    prefix_verification_path, prefix_verification = load_bound(
        args.prefix_verification,
        V12_PREFIX_VERIFICATION_SHA256,
        "transit-h2oplus-v12-staged-release-prefix-closeout-independent-verification-v2",
    )
    closeout_path, closeout = load_bound(
        args.branch_closeout,
        V12_CLOSEOUT_SHA256,
        "transit-h2oplus-v12-staged-release-branch-closeout-v2",
    )
    verification_path, verification = load_bound(
        args.branch_verification,
        V12_VERIFICATION_SHA256,
        "transit-h2oplus-v12-staged-release-branch-closeout-verification-v2",
    )
    if (
        v11.get("physical_method_count") != 45
        or v11.get("logical_method_budget_alias_count") != 80
        or v11.get("development_evaluation_outcomes_read") is not False
        or protocol.get("development_evaluation_submitted") is not False
        or prefix_closeout.get("artifact_integrity_passed") is not True
        or prefix_closeout.get("scientific_stability_passed_for_all_five_prefix_runs")
        is not True
        or prefix_verification.get("artifact_integrity_passed") is not True
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("scientific_stability_passed_for_all_30_training_runs")
        is not True
        or closeout.get("development_evaluation_gate_open") is not True
        or verification.get("artifact_integrity_passed") is not True
        or verification.get("scientific_stability_passed_for_all_30_training_runs")
        is not True
        or verification.get("development_evaluation_gate_open") is not True
        or any(
            payload.get("formal_held_out_outcomes_read") is not False
            for payload in (
                v11,
                protocol,
                prefix_closeout,
                prefix_verification,
                closeout,
                verification,
            )
        )
    ):
        raise ValueError("sealed controls or v12 training gate drifted")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite methods manifest: {output}")

    old_methods = {}
    physical_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in v11["methods"]:
        row = dict(raw)
        checkpoint = resolve(v11_path, row["checkpoint"])
        if sha256(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"v11 physical checkpoint drifted: {row['name']}")
        row["checkpoint"] = checkpoint
        old_methods[row["name"]] = row
        key = (int(row["training_seed"]), row["actor_tensor_sha256"])
        if key in physical_by_key:
            raise ValueError("v11 manifest contains duplicate physical actor tensors")
        physical_by_key[key] = row
    aliases = [dict(row) for row in v11["aliases"]]
    identity_evidence = []

    plan = json.loads(
        (REPRO / "h2oplus_v12_staged_release_training_plan_v2.json").read_text()
    )
    jobs = {
        (job["variant"], int(job["seed"])): job
        for job in plan["jobs"]
    }
    for seed in SEEDS:
        initializer_alias = next(
            row
            for row in aliases
            if row["algorithm"] == "h2oplus"
            and row["training_seed"] == seed
            and row["online_events"] == 0
        )
        initializer_key = (seed, initializer_alias["actor_tensor_sha256"])
        initializer = physical_by_key[initializer_key]
        prefix_job = jobs[("shared-prefix", seed)]
        prefix_dir = STUDY_ROOT / "prefix_v2" / prefix_job["name"]
        prefix_admission = (
            REPRO
            / "h2oplus_v12_staged_release_training_admissions_v2"
            / f"{prefix_job['name']}.json"
        )
        prefix_actor_hashes = set()
        prefix_temperature_hashes = set()
        prefix_sources = []
        for events in EVENTS[:3]:
            policy = prefix_dir / "v12_policies" / f"online_events_{events:09d}.pt"
            extra, actor_sha = load_v12_policy(
                policy,
                seed=seed,
                events=events,
                variant="shared-prefix",
                stage="prefix",
                admission_sha256=sha256(prefix_admission),
            )
            if actor_sha != initializer["actor_tensor_sha256"]:
                raise ValueError(f"v12 prefix actor differs from fixed H2O: seed {seed}")
            prefix_actor_hashes.add(actor_sha)
            prefix_temperature_hashes.add(extra["v12_temperature_tensor_sha256"])
            prefix_sources.append(
                {
                    "online_events": events,
                    "actual_online_events": int(extra["v12_online_events_actual"]),
                    "actor_tensor_sha256": actor_sha,
                    "temperature_tensor_sha256": extra[
                        "v12_temperature_tensor_sha256"
                    ],
                    "policy": descriptor(policy, output),
                }
            )
        if len(prefix_actor_hashes) != 1 or len(prefix_temperature_hashes) != 1:
            raise ValueError(f"v12 prefix actor/temperature hold drifted: seed {seed}")

        branch_sources = []
        for variant in VARIANTS:
            branch_job = jobs[(variant, seed)]
            branch_dir = STUDY_ROOT / "branches_v2" / branch_job["name"]
            admission = (
                REPRO
                / "h2oplus_v12_staged_release_training_admissions_v2"
                / f"{branch_job['name']}.json"
            )
            policy = branch_dir / "v12_policies/online_events_000025000.pt"
            extra, actor_sha = load_v12_policy(
                policy,
                seed=seed,
                events=25_000,
                variant=variant,
                stage="branch",
                admission_sha256=sha256(admission),
            )
            algorithm = f"h2oplus-v12-{variant}"
            physical_name = f"{algorithm}-seed{seed}-events000025000"
            row = {
                "name": physical_name,
                "algorithm": algorithm,
                "variant": variant,
                "training_seed": seed,
                "type": "checkpoint",
                "checkpoint": policy.resolve(strict=True),
                "checkpoint_sha256": sha256(policy),
                "training_run_signature": extra["run_signature"],
                "training_admission_sha256": sha256(admission),
                "training_checkpoint_sha256": extra[
                    "training_checkpoint_sha256"
                ],
                "online_events": 25_000,
                "actual_online_events": int(extra["v12_online_events_actual"]),
                "learner_update_steps": int(extra["v12_learner_update_steps"]),
                "ratio_optimizer_steps": int(extra["v12_ratio_optimizer_steps"]),
                "actor_tensor_sha256": actor_sha,
                "temperature_tensor_sha256": extra[
                    "v12_temperature_tensor_sha256"
                ],
                "artifact_lineage": "v12-staged-release",
                "source_protocol_sha256": V12_PROTOCOL_SHA256,
            }
            key = (seed, actor_sha)
            existing = physical_by_key.get(key)
            if existing is None:
                physical_by_key[key] = row
                physical = row
            else:
                physical = existing
            branch_sources.append(
                {
                    "variant": variant,
                    "physical_method": physical["name"],
                    "actor_tensor_sha256": actor_sha,
                    "temperature_tensor_sha256": row[
                        "temperature_tensor_sha256"
                    ],
                    "policy": descriptor(policy, output),
                }
            )

            for events in EVENTS:
                physical_for_alias = initializer if events < 25_000 else physical
                alias = {
                    "name": f"{algorithm}-seed{seed}-events{events:09d}",
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "online_events": events,
                    "physical_method": physical_for_alias["name"],
                    "actor_tensor_sha256": physical_for_alias[
                        "actor_tensor_sha256"
                    ],
                    "source_artifact_lineage": "v12-staged-release",
                    "source_variant": variant,
                    "source_checkpoint_sha256": (
                        prefix_sources[EVENTS.index(events)]["policy"]["sha256"]
                        if events < 25_000
                        else row["checkpoint_sha256"]
                    ),
                    "source_temperature_tensor_sha256": (
                        prefix_sources[EVENTS.index(events)][
                            "temperature_tensor_sha256"
                        ]
                        if events < 25_000
                        else row["temperature_tensor_sha256"]
                    ),
                }
                aliases.append(alias)

        identity_evidence.append(
            {
                "training_seed": seed,
                "v12_logical_aliases": 20,
                "prefix_zero_5k_10k_equal_fixed_h2oplus_zero": True,
                "prefix_zero_5k_10k_temperature_equal": True,
                "prefix_source_policies": prefix_sources,
                "branch_source_policies": branch_sources,
            }
        )

    physical = sorted(physical_by_key.values(), key=lambda row: row["name"])
    if len(aliases) != 180 or len(physical) > 180:
        raise ValueError(
            f"v12 identity arithmetic drifted: {len(aliases)} aliases, "
            f"{len(physical)} actors"
        )
    if len({row["name"] for row in aliases}) != 180:
        raise ValueError("duplicate logical aliases")
    if len({(row["training_seed"], row["actor_tensor_sha256"]) for row in physical}) != len(physical):
        raise ValueError("physical actors were not exactly deduplicated")
    if len(physical) * 9 > 1620:
        raise ValueError("registered maximum remote SUMO budget exceeded")

    payload = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "selection_rule": SELECTION_RULE,
        "formal_held_out_outcomes_read": False,
        "development_evaluation_outcomes_read": False,
        "v8_protocol_sha256": v11["v8_protocol_sha256"],
        "v10_protocol_sha256": v11["v10_protocol_sha256"],
        "v11_protocol_sha256": v11["v11_protocol_sha256"],
        "v12_protocol_sha256": V12_PROTOCOL_SHA256,
        "logical_method_budget_alias_count": len(aliases),
        "physical_method_count": len(physical),
        "expected_remote_sumo_episodes": len(physical) * 9,
        "registered_maximum_remote_sumo_episodes": 1620,
        "identity_evidence": identity_evidence,
        "bindings": {
            "v11_methods": descriptor(v11_path, output),
            "v12_protocol": descriptor(protocol_path, output),
            "v12_prefix_closeout": descriptor(prefix_closeout_path, output),
            "v12_prefix_verification": descriptor(
                prefix_verification_path, output
            ),
            "v12_branch_closeout": descriptor(closeout_path, output),
            "v12_branch_verification": descriptor(verification_path, output),
        },
        "aliases": sorted(aliases, key=lambda row: row["name"]),
        "methods": [],
    }
    for raw in physical:
        row = dict(raw)
        row["checkpoint"] = os.path.relpath(Path(raw["checkpoint"]), output.parent)
        payload["methods"].append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "methods_sha256": sha256(output),
                "logical_aliases": len(aliases),
                "physical_actors": len(physical),
                "remote_sumo_episodes": len(physical) * 9,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
