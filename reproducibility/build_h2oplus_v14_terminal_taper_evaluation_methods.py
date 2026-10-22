#!/usr/bin/env python3
"""Freeze the logical-to-physical actor map for v14 development evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
REPRO = ROOT / "reproducibility"
STUDY_ROOT = REPRO / "h2oplus_v14_terminal_taper_results"
BRANCH_ROOT = STUDY_ROOT / "branches_v1"
DEPLOYMENT = REPRO / "deployments/h2oplus_v14_terminal_taper_pilot_v1"
CONTROL_ROOT = DEPLOYMENT / "workspace/reproducibility/h2oplus_v14_control_v1"
HANDOFF_ROOT = DEPLOYMENT / "workspace/reproducibility/h2oplus_v14_source_handoffs_v1"
SEEDS = tuple(range(6201, 6206))
EVENTS = (0, 5_000, 10_000, 25_000)
VARIANTS = ("terminal-half", "terminal-quarter", "terminal-zero")
CONTROL_ALGORITHMS = (
    "h2oplus",
    "sac-online",
    "h2oplus-v10-delayed-actor",
    "h2oplus-v11-actor-hold",
    "h2oplus-v12-actor-ramp",
    "h2oplus-v13-proximal-strong",
)
SCHEMA = "transit-evaluation-methods-v14-terminal-taper-v1"
SELECTION_RULE = (
    "all preregistered common-budget aliases fixed before evaluation; physical "
    "evaluation deduplicated only by exact actor tensor SHA256 within a seed"
)
EXPECTED = {
    "v13_methods": "712cea480fe1c35bfee4186fbbea137682b8515888d40a05870a55f695caa104",
    "protocol": "ba275fc4c134feadb564b49548699571ca5bf6bfd1aee529759d6ce791df12c5",
    "plan": "71e43c0a6ec063f2b96459388835e0b29277695d1714f375d5115921cbd8c831",
    "closeout": "5d609f76f6d139dc8982a0ce4f8c9e95a4f0e5f4c4d9655b3545fa15270d2ffc",
    "verification": "56cf282615364dc1b93bbf118cd1f3101b20fa2398ca87e08b202322c833c15b",
    "audit": "51296e4eb91cf785dd484ae726f172d6828b16b1aeaa8af8469a6356b45ea136",
    "file_manifest": "bb1e4bb04e437091c137595399abfae1d4f80294b37a471c39ee53fe3eaae558",
}
TERMINAL_MULTIPLIERS = {
    "terminal-half": 0.5,
    "terminal-quarter": 0.25,
    "terminal-zero": 0.0,
}
FULL_RATE_EQUIVALENT_UPDATES = {
    "terminal-half": 10249.449999999913,
    "terminal-quarter": 8999.449999999915,
    "terminal-zero": 7749.449999999926,
}


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
        "path": os.path.relpath(resolved, owner.parent) if owner else str(resolved),
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


def load_v14_policy(path: Path, job: dict, admission_path: Path) -> tuple[dict, str]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("artifact_schema") != "transit-policy-artifact-v7-v1":
        raise ValueError(f"v14 policy schema drifted: {path}")
    extra = payload.get("extra", {})
    actor_sha = actor_digest(payload.get("actor", {}))
    admission_sha = sha256(admission_path)
    expected = {
        "method": "h2oplus",
        "variant": job["variant"],
        "stage": "branch",
        "phase": "complete",
        "training_seed": int(job["seed"]),
        "run_signature": job["run_signature"],
        "v14_online_events_nominal": 25_000,
        "v14_learner_update_steps": 70_000,
        "v14_ratio_optimizer_steps": 6_000,
        "v14_actor_tensor_sha256": actor_sha,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {
        key: (extra.get(key), value)
        for key, value in expected.items()
        if extra.get(key) != value
    }
    admission = extra.get("training_admission", {})
    selected = admission.get("selected_arguments", {})
    source = selected.get("source_handoff", {})
    intervention = extra.get("v14_intervention", {})
    multiplier = TERMINAL_MULTIPLIERS[job["variant"]]
    expected_actual = 9_999 if int(job["seed"]) == 6205 else 10_000
    if (
        mismatches
        or admission.get("schema_version")
        != "transit-training-admission-v14-terminal-taper-development-v1"
        or admission.get("sha256") != admission_sha
        or extra.get("v14_protocol", {}).get("sha256") != EXPECTED["protocol"]
        or source.get("online_events_actual") != expected_actual
        or source.get("v12_protocol_sha256")
        != "ed6cd5e32f4719a0ef8925272688fdb3393e5c5056027ae9d6430b1ba20d5f1c"
        or intervention.get("schema_version")
        != "transit-h2oplus-v14-terminal-taper-variant-v1"
        or intervention.get("variant") != job["variant"]
        or intervention.get("proximal_rate_per_actor_update") != 5e-5
        or intervention.get("registered_post_10k_actor_updates") != 15_000
        or not math.isclose(
            float(intervention.get("no_gradient_anchor_retention_at_25k", math.nan)),
            (1.0 - 5e-5) ** 15_000,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or intervention.get("predesignated_candidate")
        is not (job["variant"] == "terminal-quarter")
        or intervention.get("terminal_actor_lr_taper_events") != [15_000, 25_000]
        or intervention.get("terminal_actor_lr_taper_critic_updates") != [10_000, 19_999]
        or intervention.get("terminal_actor_lr_multiplier") != multiplier
        or not math.isclose(
            float(intervention.get("full_rate_equivalent_actor_updates", math.nan)),
            FULL_RATE_EQUIVALENT_UPDATES[job["variant"]],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or intervention.get("unit_ratio_weights") is not True
        or intervention.get("checkpoint_selection_by_behavior") is not False
        or not isinstance(extra.get("v14_temperature_tensor_sha256"), str)
        or len(extra["v14_temperature_tensor_sha256"]) != 64
    ):
        raise ValueError(f"v14 policy provenance drifted: {path}: {mismatches}")
    return extra, actor_sha


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v13-methods",
        type=Path,
        default=REPRO / "h2oplus_v13_proximal_release_evaluation_methods_v1.json",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_protocol_v1.json",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_training_plan_v1.json",
    )
    parser.add_argument(
        "--branch-closeout",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_branch_closeout_v1.json",
    )
    parser.add_argument(
        "--branch-verification",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_branch_closeout_verification_v1.json",
    )
    parser.add_argument(
        "--branch-audit",
        type=Path,
        default=STUDY_ROOT / "branch_audit_v1.json",
    )
    parser.add_argument(
        "--branch-file-manifest",
        type=Path,
        default=STUDY_ROOT / "branch_file_manifest_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPRO / "h2oplus_v14_terminal_taper_evaluation_methods_v1.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v13_path, v13 = load_bound(
        args.v13_methods,
        EXPECTED["v13_methods"],
        "transit-evaluation-methods-v13-proximal-release-v1",
    )
    protocol_path, protocol = load_bound(
        args.protocol,
        EXPECTED["protocol"],
        "transit-h2oplus-v14-terminal-taper-protocol-v1",
    )
    plan_path, plan = load_bound(
        args.plan,
        EXPECTED["plan"],
        "transit-h2oplus-v14-terminal-taper-training-plan-v1",
    )
    closeout_path, closeout = load_bound(
        args.branch_closeout,
        EXPECTED["closeout"],
        "transit-h2oplus-v14-terminal-taper-branch-closeout-v1",
    )
    verification_path, verification = load_bound(
        args.branch_verification,
        EXPECTED["verification"],
        "transit-h2oplus-v14-terminal-taper-branch-closeout-verification-v1",
    )
    audit_path, audit = load_bound(
        args.branch_audit,
        EXPECTED["audit"],
        "transit-h2oplus-v14-terminal-taper-branch-audit-v1",
    )
    manifest_path, file_manifest = load_bound(
        args.branch_file_manifest,
        EXPECTED["file_manifest"],
        "transit-h2oplus-v14-terminal-taper-branch-file-manifest-v1",
    )
    if (
        v13.get("physical_method_count") != 65
        or v13.get("logical_method_budget_alias_count") != 160
        or v13.get("development_evaluation_outcomes_read") is not False
        or protocol.get("development_evaluation_submitted") is not False
        or closeout.get("status") != "passed"
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("scientific_stability_passed_for_all_15_branches") is not True
        or closeout.get("exact_seed_matched_v12_10k_handoffs_passed") is not True
        or closeout.get("development_evaluation_gate_open") is not True
        or verification.get("passed") is not True
        or verification.get("all_15_scientific_stability_passed") is not True
        or verification.get("exact_seed_matched_v12_10k_handoffs_passed") is not True
        or verification.get("development_evaluation_gate_open") is not True
        or audit.get("passed") is not True
        or audit.get("artifact_integrity_passed") is not True
        or audit.get("all_15_stability_passed") is not True
        or audit.get("all_seed_matched_v12_10k_handoffs_exact") is not True
        or file_manifest.get("file_count") != 255
        or any(
            payload.get("formal_held_out_outcomes_read") is not False
            for payload in (v13, protocol, closeout, verification, audit, file_manifest)
        )
    ):
        raise ValueError("sealed controls or v14 training gate drifted")

    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite methods manifest: {output}")

    aliases = [
        dict(row)
        for row in v13["aliases"]
        if row.get("algorithm") in CONTROL_ALGORITHMS
    ]
    referenced = {row["physical_method"] for row in aliases}
    physical_by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in v13["methods"]:
        if raw["name"] not in referenced:
            continue
        row = dict(raw)
        checkpoint = resolve(v13_path, row["checkpoint"])
        if sha256(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"v12 control checkpoint drifted: {row['name']}")
        row["checkpoint"] = checkpoint
        key = (int(row["training_seed"]), row["actor_tensor_sha256"])
        if key in physical_by_key:
            raise ValueError("selected controls contain duplicate physical actors")
        physical_by_key[key] = row
    if len(aliases) != 120 or len(physical_by_key) != 55:
        raise ValueError("v14 control subset arithmetic drifted")

    old_identity = {int(row["training_seed"]): row for row in v13["identity_evidence"]}
    jobs = {(job["variant"], int(job["seed"])): job for job in plan["jobs"]}
    identity_evidence = []
    for seed in SEEDS:
        initializer_alias = next(
            row
            for row in aliases
            if row["algorithm"] == "h2oplus"
            and row["training_seed"] == seed
            and row["online_events"] == 0
        )
        initializer = physical_by_key[(seed, initializer_alias["actor_tensor_sha256"])]
        inherited_prefix = old_identity[seed]["prefix_source_policies"]
        if len(inherited_prefix) != 3:
            raise ValueError(f"v12 prefix witnesses drifted: seed {seed}")
        prefix_sources = []
        for witness in inherited_prefix:
            policy = resolve(v13_path, witness["policy"]["path"])
            if sha256(policy) != witness["policy"]["sha256"]:
                raise ValueError(f"v12 prefix witness drifted: {policy}")
            if witness["actor_tensor_sha256"] != initializer["actor_tensor_sha256"]:
                raise ValueError(f"v14 prefix actor identity drifted: seed {seed}")
            prefix_sources.append(
                {
                    **{key: value for key, value in witness.items() if key != "policy"},
                    "policy": descriptor(policy, output),
                }
            )
        prefix_sources.sort(key=lambda row: row["online_events"])

        branch_sources = []
        for variant in VARIANTS:
            job = jobs[(variant, seed)]
            run = BRANCH_ROOT / job["name"]
            admission = CONTROL_ROOT / "admissions" / f"{job['name']}.json"
            policy = run / "v14_policies/online_events_000025000.pt"
            extra, actor_sha = load_v14_policy(policy, job, admission)
            source = extra["training_admission"]["selected_arguments"]["source_handoff"]
            handoff_manifest = HANDOFF_ROOT / f"seed{seed}/v12_handoff/manifest.json"
            if sha256(handoff_manifest) != source["manifest_sha256"]:
                raise ValueError(f"v14 source handoff drifted: seed {seed}")
            algorithm = f"h2oplus-v14-{variant}"
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
                "training_checkpoint_sha256": extra["training_checkpoint_sha256"],
                "online_events": 25_000,
                "actual_online_events": int(extra["v14_online_events_actual"]),
                "learner_update_steps": int(extra["v14_learner_update_steps"]),
                "ratio_optimizer_steps": int(extra["v14_ratio_optimizer_steps"]),
                "actor_tensor_sha256": actor_sha,
                "temperature_tensor_sha256": extra["v14_temperature_tensor_sha256"],
                "artifact_lineage": "v14-terminal-taper",
                "source_protocol_sha256": EXPECTED["protocol"],
                "source_handoff_manifest_sha256": source["manifest_sha256"],
                "source_handoff_checkpoint_state_sha256": source[
                    "checkpoint_state_sha256"
                ],
                "base_proximal_rate": 5e-5,
                "terminal_actor_lr_multiplier": TERMINAL_MULTIPLIERS[variant],
                "full_rate_equivalent_actor_updates": FULL_RATE_EQUIVALENT_UPDATES[variant],
                "predesignated_candidate": variant == "terminal-quarter",
            }
            key = (seed, actor_sha)
            if key in physical_by_key:
                raise ValueError(f"v14 terminal actor unexpectedly duplicates a control: {physical_name}")
            physical_by_key[key] = row
            branch_sources.append(
                {
                    "variant": variant,
                    "physical_method": physical_name,
                    "actor_tensor_sha256": actor_sha,
                    "temperature_tensor_sha256": row["temperature_tensor_sha256"],
                    "base_proximal_rate": row["base_proximal_rate"],
                    "terminal_actor_lr_multiplier": row["terminal_actor_lr_multiplier"],
                    "full_rate_equivalent_actor_updates": row[
                        "full_rate_equivalent_actor_updates"
                    ],
                    "predesignated_candidate": row["predesignated_candidate"],
                    "source_handoff_manifest": descriptor(handoff_manifest, output),
                    "policy": descriptor(policy, output),
                }
            )
            for events in EVENTS:
                physical = initializer if events < 25_000 else row
                source_policy = prefix_sources[EVENTS.index(events)] if events < 25_000 else None
                aliases.append(
                    {
                        "name": f"{algorithm}-seed{seed}-events{events:09d}",
                        "algorithm": algorithm,
                        "training_seed": seed,
                        "online_events": events,
                        "physical_method": physical["name"],
                        "actor_tensor_sha256": physical["actor_tensor_sha256"],
                        "source_artifact_lineage": "v14-terminal-taper",
                        "source_variant": variant,
                        "source_checkpoint_sha256": (
                            source_policy["policy"]["sha256"]
                            if source_policy is not None
                            else row["checkpoint_sha256"]
                        ),
                        "source_temperature_tensor_sha256": (
                            source_policy["temperature_tensor_sha256"]
                            if source_policy is not None
                            else row["temperature_tensor_sha256"]
                        ),
                        "source_handoff_manifest_sha256": source[
                            "manifest_sha256"
                        ],
                        "predesignated_candidate": variant == "terminal-quarter",
                    }
                )
        identity_evidence.append(
            {
                "training_seed": seed,
                "v14_logical_aliases": 12,
                "prefix_zero_5k_10k_equal_fixed_h2oplus_zero": True,
                "exact_seed_matched_v12_10k_handoff": True,
                "prefix_source_policies": prefix_sources,
                "branch_source_policies": branch_sources,
            }
        )

    physical = sorted(physical_by_key.values(), key=lambda row: row["name"])
    if len(aliases) != 180 or len(physical) != 70:
        raise ValueError(
            f"v14 identity arithmetic drifted: {len(aliases)} aliases, {len(physical)} actors"
        )
    if len({row["name"] for row in aliases}) != 180:
        raise ValueError("duplicate logical aliases")
    if len({(row["training_seed"], row["actor_tensor_sha256"]) for row in physical}) != 70:
        raise ValueError("physical actors were not exactly deduplicated")

    payload = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "development",
        "selection_rule": SELECTION_RULE,
        "formal_held_out_outcomes_read": False,
        "development_evaluation_outcomes_read": False,
        "v8_protocol_sha256": v13["v8_protocol_sha256"],
        "v10_protocol_sha256": v13["v10_protocol_sha256"],
        "v11_protocol_sha256": v13["v11_protocol_sha256"],
        "v12_protocol_sha256": v13["v12_protocol_sha256"],
        "v13_protocol_sha256": v13["v13_protocol_sha256"],
        "v14_protocol_sha256": EXPECTED["protocol"],
        "logical_method_budget_alias_count": len(aliases),
        "physical_method_count": len(physical),
        "expected_remote_sumo_episodes": len(physical) * 9,
        "registered_maximum_remote_sumo_episodes": len(aliases) * 9,
        "predesignated_candidate": "h2oplus-v14-terminal-quarter",
        "control_algorithms": list(CONTROL_ALGORITHMS),
        "diagnostic_variants": [
            "h2oplus-v14-terminal-half",
            "h2oplus-v14-terminal-zero",
        ],
        "identity_evidence": identity_evidence,
        "bindings": {
            "v13_methods": descriptor(v13_path, output),
            "v14_protocol": descriptor(protocol_path, output),
            "v14_training_plan": descriptor(plan_path, output),
            "v14_branch_closeout": descriptor(closeout_path, output),
            "v14_branch_verification": descriptor(verification_path, output),
            "v14_branch_audit": descriptor(audit_path, output),
            "v14_branch_file_manifest": descriptor(manifest_path, output),
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
