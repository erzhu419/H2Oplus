#!/usr/bin/env python3
"""Run the v7 matrix engine with frozen v8/v10/v11/v12/v13 provenance."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility import run_evaluation_matrix_v12_staged_release as v12


METHODS_SCHEMA = "transit-evaluation-methods-v13-proximal-release-v1"
SELECTION_RULE = (
    "all preregistered common-budget aliases fixed before evaluation; physical "
    "evaluation deduplicated only by exact actor tensor SHA256 within a seed"
)
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus|sac-online|h2oplus-v10-shared-prefix|"
    r"h2oplus-v10-delayed-actor|"
    r"h2oplus-v11-actor-hold|h2oplus-v12-actor-ramp|"
    r"h2oplus-v13-proximal-(?:weak|medium|strong))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
V13_PROTOCOL_SHA_ENV = "H2OPLUS_V13_PROXIMAL_RELEASE_PROTOCOL_SHA256"
PROXIMAL_RATES = {
    "proximal-weak": 1e-5,
    "proximal-medium": 2.5e-5,
    "proximal-strong": 5e-5,
}


def bind_protocol_environment(payload: dict) -> None:
    v12.bind_protocol_environment(payload)
    value = payload["v13_protocol_sha256"]
    existing = os.environ.get(V13_PROTOCOL_SHA_ENV)
    if existing not in (None, value):
        raise ValueError(f"protocol environment conflicts for {V13_PROTOCOL_SHA_ENV}")
    os.environ[V13_PROTOCOL_SHA_ENV] = value


def _resolve(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def _validate_v13_artifact(raw: dict, checkpoint: Path, protocol_hash: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    extra = payload.get("extra", {})
    variant = raw["algorithm"][len("h2oplus-v13-") :]
    rate = PROXIMAL_RATES[variant]
    common = {
        "method": "h2oplus",
        "variant": variant,
        "stage": "branch",
        "phase": "complete",
        "training_seed": raw["training_seed"],
        "run_signature": raw["training_run_signature"],
        "training_checkpoint_sha256": raw["training_checkpoint_sha256"],
        "v13_online_events_nominal": raw["online_events"],
        "v13_online_events_actual": raw["actual_online_events"],
        "v13_learner_update_steps": raw["learner_update_steps"],
        "v13_ratio_optimizer_steps": raw["ratio_optimizer_steps"],
        "v13_actor_tensor_sha256": raw["actor_tensor_sha256"],
        "v13_temperature_tensor_sha256": raw["temperature_tensor_sha256"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {
        key: (extra.get(key), value)
        for key, value in common.items()
        if extra.get(key) != value
    }
    admission = extra.get("training_admission", {})
    handoff = admission.get("selected_arguments", {}).get("source_handoff", {})
    intervention = extra.get("v13_intervention", {})
    expected_handoff_events = 9_999 if int(raw["training_seed"]) == 6205 else 10_000
    if (
        mismatches
        or raw["online_events"] != 25_000
        or payload.get("artifact_schema") != "transit-policy-artifact-v7-v1"
        or v12.actor_digest(payload.get("actor", {})) != raw["actor_tensor_sha256"]
        or admission.get("schema_version")
        != "transit-training-admission-v13-proximal-release-development-v2"
        or admission.get("sha256") != raw["training_admission_sha256"]
        or extra.get("v13_protocol", {}).get("sha256") != protocol_hash
        or handoff.get("manifest_sha256") != raw["source_handoff_manifest_sha256"]
        or handoff.get("checkpoint_state_sha256")
        != raw["source_handoff_checkpoint_state_sha256"]
        or handoff.get("online_events_actual") != expected_handoff_events
        or intervention.get("schema_version")
        != "transit-h2oplus-v13-proximal-release-variant-v1"
        or intervention.get("variant") != variant
        or intervention.get("proximal_rate_per_actor_update") != rate
        or intervention.get("registered_post_10k_actor_updates") != 15_000
        or not math.isclose(
            float(intervention.get("no_gradient_anchor_retention_at_25k", math.nan)),
            (1.0 - rate) ** 15_000,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or intervention.get("predesignated_candidate")
        is not (variant == "proximal-medium")
        or intervention.get("unit_ratio_weights") is not True
        or intervention.get("checkpoint_selection_by_behavior") is not False
        or sha256(checkpoint) != raw["checkpoint_sha256"]
    ):
        raise ValueError(f"v13 policy artifact is misbound: {raw['name']}: {mismatches}")


def load_development_methods(path: Path, schema: str) -> list[dict]:
    manifest = path.expanduser().resolve(strict=True)
    payload = json.loads(manifest.read_text())
    protocol_hashes = tuple(
        payload.get(key)
        for key in (
            "v8_protocol_sha256",
            "v10_protocol_sha256",
            "v11_protocol_sha256",
            "v12_protocol_sha256",
            "v13_protocol_sha256",
        )
    )
    if (
        schema != METHODS_SCHEMA
        or payload.get("schema_version") != METHODS_SCHEMA
        or payload.get("study_stage") != "development"
        or payload.get("selection_rule") != SELECTION_RULE
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("development_evaluation_outcomes_read") is not False
        or payload.get("physical_method_count") != 65
        or payload.get("logical_method_budget_alias_count") != 160
        or payload.get("expected_remote_sumo_episodes") != 585
        or payload.get("registered_maximum_remote_sumo_episodes") != 1440
        or any(
            not isinstance(value, str) or not SHA256.fullmatch(value)
            for value in protocol_hashes
        )
        or len(payload.get("aliases", [])) != 160
        or len(payload.get("methods", [])) != 65
    ):
        raise ValueError("v13 proximal-release methods manifest scope drifted")
    bind_protocol_environment(payload)
    methods = []
    names, actor_keys = set(), set()
    for raw in payload["methods"]:
        name = raw.get("name")
        match = METHOD_NAME.fullmatch(name or "")
        hashes = (
            raw.get("checkpoint_sha256"),
            raw.get("training_run_signature"),
            raw.get("training_admission_sha256"),
            raw.get("training_checkpoint_sha256"),
            raw.get("actor_tensor_sha256"),
            raw.get("source_protocol_sha256"),
        )
        key = (raw.get("training_seed"), raw.get("actor_tensor_sha256"))
        if (
            match is None
            or name in names
            or key in actor_keys
            or any(
                not isinstance(value, str) or not SHA256.fullmatch(value)
                for value in hashes
            )
            or raw.get("type") != "checkpoint"
            or raw.get("training_seed") != int(match.group("seed"))
            or raw.get("online_events") != int(match.group("events"))
            or raw.get("algorithm") != match.group("algorithm")
        ):
            raise ValueError(f"malformed or duplicate v13 method: {raw!r}")
        checkpoint = _resolve(manifest, raw["checkpoint"])
        lineage = raw.get("artifact_lineage")
        if lineage == "v13-proximal-release":
            if raw.get("source_protocol_sha256") != payload["v13_protocol_sha256"]:
                raise ValueError(f"v13 protocol binding drifted: {name}")
            _validate_v13_artifact(raw, checkpoint, payload["v13_protocol_sha256"])
        elif lineage == "v12-staged-release":
            if raw.get("source_protocol_sha256") != payload["v12_protocol_sha256"]:
                raise ValueError(f"v12 protocol binding drifted: {name}")
            v12._validate_v12_artifact(raw, checkpoint, payload["v12_protocol_sha256"])
        else:
            v12.v11._validate_artifact(
                raw,
                checkpoint,
                payload["v10_protocol_sha256"],
                payload["v11_protocol_sha256"],
            )
        names.add(name)
        actor_keys.add(key)
        row = {
            "name": name,
            "algorithm": raw["algorithm"],
            "training_seed": raw["training_seed"],
            "type": "checkpoint",
            "checkpoint": checkpoint,
            "checkpoint_sha256": sha256(checkpoint),
            "training_run_signature": raw["training_run_signature"],
            "training_admission_sha256": raw["training_admission_sha256"],
            "training_checkpoint_sha256": raw["training_checkpoint_sha256"],
            "training_plan_sha256": raw["source_protocol_sha256"],
            "online_events": raw["online_events"],
            "actual_online_events": raw["actual_online_events"],
            "learner_update_steps": raw["learner_update_steps"],
            "ratio_optimizer_steps": raw["ratio_optimizer_steps"],
            "actor_tensor_sha256": raw["actor_tensor_sha256"],
            "artifact_lineage": lineage,
        }
        if "temperature_tensor_sha256" in raw:
            row["temperature_tensor_sha256"] = raw["temperature_tensor_sha256"]
        methods.append(row)
    if len(methods) != 65 or len(actor_keys) != 65:
        raise ValueError(f"v13 matrix requires 65 unique actors, got {len(methods)}")
    return sorted(methods, key=lambda item: item["name"])


def evaluation_sources(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v12.evaluation_sources(sumo_dir, edge_xml))
    paths.update(
        {
            ROOT / "SimpleSAC/evaluate_transit_v13_proximal_release.py",
            Path(__file__).resolve(),
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def job_command(job: dict) -> list[str]:
    command = v12.job_command(job)
    method = job["method"]
    if method.get("artifact_lineage") == "v13-proximal-release":
        command.extend(
            (
                "--expected-temperature-tensor-sha256",
                method["temperature_tensor_sha256"],
            )
        )
    return command


v12.v11.base.EVALUATORS["v7"] = ROOT / "SimpleSAC/evaluate_transit_v13_proximal_release.py"
v12.v11.base.METHODS_SCHEMAS["v7"] = METHODS_SCHEMA
v12.v11.base.load_methods = load_development_methods
v12.v11.base.evaluation_source_paths_v7 = evaluation_sources
v12.v11.base._job_command = job_command
v12.v11.base.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    v12.v11.base.main()
