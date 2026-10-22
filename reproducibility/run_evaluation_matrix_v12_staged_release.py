#!/usr/bin/env python3
"""Run the v7 matrix engine with frozen v8/v10/v11/v12 provenance."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from SimpleSAC.evaluate_transit_v11_actor_hold import actor_digest
from reproducibility import run_evaluation_matrix_v11_actor_hold as v11


METHODS_SCHEMA = "transit-evaluation-methods-v12-staged-release-v1"
SELECTION_RULE = (
    "all preregistered common-budget aliases fixed before evaluation; physical "
    "evaluation deduplicated only by exact actor tensor SHA256 within a seed"
)
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus|sac-online|h2oplus-v10-(?:shared-prefix|delayed-actor)|"
    r"h2oplus-v11-actor-hold|h2oplus-v12-(?:abrupt-control|actor-ramp|"
    r"temperature-ramp|actor-anchor|staged-combined))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
V12_PROTOCOL_SHA_ENV = "H2OPLUS_V12_STAGED_RELEASE_PROTOCOL_SHA256"


def bind_protocol_environment(payload: dict) -> None:
    v11.bind_protocol_environment(
        payload["v10_protocol_sha256"], payload["v11_protocol_sha256"]
    )
    value = payload["v12_protocol_sha256"]
    existing = os.environ.get(V12_PROTOCOL_SHA_ENV)
    if existing not in (None, value):
        raise ValueError(f"protocol environment conflicts for {V12_PROTOCOL_SHA_ENV}")
    os.environ[V12_PROTOCOL_SHA_ENV] = value


def _resolve(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def _validate_v12_artifact(raw: dict, checkpoint: Path, protocol_hash: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    extra = payload.get("extra", {})
    variant = raw["algorithm"][len("h2oplus-v12-") :]
    common = {
        "method": "h2oplus",
        "variant": variant,
        "stage": "branch",
        "phase": "complete",
        "training_seed": raw["training_seed"],
        "run_signature": raw["training_run_signature"],
        "training_checkpoint_sha256": raw["training_checkpoint_sha256"],
        "v12_online_events_nominal": raw["online_events"],
        "v12_online_events_actual": raw["actual_online_events"],
        "v12_learner_update_steps": raw["learner_update_steps"],
        "v12_ratio_optimizer_steps": raw["ratio_optimizer_steps"],
        "v12_actor_tensor_sha256": raw["actor_tensor_sha256"],
        "v12_temperature_tensor_sha256": raw["temperature_tensor_sha256"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {
        key: (extra.get(key), value)
        for key, value in common.items()
        if extra.get(key) != value
    }
    flags = {
        "abrupt-control": (False, False, False),
        "actor-ramp": (True, False, False),
        "temperature-ramp": (False, True, False),
        "actor-anchor": (False, False, True),
        "staged-combined": (True, True, True),
    }[variant]
    intervention = extra.get("v12_intervention", {})
    admission = extra.get("training_admission", {})
    if (
        mismatches
        or raw["online_events"] != 25_000
        or payload.get("artifact_schema") != "transit-policy-artifact-v7-v1"
        or actor_digest(payload.get("actor", {})) != raw["actor_tensor_sha256"]
        or admission.get("schema_version")
        != "transit-training-admission-v12-staged-release-development-v2"
        or admission.get("sha256") != raw["training_admission_sha256"]
        or extra.get("v12_protocol", {}).get("sha256") != protocol_hash
        or intervention.get("schema_version")
        != "transit-h2oplus-v12-staged-release-variant-v1"
        or intervention.get("variant") != variant
        or intervention.get("actor_learning_rate_ramp") is not flags[0]
        or intervention.get("temperature_learning_rate_ramp") is not flags[1]
        or intervention.get("actor_state_anchor") is not flags[2]
        or intervention.get("unit_ratio_weights") is not True
        or intervention.get("checkpoint_selection_by_behavior") is not False
        or sha256(checkpoint) != raw["checkpoint_sha256"]
    ):
        raise ValueError(f"v12 policy artifact is misbound: {raw['name']}: {mismatches}")


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
        )
    )
    if (
        schema != METHODS_SCHEMA
        or payload.get("schema_version") != METHODS_SCHEMA
        or payload.get("study_stage") != "development"
        or payload.get("selection_rule") != SELECTION_RULE
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("development_evaluation_outcomes_read") is not False
        or payload.get("physical_method_count") != 70
        or payload.get("logical_method_budget_alias_count") != 180
        or payload.get("expected_remote_sumo_episodes") != 630
        or payload.get("registered_maximum_remote_sumo_episodes") != 1620
        or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in protocol_hashes)
        or len(payload.get("aliases", [])) != 180
        or len(payload.get("methods", [])) != 70
    ):
        raise ValueError("v12 staged-release methods manifest scope drifted")
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
            or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in hashes)
            or raw.get("type") != "checkpoint"
            or raw.get("training_seed") != int(match.group("seed"))
            or raw.get("online_events") != int(match.group("events"))
            or raw.get("algorithm") != match.group("algorithm")
        ):
            raise ValueError(f"malformed or duplicate v12 method: {raw!r}")
        checkpoint = _resolve(manifest, raw["checkpoint"])
        if raw.get("artifact_lineage") == "v12-staged-release":
            if raw.get("source_protocol_sha256") != payload["v12_protocol_sha256"]:
                raise ValueError(f"v12 protocol binding drifted: {name}")
            _validate_v12_artifact(raw, checkpoint, payload["v12_protocol_sha256"])
        else:
            v11._validate_artifact(
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
            "artifact_lineage": raw["artifact_lineage"],
        }
        if "temperature_tensor_sha256" in raw:
            row["temperature_tensor_sha256"] = raw["temperature_tensor_sha256"]
        methods.append(row)
    if len(methods) != 70 or len(actor_keys) != 70:
        raise ValueError(f"v12 matrix requires 70 unique actors, got {len(methods)}")
    return sorted(methods, key=lambda item: item["name"])


def evaluation_sources(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v11.original_sources(sumo_dir, edge_xml))
    paths.update(
        {
            ROOT / "SimpleSAC/evaluate_transit_v11_actor_hold.py",
            ROOT / "SimpleSAC/evaluate_transit_v12_staged_release.py",
            ROOT / "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
            Path(__file__).resolve(),
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def job_command(job: dict) -> list[str]:
    command = v11.job_command(job)
    method = job["method"]
    if method.get("artifact_lineage") == "v12-staged-release":
        command.extend(
            (
                "--expected-temperature-tensor-sha256",
                method["temperature_tensor_sha256"],
            )
        )
    return command


v11.base.EVALUATORS["v7"] = ROOT / "SimpleSAC/evaluate_transit_v12_staged_release.py"
v11.base.METHODS_SCHEMAS["v7"] = METHODS_SCHEMA
v11.base.load_methods = load_development_methods
v11.base.evaluation_source_paths_v7 = evaluation_sources
v11.base._job_command = job_command
v11.base.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    v11.base.main()
