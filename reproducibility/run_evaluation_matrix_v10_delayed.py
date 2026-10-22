#!/usr/bin/env python3
"""Run the v7 matrix engine with frozen v8/v10 provenance rules."""

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
from SimpleSAC.evaluate_transit_v10_delayed import actor_digest
from reproducibility import run_evaluation_matrix_v4 as base


METHODS_SCHEMA = "transit-evaluation-methods-v10-delayed-v1"
SELECTION_RULE = (
    "all preregistered common-budget aliases fixed before evaluation; physical "
    "evaluation deduplicated only by exact actor tensor SHA256 within a seed"
)
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus|sac-online|h2oplus-v10-(?:shared-prefix|unit-control|"
    r"delayed-replay|delayed-actor|delayed-order|delayed-combined))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
V8_PROTOCOL_SHA256 = "26d4f5c4ca9f337c11886855658127ef93a35954ca58eb43710aa70350610294"
V8_PROTOCOL_SHA_ENV = "H2OPLUS_V8_FEWSHOT_PROTOCOL_SHA256"
V10_PROTOCOL_SHA_ENV = "H2OPLUS_V10_DELAYED_PROTOCOL_SHA256"


def bind_protocol_environment(v10_protocol_hash: str) -> None:
    for name, value in (
        (V8_PROTOCOL_SHA_ENV, V8_PROTOCOL_SHA256),
        (V10_PROTOCOL_SHA_ENV, v10_protocol_hash),
    ):
        existing = os.environ.get(name)
        if existing not in (None, value):
            raise ValueError(f"protocol environment conflicts for {name}: {existing}")
        os.environ[name] = value


def _resolve(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def _validate_artifact(raw: dict, checkpoint: Path, protocol_hash: str) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    extra = payload.get("extra", {})
    lineage = raw["artifact_lineage"]
    common = {
        "training_seed": raw["training_seed"],
        "run_signature": raw["training_run_signature"],
        "training_checkpoint_sha256": raw["training_checkpoint_sha256"],
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    mismatches = {
        key: (extra.get(key), value)
        for key, value in common.items()
        if extra.get(key) != value
    }
    if lineage == "v8-fewshot":
        specific = {
            "method": raw["algorithm"],
            "fewshot_online_events_nominal": raw["online_events"],
            "fewshot_online_events_actual": raw["actual_online_events"],
            "fewshot_learner_update_steps": raw["learner_update_steps"],
            "fewshot_ratio_optimizer_steps": raw["ratio_optimizer_steps"],
            "fewshot_actor_tensor_sha256": raw["actor_tensor_sha256"],
        }
        artifact_protocol = extra.get("fewshot_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v8-fewshot-development-v1"
    elif lineage == "v10-delayed":
        variant = raw["algorithm"][len("h2oplus-v10-") :]
        specific = {
            "method": "h2oplus",
            "variant": variant,
            "v10_online_events_nominal": raw["online_events"],
            "v10_online_events_actual": raw["actual_online_events"],
            "v10_learner_update_steps": raw["learner_update_steps"],
            "v10_ratio_optimizer_steps": raw["ratio_optimizer_steps"],
            "v10_actor_tensor_sha256": raw["actor_tensor_sha256"],
        }
        artifact_protocol = extra.get("v10_protocol", {}).get("sha256")
        admission_schema = "transit-training-admission-v10-delayed-development-v4"
        if (
            extra.get("v10_intervention", {}).get("variant") != variant
            or extra.get("v10_intervention", {}).get("unit_ratio_weights") is not True
        ):
            raise ValueError(f"v10 intervention drifted: {raw['name']}")
    else:
        raise ValueError(f"unsupported actor lineage: {lineage!r}")
    mismatches.update(
        {
            key: (extra.get(key), value)
            for key, value in specific.items()
            if extra.get(key) != value
        }
    )
    admission = extra.get("training_admission", {})
    if (
        mismatches
        or payload.get("artifact_schema") != "transit-policy-artifact-v7-v1"
        or actor_digest(payload.get("actor", {})) != raw["actor_tensor_sha256"]
        or artifact_protocol != raw["source_protocol_sha256"]
        or artifact_protocol not in {V8_PROTOCOL_SHA256, protocol_hash}
        or admission.get("schema_version") != admission_schema
        or admission.get("sha256") != raw["training_admission_sha256"]
        or sha256(checkpoint) != raw["checkpoint_sha256"]
    ):
        raise ValueError(f"policy artifact is misbound: {raw['name']}: {mismatches}")


def load_development_methods(path: Path, schema: str) -> list[dict]:
    manifest = path.expanduser().resolve(strict=True)
    payload = json.loads(manifest.read_text())
    protocol_hash = payload.get("v10_protocol_sha256")
    if (
        schema != METHODS_SCHEMA
        or payload.get("schema_version") != METHODS_SCHEMA
        or payload.get("study_stage") != "development"
        or payload.get("selection_rule") != SELECTION_RULE
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("development_evaluation_outcomes_read") is not False
        or payload.get("physical_method_count") != 60
        or payload.get("logical_method_budget_alias_count") != 140
        or payload.get("expected_remote_sumo_episodes") != 540
        or payload.get("v8_protocol_sha256") != V8_PROTOCOL_SHA256
        or not isinstance(protocol_hash, str)
        or not SHA256.fullmatch(protocol_hash)
        or len(payload.get("aliases", [])) != 140
    ):
        raise ValueError("v10 delayed methods manifest scope drifted")
    bind_protocol_environment(protocol_hash)
    methods = []
    names, actor_hashes = set(), set()
    for raw in payload.get("methods", []):
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
        if (
            match is None
            or name in names
            or any(not isinstance(value, str) or not SHA256.fullmatch(value) for value in hashes)
            or raw.get("type") != "checkpoint"
            or raw.get("training_seed") != int(match.group("seed"))
            or raw.get("online_events") != int(match.group("events"))
            or raw.get("algorithm") != match.group("algorithm")
            or raw.get("actor_tensor_sha256") in actor_hashes
        ):
            raise ValueError(f"malformed or duplicate v10 method: {raw!r}")
        checkpoint = _resolve(manifest, raw["checkpoint"])
        _validate_artifact(raw, checkpoint, protocol_hash)
        names.add(name)
        actor_hashes.add(raw["actor_tensor_sha256"])
        methods.append(
            {
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
        )
    if len(methods) != 60 or len(actor_hashes) != 60:
        raise ValueError(f"v10 matrix requires 60 unique actors, got {len(methods)}")
    return sorted(methods, key=lambda item: item["name"])


original_sources = base.evaluation_source_paths_v7


def evaluation_sources(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(original_sources(sumo_dir, edge_xml))
    paths.update(
        {
            Path(__file__).resolve(),
            ROOT / "SimpleSAC/evaluate_transit_v10_delayed.py",
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


original_job_command = base._job_command


def job_command(job: dict) -> list[str]:
    command = original_job_command(job)
    method = job["method"]
    if method["type"] == "checkpoint":
        command.extend(
            (
                "--expected-actor-tensor-sha256",
                method["actor_tensor_sha256"],
                "--expected-learner-update-steps",
                str(method["learner_update_steps"]),
                "--expected-ratio-optimizer-steps",
                str(method["ratio_optimizer_steps"]),
            )
        )
    return command


base.EVALUATORS["v7"] = ROOT / "SimpleSAC/evaluate_transit_v10_delayed.py"
base.METHODS_SCHEMAS["v7"] = METHODS_SCHEMA
base.load_methods = load_development_methods
base.evaluation_source_paths_v7 = evaluation_sources
base._job_command = job_command
base.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    base.main()
