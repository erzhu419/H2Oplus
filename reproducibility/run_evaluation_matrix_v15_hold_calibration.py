#!/usr/bin/env python3
"""Run only the preregistered v15 calibrated 25k development policies."""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility import run_evaluation_matrix_v14_terminal_taper as v14


METHODS_SCHEMA = "transit-evaluation-methods-v15-hold-calibration-v1"
SELECTION_RULE = (
    "sealed v14 terminal-quarter 25k parents fixed before v15; candidate scale "
    "0.25 and diagnostic scales 0.125/0.375 fixed before any v15 outcome; "
    "physical behavior identity is seed plus actor tensor SHA256 plus hold scale"
)
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus-v15-hold-(?P<variant>eighth|quarter|three-eighths))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
V15_PROTOCOL_SHA_ENV = "H2OPLUS_V15_HOLD_CALIBRATION_PROTOCOL_SHA256"
HOLD_SCALES = {
    "eighth": 0.125,
    "quarter": 0.25,
    "three-eighths": 0.375,
}


def bind_protocol_environment(payload: dict) -> None:
    v14.bind_protocol_environment(payload)
    value = payload["v15_protocol_sha256"]
    existing = os.environ.get(V15_PROTOCOL_SHA_ENV)
    if existing not in (None, value):
        raise ValueError(f"protocol environment conflicts for {V15_PROTOCOL_SHA_ENV}")
    os.environ[V15_PROTOCOL_SHA_ENV] = value


def _resolve(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


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
            "v14_protocol_sha256",
            "v15_protocol_sha256",
        )
    )
    if (
        schema != METHODS_SCHEMA
        or payload.get("schema_version") != METHODS_SCHEMA
        or payload.get("study_stage") != "development"
        or payload.get("selection_rule") != SELECTION_RULE
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("v15_development_evaluation_outcomes_read") is not False
        or payload.get("physical_method_count") != 15
        or payload.get("logical_method_count") != 15
        or payload.get("expected_new_remote_sumo_episodes") != 135
        or payload.get("predesignated_candidate") != "h2oplus-v15-hold-quarter"
        or any(
            not isinstance(value, str) or not SHA256.fullmatch(value)
            for value in protocol_hashes
        )
        or len(payload.get("aliases", [])) != 15
        or len(payload.get("methods", [])) != 15
    ):
        raise ValueError("v15 hold-calibration methods manifest scope drifted")
    bind_protocol_environment(payload)

    methods = []
    names: set[str] = set()
    behavior_keys: set[tuple[int, str, float]] = set()
    for raw in payload["methods"]:
        name = raw.get("name")
        match = METHOD_NAME.fullmatch(name or "")
        variant = match.group("variant") if match else ""
        hold_scale = HOLD_SCALES.get(variant)
        hashes = (
            raw.get("checkpoint_sha256"),
            raw.get("training_run_signature"),
            raw.get("training_admission_sha256"),
            raw.get("training_checkpoint_sha256"),
            raw.get("actor_tensor_sha256"),
            raw.get("temperature_tensor_sha256"),
            raw.get("source_protocol_sha256"),
            raw.get("v15_protocol_sha256"),
        )
        key = (raw.get("training_seed"), raw.get("actor_tensor_sha256"), hold_scale)
        seed = int(match.group("seed")) if match else -1
        parent_name = f"h2oplus-v14-terminal-quarter-seed{seed}-events000025000"
        if (
            match is None
            or name in names
            or key in behavior_keys
            or any(
                not isinstance(value, str) or not SHA256.fullmatch(value)
                for value in hashes
            )
            or raw.get("type") != "checkpoint"
            or raw.get("artifact_lineage") != "v15-hold-calibration"
            or raw.get("parent_artifact_lineage") != "v14-terminal-taper"
            or raw.get("training_seed") != seed
            or raw.get("online_events") != 25_000
            or raw.get("algorithm") != match.group("algorithm")
            or raw.get("hold_scale") != hold_scale
            or raw.get("parent_method") != parent_name
            or raw.get("predesignated_candidate") is not (variant == "quarter")
            or raw.get("source_protocol_sha256") != payload["v14_protocol_sha256"]
            or raw.get("v15_protocol_sha256") != payload["v15_protocol_sha256"]
        ):
            raise ValueError(f"malformed or duplicate v15 method: {raw!r}")
        checkpoint = _resolve(manifest, raw["checkpoint"])
        parent = dict(raw)
        parent.update(
            {
                "name": parent_name,
                "algorithm": "h2oplus-v14-terminal-quarter",
                "artifact_lineage": "v14-terminal-taper",
            }
        )
        v14._validate_v14_artifact(parent, checkpoint, payload["v14_protocol_sha256"])
        if sha256(checkpoint) != raw["checkpoint_sha256"]:
            raise ValueError(f"v15 parent checkpoint drifted: {name}")
        names.add(name)
        behavior_keys.add(key)
        methods.append(
            {
                "name": name,
                "algorithm": raw["algorithm"],
                "training_seed": seed,
                "type": "checkpoint",
                "checkpoint": checkpoint,
                "checkpoint_sha256": raw["checkpoint_sha256"],
                "training_run_signature": raw["training_run_signature"],
                "training_admission_sha256": raw["training_admission_sha256"],
                "training_checkpoint_sha256": raw["training_checkpoint_sha256"],
                "training_plan_sha256": raw["source_protocol_sha256"],
                "online_events": 25_000,
                "actual_online_events": raw["actual_online_events"],
                "learner_update_steps": raw["learner_update_steps"],
                "ratio_optimizer_steps": raw["ratio_optimizer_steps"],
                "actor_tensor_sha256": raw["actor_tensor_sha256"],
                "temperature_tensor_sha256": raw["temperature_tensor_sha256"],
                "artifact_lineage": "v15-hold-calibration",
                "parent_artifact_lineage": "v14-terminal-taper",
                "parent_method": parent_name,
                "hold_scale": hold_scale,
            }
        )
    if len(methods) != 15 or len(behavior_keys) != 15:
        raise ValueError("v15 matrix requires exactly 15 calibrated physical behaviors")
    return sorted(methods, key=lambda item: item["name"])


def evaluation_sources(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v14.evaluation_sources(sumo_dir, edge_xml))
    paths.update(
        {
            ROOT / "SimpleSAC/evaluate_transit_v15_hold_calibration.py",
            Path(__file__).resolve(),
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def job_command(job: dict) -> list[str]:
    command = v14.job_command(job)
    method = job["method"]
    command.extend(
        (
            "--expected-temperature-tensor-sha256",
            method["temperature_tensor_sha256"],
            "--expected-hold-scale",
            str(method["hold_scale"]),
            "--expected-parent-method-name",
            method["parent_method"],
        )
    )
    return command


v14.v13.v12.v11.base.EVALUATORS["v7"] = (
    ROOT / "SimpleSAC/evaluate_transit_v15_hold_calibration.py"
)
v14.v13.v12.v11.base.METHODS_SCHEMAS["v7"] = METHODS_SCHEMA
v14.v13.v12.v11.base.load_methods = load_development_methods
v14.v13.v12.v11.base.evaluation_source_paths_v7 = evaluation_sources
v14.v13.v12.v11.base._job_command = job_command
v14.v13.v12.v11.base.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    v14.v13.v12.v11.base.main()
