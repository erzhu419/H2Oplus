#!/usr/bin/env python3
"""Run only the preregistered v16 headway-reserve 25k policies."""

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
from reproducibility import run_evaluation_matrix_v15_hold_calibration as v15


METHODS_SCHEMA = "transit-evaluation-methods-v16-headway-reserve-v1"
SELECTION_RULE = (
    "sealed v15 hold-eighth 25k parents and exact seed-matched actors fixed before "
    "v16; base scale 0.125, schedule-adjusted headway gain 1.0, candidate reserve "
    "cutoff 0.75 and diagnostic cutoffs 0.5/1.0 fixed before any v16 outcome; "
    "physical behavior identity is seed plus actor tensor SHA256 plus the complete "
    "headway-reserve transform"
)
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus-v16-headway-reserve-"
    r"(?P<variant>half|three-quarter|full))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
V16_PROTOCOL_SHA_ENV = "H2OPLUS_V16_HEADWAY_RESERVE_PROTOCOL_SHA256"
HORIZON_CUTOFFS = {"half": 0.5, "three-quarter": 0.75, "full": 1.0}
BASE_HOLD_SCALE = 0.125
HEADWAY_BALANCE_GAIN = 1.0


def bind_protocol_environment(payload: dict) -> None:
    v15.bind_protocol_environment(payload)
    value = payload["v16_protocol_sha256"]
    existing = os.environ.get(V16_PROTOCOL_SHA_ENV)
    if existing not in (None, value):
        raise ValueError(f"protocol environment conflicts for {V16_PROTOCOL_SHA_ENV}")
    os.environ[V16_PROTOCOL_SHA_ENV] = value


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
            "v16_protocol_sha256",
        )
    )
    if (
        schema != METHODS_SCHEMA
        or payload.get("schema_version") != METHODS_SCHEMA
        or payload.get("study_stage") != "development"
        or payload.get("selection_rule") != SELECTION_RULE
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("v16_development_evaluation_outcomes_read") is not False
        or payload.get("physical_method_count") != 15
        or payload.get("logical_method_count") != 15
        or payload.get("expected_new_remote_sumo_episodes") != 135
        or payload.get("predesignated_candidate")
        != "h2oplus-v16-headway-reserve-three-quarter"
        or any(
            not isinstance(value, str) or not SHA256.fullmatch(value)
            for value in protocol_hashes
        )
        or len(payload.get("aliases", [])) != 15
        or len(payload.get("methods", [])) != 15
    ):
        raise ValueError("v16 headway-reserve methods manifest scope drifted")
    bind_protocol_environment(payload)

    methods = []
    names: set[str] = set()
    behavior_keys: set[tuple[int, str, float, float, float]] = set()
    for raw in payload["methods"]:
        name = raw.get("name")
        match = METHOD_NAME.fullmatch(name or "")
        variant = match.group("variant") if match else ""
        cutoff = HORIZON_CUTOFFS.get(variant)
        hashes = (
            raw.get("checkpoint_sha256"),
            raw.get("training_run_signature"),
            raw.get("training_admission_sha256"),
            raw.get("training_checkpoint_sha256"),
            raw.get("actor_tensor_sha256"),
            raw.get("temperature_tensor_sha256"),
            raw.get("source_protocol_sha256"),
            raw.get("v16_protocol_sha256"),
        )
        key = (
            raw.get("training_seed"),
            raw.get("actor_tensor_sha256"),
            raw.get("base_hold_scale"),
            raw.get("headway_balance_gain"),
            cutoff,
        )
        seed = int(match.group("seed")) if match else -1
        parent_name = f"h2oplus-v15-hold-eighth-seed{seed}-events000025000"
        if (
            match is None
            or name in names
            or key in behavior_keys
            or any(
                not isinstance(value, str) or not SHA256.fullmatch(value)
                for value in hashes
            )
            or raw.get("type") != "checkpoint"
            or raw.get("artifact_lineage") != "v16-headway-reserve"
            or raw.get("parent_artifact_lineage") != "v15-hold-calibration"
            or raw.get("training_seed") != seed
            or raw.get("online_events") != 25_000
            or raw.get("algorithm") != match.group("algorithm")
            or not math.isclose(
                float(raw.get("base_hold_scale", -1.0)),
                BASE_HOLD_SCALE,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(raw.get("headway_balance_gain", -1.0)),
                HEADWAY_BALANCE_GAIN,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or raw.get("reserve_horizon_fraction") != cutoff
            or raw.get("parent_method") != parent_name
            or raw.get("predesignated_candidate")
            is not (variant == "three-quarter")
            or raw.get("source_protocol_sha256") != payload["v15_protocol_sha256"]
            or raw.get("v16_protocol_sha256") != payload["v16_protocol_sha256"]
        ):
            raise ValueError(f"malformed or duplicate v16 method: {raw!r}")
        checkpoint = _resolve(manifest, raw["checkpoint"])
        parent = dict(raw)
        parent.update(
            {
                "name": f"h2oplus-v14-terminal-quarter-seed{seed}-events000025000",
                "algorithm": "h2oplus-v14-terminal-quarter",
                "artifact_lineage": "v14-terminal-taper",
            }
        )
        v15.v14._validate_v14_artifact(
            parent, checkpoint, payload["v14_protocol_sha256"]
        )
        if sha256(checkpoint) != raw["checkpoint_sha256"]:
            raise ValueError(f"v16 parent checkpoint drifted: {name}")
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
                "artifact_lineage": "v16-headway-reserve",
                "parent_artifact_lineage": "v15-hold-calibration",
                "parent_method": parent_name,
                "base_hold_scale": BASE_HOLD_SCALE,
                "headway_balance_gain": HEADWAY_BALANCE_GAIN,
                "reserve_horizon_fraction": cutoff,
            }
        )
    if len(methods) != 15 or len(behavior_keys) != 15:
        raise ValueError("v16 matrix requires exactly 15 reserve physical behaviors")
    return sorted(methods, key=lambda item: item["name"])


def evaluation_sources(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v15.evaluation_sources(sumo_dir, edge_xml))
    paths.update(
        {
            ROOT / "SimpleSAC/evaluate_transit_v16_headway_reserve.py",
            Path(__file__).resolve(),
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def job_command(job: dict) -> list[str]:
    command = v15.v14.job_command(job)
    method = job["method"]
    command.extend(
        (
            "--expected-temperature-tensor-sha256",
            method["temperature_tensor_sha256"],
            "--expected-base-hold-scale",
            str(method["base_hold_scale"]),
            "--expected-headway-balance-gain",
            str(method["headway_balance_gain"]),
            "--expected-reserve-horizon-fraction",
            str(method["reserve_horizon_fraction"]),
            "--expected-parent-method-name",
            method["parent_method"],
        )
    )
    return command


v15.v14.v13.v12.v11.base.EVALUATORS["v7"] = (
    ROOT / "SimpleSAC/evaluate_transit_v16_headway_reserve.py"
)
v15.v14.v13.v12.v11.base.METHODS_SCHEMAS["v7"] = METHODS_SCHEMA
v15.v14.v13.v12.v11.base.load_methods = load_development_methods
v15.v14.v13.v12.v11.base.evaluation_source_paths_v7 = evaluation_sources
v15.v14.v13.v12.v11.base._job_command = job_command
v15.v14.v13.v12.v11.base.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    v15.v14.v13.v12.v11.base.main()
