#!/usr/bin/env python3
"""Run only the preregistered v17 guarded tail-service 25k policies."""

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
from reproducibility import run_evaluation_matrix_v16_headway_reserve as v16


METHODS_SCHEMA = "transit-evaluation-methods-v17-tail-service-reserve-v1"
SELECTION_RULE = (
    "sealed v16 headway-reserve-three-quarter 25k parents and exact seed-matched "
    "actors fixed before v17; the tail-only low-load positive-arrival guard, "
    "candidate 30-second floor and diagnostic 20/40-second floors fixed before any "
    "v17 outcome; physical behavior identity is seed plus actor tensor SHA256 plus "
    "the complete inherited v16 transform and guarded v17 floor"
)
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus-v17-tail-service-reserve-"
    r"(?P<variant>twenty|thirty|forty))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
V17_PROTOCOL_SHA_ENV = "H2OPLUS_V17_TAIL_SERVICE_RESERVE_PROTOCOL_SHA256"
TAIL_SERVICE_FLOORS = {"twenty": 20.0, "thirty": 30.0, "forty": 40.0}
PARENT_BASE_HOLD_SCALE = 0.125
PARENT_HEADWAY_BALANCE_GAIN = 1.0
PARENT_RESERVE_HORIZON_FRACTION = 0.75
ACTIVATION_START_FRACTION = 0.5
ACTIVATION_END_FRACTION = 0.75
MAX_ONBOARD_CAPACITY_FRACTION = 0.25


def bind_protocol_environment(payload: dict) -> None:
    v16.bind_protocol_environment(payload)
    value = payload["v17_protocol_sha256"]
    existing = os.environ.get(V17_PROTOCOL_SHA_ENV)
    if existing not in (None, value):
        raise ValueError(f"protocol environment conflicts for {V17_PROTOCOL_SHA_ENV}")
    os.environ[V17_PROTOCOL_SHA_ENV] = value


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
            "v17_protocol_sha256",
        )
    )
    if (
        schema != METHODS_SCHEMA
        or payload.get("schema_version") != METHODS_SCHEMA
        or payload.get("study_stage") != "development"
        or payload.get("selection_rule") != SELECTION_RULE
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("v17_development_evaluation_outcomes_read") is not False
        or payload.get("physical_method_count") != 15
        or payload.get("logical_method_count") != 15
        or payload.get("expected_new_remote_sumo_episodes") != 135
        or payload.get("predesignated_candidate")
        != "h2oplus-v17-tail-service-reserve-thirty"
        or any(
            not isinstance(value, str) or not SHA256.fullmatch(value)
            for value in protocol_hashes
        )
        or len(payload.get("aliases", [])) != 15
        or len(payload.get("methods", [])) != 15
    ):
        raise ValueError("v17 tail-service methods manifest scope drifted")
    bind_protocol_environment(payload)

    methods = []
    names: set[str] = set()
    behavior_keys: set[
        tuple[int, str, float, float, float, float, float, float, float]
    ] = set()
    for raw in payload["methods"]:
        name = raw.get("name")
        match = METHOD_NAME.fullmatch(name or "")
        variant = match.group("variant") if match else ""
        floor_seconds = TAIL_SERVICE_FLOORS.get(variant)
        hashes = (
            raw.get("checkpoint_sha256"),
            raw.get("training_run_signature"),
            raw.get("training_admission_sha256"),
            raw.get("training_checkpoint_sha256"),
            raw.get("actor_tensor_sha256"),
            raw.get("temperature_tensor_sha256"),
            raw.get("source_protocol_sha256"),
            raw.get("v17_protocol_sha256"),
        )
        key = (
            raw.get("training_seed"),
            raw.get("actor_tensor_sha256"),
            raw.get("parent_base_hold_scale"),
            raw.get("parent_headway_balance_gain"),
            raw.get("parent_reserve_horizon_fraction"),
            floor_seconds,
            raw.get("activation_start_fraction"),
            raw.get("activation_end_fraction"),
            raw.get("max_onboard_capacity_fraction"),
        )
        seed = int(match.group("seed")) if match else -1
        parent_name = (
            f"h2oplus-v16-headway-reserve-three-quarter-seed{seed}-events000025000"
        )
        if (
            match is None
            or name in names
            or key in behavior_keys
            or any(
                not isinstance(value, str) or not SHA256.fullmatch(value)
                for value in hashes
            )
            or raw.get("type") != "checkpoint"
            or raw.get("artifact_lineage") != "v17-tail-service-reserve"
            or raw.get("parent_artifact_lineage") != "v16-headway-reserve"
            or raw.get("training_seed") != seed
            or raw.get("online_events") != 25_000
            or raw.get("algorithm") != match.group("algorithm")
            or not math.isclose(
                float(raw.get("parent_base_hold_scale", -1.0)),
                PARENT_BASE_HOLD_SCALE,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or not math.isclose(
                float(raw.get("parent_headway_balance_gain", -1.0)),
                PARENT_HEADWAY_BALANCE_GAIN,
                rel_tol=0.0,
                abs_tol=0.0,
            )
            or raw.get("parent_reserve_horizon_fraction")
            != PARENT_RESERVE_HORIZON_FRACTION
            or raw.get("tail_service_floor_seconds") != floor_seconds
            or raw.get("activation_start_fraction") != ACTIVATION_START_FRACTION
            or raw.get("activation_end_fraction") != ACTIVATION_END_FRACTION
            or raw.get("max_onboard_capacity_fraction")
            != MAX_ONBOARD_CAPACITY_FRACTION
            or raw.get("parent_method") != parent_name
            or raw.get("predesignated_candidate")
            is not (variant == "thirty")
            or raw.get("source_protocol_sha256") != payload["v16_protocol_sha256"]
            or raw.get("v17_protocol_sha256") != payload["v17_protocol_sha256"]
        ):
            raise ValueError(f"malformed or duplicate v17 method: {raw!r}")
        checkpoint = _resolve(manifest, raw["checkpoint"])
        parent = dict(raw)
        parent.update(
            {
                "name": f"h2oplus-v14-terminal-quarter-seed{seed}-events000025000",
                "algorithm": "h2oplus-v14-terminal-quarter",
                "artifact_lineage": "v14-terminal-taper",
            }
        )
        v16.v15.v14._validate_v14_artifact(
            parent, checkpoint, payload["v14_protocol_sha256"]
        )
        if sha256(checkpoint) != raw["checkpoint_sha256"]:
            raise ValueError(f"v17 parent checkpoint drifted: {name}")
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
                "artifact_lineage": "v17-tail-service-reserve",
                "parent_artifact_lineage": "v16-headway-reserve",
                "parent_method": parent_name,
                "parent_base_hold_scale": PARENT_BASE_HOLD_SCALE,
                "parent_headway_balance_gain": PARENT_HEADWAY_BALANCE_GAIN,
                "parent_reserve_horizon_fraction": PARENT_RESERVE_HORIZON_FRACTION,
                "tail_service_floor_seconds": floor_seconds,
                "activation_start_fraction": ACTIVATION_START_FRACTION,
                "activation_end_fraction": ACTIVATION_END_FRACTION,
                "max_onboard_capacity_fraction": MAX_ONBOARD_CAPACITY_FRACTION,
            }
        )
    if len(methods) != 15 or len(behavior_keys) != 15:
        raise ValueError("v17 matrix requires exactly 15 tail-service physical behaviors")
    return sorted(methods, key=lambda item: item["name"])


def evaluation_sources(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v16.evaluation_sources(sumo_dir, edge_xml))
    paths.update(
        {
            ROOT / "SimpleSAC/evaluate_transit_v17_tail_service_reserve.py",
            Path(__file__).resolve(),
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def job_command(job: dict) -> list[str]:
    command = v16.v15.v14.job_command(job)
    method = job["method"]
    command.extend(
        (
            "--expected-temperature-tensor-sha256",
            method["temperature_tensor_sha256"],
            "--expected-base-hold-scale",
            str(method["parent_base_hold_scale"]),
            "--expected-headway-balance-gain",
            str(method["parent_headway_balance_gain"]),
            "--expected-parent-reserve-horizon-fraction",
            str(method["parent_reserve_horizon_fraction"]),
            "--expected-tail-service-floor-seconds",
            str(method["tail_service_floor_seconds"]),
            "--expected-activation-start-fraction",
            str(method["activation_start_fraction"]),
            "--expected-activation-end-fraction",
            str(method["activation_end_fraction"]),
            "--expected-max-onboard-capacity-fraction",
            str(method["max_onboard_capacity_fraction"]),
            "--expected-parent-method-name",
            method["parent_method"],
        )
    )
    return command


v16.v15.v14.v13.v12.v11.base.EVALUATORS["v7"] = (
    ROOT / "SimpleSAC/evaluate_transit_v17_tail_service_reserve.py"
)
v16.v15.v14.v13.v12.v11.base.METHODS_SCHEMAS["v7"] = METHODS_SCHEMA
v16.v15.v14.v13.v12.v11.base.load_methods = load_development_methods
v16.v15.v14.v13.v12.v11.base.evaluation_source_paths_v7 = evaluation_sources
v16.v15.v14.v13.v12.v11.base._job_command = job_command
v16.v15.v14.v13.v12.v11.base.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    v16.v15.v14.v13.v12.v11.base.main()
