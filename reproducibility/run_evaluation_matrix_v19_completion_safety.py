#!/usr/bin/env python3
"""Run the frozen v19 completion-safety paired development screen."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256  # noqa: E402
from reproducibility import (  # noqa: E402
    run_evaluation_matrix_v17_tail_service_reserve as v17,
)


METHODS_SCHEMA = "transit-evaluation-methods-v19-completion-safety-v1"
SELECTION_RULE = (
    "sealed v17 twenty-second 25k parents and exact seed-matched actors fixed "
    "before v19; baseline, quarter-load-sixty sole candidate and quarter-load-"
    "fifty diagnostic fixed before any v19 outcome; physical behavior identity "
    "is seed plus actor tensor SHA256 plus the complete inherited v17 transform "
    "and the registered completion-safety reserve"
)
METHOD_NAME = re.compile(
    r"^(?P<algorithm>h2oplus-v19-completion-safety-"
    r"(?P<variant>baseline|quarter-load-fifty|quarter-load-sixty))"
    r"-seed(?P<seed>62\d{2})-events(?P<events>\d{9})$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
V19_PROTOCOL_SHA_ENV = "H2OPLUS_V19_COMPLETION_SAFETY_PROTOCOL_SHA256"
VARIANT_CONFIG = {
    "baseline": {
        "enabled": False,
        "floor_seconds": 20.0,
        "max_onboard_fraction": 0.25,
    },
    "quarter-load-fifty": {
        "enabled": True,
        "floor_seconds": 50.0,
        "max_onboard_fraction": 0.25,
    },
    "quarter-load-sixty": {
        "enabled": True,
        "floor_seconds": 60.0,
        "max_onboard_fraction": 0.25,
    },
}
PARENT_BASE_HOLD_SCALE = 0.125
PARENT_HEADWAY_BALANCE_GAIN = 1.0
PARENT_RESERVE_HORIZON_FRACTION = 0.75
PARENT_TAIL_SERVICE_FLOOR_SECONDS = 20.0
PARENT_ACTIVATION_START_FRACTION = 0.5
PARENT_ACTIVATION_END_FRACTION = 0.75
PARENT_MAX_ONBOARD_CAPACITY_FRACTION = 0.25
ACTIVATION_START_FRACTION = 0.15
ACTIVATION_END_FRACTION = 0.75


def bind_protocol_environment(payload: dict) -> None:
    v17.bind_protocol_environment(payload)
    value = payload["v19_protocol_sha256"]
    existing = os.environ.get(V19_PROTOCOL_SHA_ENV)
    if existing not in (None, value):
        raise ValueError(f"protocol environment conflicts for {V19_PROTOCOL_SHA_ENV}")
    os.environ[V19_PROTOCOL_SHA_ENV] = value


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
            "v19_protocol_sha256",
        )
    )
    if (
        schema != METHODS_SCHEMA
        or payload.get("schema_version") != METHODS_SCHEMA
        or payload.get("study_stage") != "development"
        or payload.get("screen_type") != "paired-mechanism-screen"
        or payload.get("selection_rule") != SELECTION_RULE
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("v19_development_evaluation_outcomes_read") is not False
        or payload.get("physical_method_count") != 15
        or payload.get("logical_method_count") != 15
        or payload.get("expected_new_remote_sumo_episodes") != 135
        or payload.get("predesignated_candidate")
        != "h2oplus-v19-completion-safety-quarter-load-sixty"
        or any(
            not isinstance(value, str) or not SHA256.fullmatch(value)
            for value in protocol_hashes
        )
        or len(payload.get("aliases", [])) != 15
        or len(payload.get("methods", [])) != 15
    ):
        raise ValueError("v19 completion-safety methods manifest scope drifted")
    bind_protocol_environment(payload)

    methods = []
    names = set()
    behavior_keys = set()
    for raw in payload["methods"]:
        name = raw.get("name")
        match = METHOD_NAME.fullmatch(name or "")
        variant = match.group("variant") if match else ""
        registered = VARIANT_CONFIG.get(variant)
        seed = int(match.group("seed")) if match else -1
        v16_parent = (
            f"h2oplus-v16-headway-reserve-three-quarter-seed{seed}-events000025000"
        )
        v17_parent = (
            f"h2oplus-v17-tail-service-reserve-twenty-seed{seed}-events000025000"
        )
        hashes = (
            raw.get("checkpoint_sha256"),
            raw.get("training_run_signature"),
            raw.get("training_admission_sha256"),
            raw.get("training_checkpoint_sha256"),
            raw.get("actor_tensor_sha256"),
            raw.get("temperature_tensor_sha256"),
            raw.get("source_protocol_sha256"),
            raw.get("v17_protocol_sha256"),
            raw.get("v19_protocol_sha256"),
        )
        key = (
            seed,
            raw.get("actor_tensor_sha256"),
            variant,
            raw.get("completion_safety_reserve_enabled"),
            raw.get("completion_safety_reserve_floor_seconds"),
            raw.get("activation_start_fraction"),
            raw.get("activation_end_fraction"),
            raw.get("max_onboard_capacity_fraction"),
        )
        if (
            match is None
            or registered is None
            or name in names
            or key in behavior_keys
            or any(
                not isinstance(value, str) or not SHA256.fullmatch(value)
                for value in hashes
            )
            or raw.get("type") != "checkpoint"
            or raw.get("artifact_lineage") != "v19-completion-safety"
            or raw.get("parent_artifact_lineage") != "v17-tail-service-reserve"
            or raw.get("training_seed") != seed
            or raw.get("online_events") != 25_000
            or int(match.group("events")) != 25_000
            or raw.get("algorithm") != match.group("algorithm")
            or raw.get("v16_parent_method") != v16_parent
            or raw.get("parent_method") != v17_parent
            or raw.get("parent_base_hold_scale") != PARENT_BASE_HOLD_SCALE
            or raw.get("parent_headway_balance_gain")
            != PARENT_HEADWAY_BALANCE_GAIN
            or raw.get("parent_reserve_horizon_fraction")
            != PARENT_RESERVE_HORIZON_FRACTION
            or raw.get("parent_tail_service_floor_seconds")
            != PARENT_TAIL_SERVICE_FLOOR_SECONDS
            or raw.get("parent_activation_start_fraction")
            != PARENT_ACTIVATION_START_FRACTION
            or raw.get("parent_activation_end_fraction")
            != PARENT_ACTIVATION_END_FRACTION
            or raw.get("parent_max_onboard_capacity_fraction")
            != PARENT_MAX_ONBOARD_CAPACITY_FRACTION
            or raw.get("completion_safety_reserve_enabled")
            is not registered["enabled"]
            or raw.get("completion_safety_reserve_floor_seconds")
            != registered["floor_seconds"]
            or raw.get("activation_start_fraction") != ACTIVATION_START_FRACTION
            or raw.get("activation_end_fraction") != ACTIVATION_END_FRACTION
            or raw.get("max_onboard_capacity_fraction")
            != registered["max_onboard_fraction"]
            or raw.get("predesignated_candidate")
            is not (variant == "quarter-load-sixty")
            or raw.get("source_protocol_sha256") != payload["v16_protocol_sha256"]
            or raw.get("v17_protocol_sha256") != payload["v17_protocol_sha256"]
            or raw.get("v19_protocol_sha256") != payload["v19_protocol_sha256"]
        ):
            raise ValueError(f"malformed or duplicate v19 method: {raw!r}")
        checkpoint = _resolve(manifest, raw["checkpoint"])
        parent = dict(raw)
        parent.update(
            {
                "name": f"h2oplus-v14-terminal-quarter-seed{seed}-events000025000",
                "algorithm": "h2oplus-v14-terminal-quarter",
                "artifact_lineage": "v14-terminal-taper",
            }
        )
        v17.v16.v15.v14._validate_v14_artifact(
            parent, checkpoint, payload["v14_protocol_sha256"]
        )
        if sha256(checkpoint) != raw["checkpoint_sha256"]:
            raise ValueError(f"v19 parent checkpoint drifted: {name}")
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
                "artifact_lineage": "v19-completion-safety",
                "parent_artifact_lineage": "v17-tail-service-reserve",
                "parent_method": v16_parent,
                "v17_parent_method": v17_parent,
                "parent_base_hold_scale": PARENT_BASE_HOLD_SCALE,
                "parent_headway_balance_gain": PARENT_HEADWAY_BALANCE_GAIN,
                "parent_reserve_horizon_fraction": PARENT_RESERVE_HORIZON_FRACTION,
                "tail_service_floor_seconds": PARENT_TAIL_SERVICE_FLOOR_SECONDS,
                "activation_start_fraction": PARENT_ACTIVATION_START_FRACTION,
                "activation_end_fraction": PARENT_ACTIVATION_END_FRACTION,
                "max_onboard_capacity_fraction": (
                    PARENT_MAX_ONBOARD_CAPACITY_FRACTION
                ),
                "v19_reserve_enabled": registered["enabled"],
                "v19_reserve_floor_seconds": registered["floor_seconds"],
                "v19_activation_start_fraction": ACTIVATION_START_FRACTION,
                "v19_activation_end_fraction": ACTIVATION_END_FRACTION,
                "v19_max_onboard_fraction": registered["max_onboard_fraction"],
            }
        )
    if len(methods) != 15 or len(behavior_keys) != 15:
        raise ValueError("v19 matrix requires exactly 15 physical behaviors")
    return sorted(methods, key=lambda item: item["name"])


def evaluation_sources(sumo_dir: Path, edge_xml: Path) -> tuple[Path, ...]:
    paths = set(v17.evaluation_sources(sumo_dir, edge_xml))
    paths.update(
        {
            ROOT / "SimpleSAC/evaluate_transit_v19_completion_safety_reserve.py",
            Path(__file__).resolve(),
        }
    )
    return tuple(sorted(path.resolve() for path in paths))


def job_command(job: dict) -> list[str]:
    command = v17.job_command(job)
    method = job["method"]
    command.extend(
        (
            "--expected-v19-parent-floor-seconds",
            str(method["tail_service_floor_seconds"]),
            "--expected-v19-reserve-enabled",
            "1" if method["v19_reserve_enabled"] else "0",
            "--expected-v19-reserve-floor-seconds",
            str(method["v19_reserve_floor_seconds"]),
            "--expected-v19-activation-start-fraction",
            str(method["v19_activation_start_fraction"]),
            "--expected-v19-activation-end-fraction",
            str(method["v19_activation_end_fraction"]),
            "--expected-v19-max-onboard-fraction",
            str(method["v19_max_onboard_fraction"]),
            "--expected-v19-parent-method-name",
            method["v17_parent_method"],
        )
    )
    return command


base = v17.v16.v15.v14.v13.v12.v11.base
base.EVALUATORS["v7"] = (
    ROOT / "SimpleSAC/evaluate_transit_v19_completion_safety_reserve.py"
)
base.METHODS_SCHEMAS["v7"] = METHODS_SCHEMA
base.load_methods = load_development_methods
base.evaluation_source_paths_v7 = evaluation_sources
base._job_command = job_command
base.__file__ = str(Path(__file__).resolve())


if __name__ == "__main__":
    base.main()
