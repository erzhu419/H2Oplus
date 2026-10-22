#!/usr/bin/env python3
"""Run one registered v10 H2O+ development intervention.

The wrapper preserves the v7 SMDP, rollout and hard-event-cap implementation,
selects one isolated v10 trainer subclass, consumes a sealed v8 offline
initializer, and exports every preregistered few-shot milestone.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ENV = "H2OPLUS_V10_WORKSPACE_ROOT"
PROTOCOL_ENV = "H2OPLUS_V10_PROTOCOL"
EXPECTED_PROTOCOL_ENV = "H2OPLUS_V10_PROTOCOL_SHA256"
BASE_DEPLOYMENT_ENV = "H2OPLUS_V10_BASE_DEPLOYMENT_MANIFEST"
EXPECTED_BASE_DEPLOYMENT_ENV = (
    "H2OPLUS_V10_BASE_DEPLOYMENT_MANIFEST_SHA256"
)
ADMISSION_SCHEMA = "transit-training-admission-v10-delayed-development-v4"
PROTOCOL_SCHEMA = "transit-h2oplus-v10-delayed-protocol-v4"
POLICY_MANIFEST_SCHEMA = "transit-h2oplus-v10-delayed-policy-manifest-v1"
EXPORT_SUMMARY_SCHEMA = "transit-h2oplus-v10-delayed-export-summary-v1"
HANDOFF_SCHEMA = "transit-h2oplus-v10-prefix-handoff-v1"
BRANCH_SCHEMA = "transit-h2oplus-v10-branch-materialization-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _required_path(variable: str) -> Path:
    raw = os.environ.get(variable)
    if not raw:
        raise RuntimeError(f"required environment variable is absent: {variable}")
    return Path(raw).expanduser().resolve(strict=True)


def _required_sha(variable: str) -> str:
    value = os.environ.get(variable, "")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"required SHA-256 environment variable is invalid: {variable}")
    return value


def _argument_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError as error:
        raise ValueError(f"v10 command must explicitly provide {name}") from error
    if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
        raise ValueError(f"v10 command lacks the value for {name}")
    return sys.argv[index + 1]


def _optional_argument_value(name: str) -> str | None:
    if name not in sys.argv:
        return None
    return _argument_value(name)


def _remove_argument(name: str) -> str | None:
    if name not in sys.argv:
        return None
    index = sys.argv.index(name)
    if index + 1 >= len(sys.argv):
        raise ValueError(f"v10 command lacks the value for {name}")
    value = sys.argv[index + 1]
    del sys.argv[index : index + 2]
    return value


def _command_without_resume(arguments: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(arguments):
        if arguments[index] == "--resume":
            if index + 1 >= len(arguments):
                raise ValueError("--resume lacks its checkpoint path")
            index += 2
            continue
        result.append(arguments[index])
        index += 1
    return result


def _actor_digest(state: dict[str, Any]) -> str:
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


def _load_protocol() -> tuple[Path, dict[str, Any]]:
    protocol_path = _required_path(PROTOCOL_ENV)
    expected = _required_sha(EXPECTED_PROTOCOL_ENV)
    if sha256(protocol_path) != expected:
        raise ValueError("v10 protocol hash drifted")
    protocol = _json(protocol_path)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("study_stage") != "development"
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("local_training_or_sumo_allowed") is not False
    ):
        raise ValueError("v10 protocol scope drifted")
    deployment_path = _required_path(BASE_DEPLOYMENT_ENV)
    expected_deployment = _required_sha(EXPECTED_BASE_DEPLOYMENT_ENV)
    if sha256(deployment_path) != expected_deployment:
        raise ValueError("frozen v7 deployment manifest hash drifted")
    bound = protocol.get("base_v7_training_deployment", {})
    if bound.get("sha256") != expected_deployment:
        raise ValueError("v10 protocol does not bind the base v7 deployment")
    return protocol_path, protocol


PROTOCOL_PATH, PROTOCOL = _load_protocol()
WORKSPACE = _required_path(WORKSPACE_ENV)
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import SimpleSAC.train_transit_v3 as base  # noqa: E402
import SimpleSAC.transit_v7.trainer as v7_trainer_module  # noqa: E402
from SimpleSAC.transit_v10.trainer import (  # noqa: E402
    V10_TRAINER_VARIANTS,
    authorize_registered_initializer_bridge,
)
from SimpleSAC.transit_v10.provenance import (  # noqa: E402
    checkpoint_state_sha256 as _checkpoint_state_sha256,
)


RUN_SIGNATURE = _argument_value("--run-signature")
RUN_JOB = next(
    (
        job
        for job in PROTOCOL.get("training_jobs", [])
        if job.get("run_signature") == RUN_SIGNATURE
    ),
    None,
)
if RUN_JOB is None:
    raise ValueError("run signature is not registered in the v10 protocol")
if _command_without_resume(sys.argv[1:]) != RUN_JOB.get("command_arguments"):
    raise ValueError("runtime command differs from the frozen v10 job")
VARIANT = str(RUN_JOB["variant"])
STAGE = str(RUN_JOB["stage"])
BRANCH_HANDOFF_ARGUMENT = _optional_argument_value("--branch-handoff-manifest")
if (STAGE == "branch") != (BRANCH_HANDOFF_ARGUMENT is not None):
    raise ValueError("v10 branch handoff argument disagrees with registered stage")
_remove_argument("--branch-handoff-manifest")
try:
    TRAINER_CLASS = V10_TRAINER_VARIANTS[VARIANT]
except KeyError as error:
    raise ValueError(f"unregistered v10 variant: {VARIANT}") from error


def _stable_object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _descriptor(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    return {
        "path": (
            resolved.relative_to(relative_to.resolve()).as_posix()
            if relative_to is not None
            else str(resolved)
        ),
        "sha256": sha256(resolved),
        "bytes": resolved.stat().st_size,
    }


def _verify_descriptor(root: Path, value: dict[str, Any]) -> Path:
    relative = Path(str(value.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("v10 handoff contains an unsafe path")
    path = (root / relative).resolve(strict=True)
    if (
        path.stat().st_size != int(value.get("bytes", -1))
        or sha256(path) != value.get("sha256")
    ):
        raise ValueError(f"v10 handoff file drifted: {path}")
    return path


def _create_prefix_handoff(
    *,
    run_dir: Path,
    runtime_checkpoint: Path,
    checkpoint_extra: dict[str, Any],
    actor_tensor_sha256: str,
) -> Path:
    if STAGE != "prefix" or VARIANT != "shared-prefix":
        raise RuntimeError("only a shared-prefix run may create a v10 handoff")
    runtime = checkpoint_extra.get("runtime_state")
    if not isinstance(runtime, dict) or not {"path", "sha256"}.issubset(runtime):
        raise RuntimeError("v10 prefix handoff lacks simulator runtime state")
    runtime_source = (run_dir / runtime["path"]).resolve(strict=True)
    if sha256(runtime_source) != runtime["sha256"]:
        raise RuntimeError("v10 prefix runtime state drifted before handoff")

    destination = run_dir / "v10_handoff"
    if destination.exists():
        manifest_path = destination / "manifest.json"
        manifest = _json(manifest_path)
        if (
            manifest.get("schema_version") != HANDOFF_SCHEMA
            or manifest.get("prefix_run_signature") != RUN_SIGNATURE
            or manifest.get("protocol_sha256") != sha256(PROTOCOL_PATH)
            or manifest.get("actor_tensor_sha256") != actor_tensor_sha256
        ):
            raise RuntimeError("existing v10 handoff differs from the prefix run")
        for value in manifest.get("files", []):
            _verify_descriptor(destination, value)
        return manifest_path

    temporary = run_dir / f"v10_handoff.partial.{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "checkpoints").mkdir(parents=True)
    sources = {
        "metadata.json": run_dir / "metadata.json",
        "metrics.jsonl": run_dir / "metrics.jsonl",
        "phase_metrics.json": run_dir / "phase_metrics.json",
        "checkpoints/latest.pt": runtime_checkpoint,
        f"checkpoints/{runtime_source.name}": runtime_source,
    }
    for relative, source in sources.items():
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError(f"v10 handoff copy drifted: {relative}")
    files = [
        _descriptor(temporary / relative, relative_to=temporary)
        for relative in sorted(sources)
    ]
    manifest = {
        "schema_version": HANDOFF_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "prefix_run_name": RUN_JOB["name"],
        "prefix_run_signature": RUN_SIGNATURE,
        "seed": RUN_JOB["seed"],
        "online_events_nominal": 10_000,
        "learner_update_steps": int(RUN_JOB["expected_total_updates"]),
        "actor_tensor_sha256": actor_tensor_sha256,
        "checkpoint_state_sha256": _checkpoint_state_sha256(
            temporary / "checkpoints/latest.pt"
        ),
        "protocol_sha256": sha256(PROTOCOL_PATH),
        "v10_source_sha256": _v10_source_sha256(),
        "files": files,
    }
    _write_json(temporary / "manifest.json", manifest)
    os.replace(temporary, destination)
    return destination / "manifest.json"


def _branch_training_admission() -> dict[str, Any]:
    path = Path(RUN_JOB["training_admission"]).resolve(strict=True)
    payload = _json(path)
    if (
        payload.get("schema_version") != ADMISSION_SCHEMA
        or payload.get("stage") != "branch"
        or payload.get("variant") != VARIANT
        or payload.get("run_signature") != RUN_SIGNATURE
    ):
        raise ValueError("v10 branch admission is stale or misbound")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "schema_version": payload["schema_version"],
        "study_stage": "development",
        "selected_arguments": payload.get("selected_arguments"),
        "v10_protocol_sha256": sha256(PROTOCOL_PATH),
        "v10_source_sha256": _v10_source_sha256(),
        "variant": VARIANT,
        "stage": STAGE,
    }


def _verify_branch_materialization(
    branch_dir: Path,
    handoff_manifest: Path,
) -> Path:
    record_path = branch_dir / "v10_branch_materialization.json"
    record = _json(record_path)
    checkpoint = branch_dir / "checkpoints/latest.pt"
    initial_checkpoint = branch_dir / "v10_branch_source/latest.pt"
    if (
        record.get("schema_version") != BRANCH_SCHEMA
        or record.get("run_signature") != RUN_SIGNATURE
        or record.get("variant") != VARIANT
        or record.get("handoff_manifest_sha256") != sha256(handoff_manifest)
        or record.get("protocol_sha256") != sha256(PROTOCOL_PATH)
        or not checkpoint.is_file()
        or not initial_checkpoint.is_file()
        or record.get("branch_initial_checkpoint_sha256")
        != sha256(initial_checkpoint)
        or record.get("checkpoint_state_sha256")
        != _checkpoint_state_sha256(initial_checkpoint)
    ):
        raise ValueError("existing v10 branch materialization drifted")
    return checkpoint


def _materialize_branch() -> Path | None:
    if STAGE != "branch":
        return None
    assert BRANCH_HANDOFF_ARGUMENT is not None
    handoff_path = Path(BRANCH_HANDOFF_ARGUMENT).expanduser().resolve(strict=True)
    if handoff_path != Path(RUN_JOB["branch_handoff_manifest"]).resolve():
        raise ValueError("runtime v10 handoff path differs from the protocol")
    handoff_root = handoff_path.parent
    handoff = _json(handoff_path)
    expected_prefix = RUN_JOB["prefix"]
    if (
        handoff.get("schema_version") != HANDOFF_SCHEMA
        or handoff.get("passed") is not True
        or handoff.get("formal_held_out_outcomes_read") is not False
        or handoff.get("prefix_run_signature") != expected_prefix["run_signature"]
        or handoff.get("seed") != RUN_JOB["seed"]
        or handoff.get("online_events_nominal") != 10_000
        or handoff.get("protocol_sha256") != sha256(PROTOCOL_PATH)
        or handoff.get("v10_source_sha256") != _v10_source_sha256()
    ):
        raise ValueError("v10 branch handoff is stale or misbound")
    files = {
        value["path"]: _verify_descriptor(handoff_root, value)
        for value in handoff.get("files", [])
    }
    required = {
        "metadata.json",
        "metrics.jsonl",
        "phase_metrics.json",
        "checkpoints/latest.pt",
    }
    if not required.issubset(files):
        raise ValueError("v10 handoff is missing required files")
    source_checkpoint = files["checkpoints/latest.pt"]
    if (
        _checkpoint_state_sha256(source_checkpoint)
        != handoff.get("checkpoint_state_sha256")
    ):
        raise ValueError("v10 handoff checkpoint state drifted")

    output_root = Path(_argument_value("--output-root")).resolve()
    branch_dir = output_root / _argument_value("--run-name")
    if branch_dir.exists():
        return _verify_branch_materialization(branch_dir, handoff_path)
    temporary = branch_dir.with_name(f"{branch_dir.name}.partial.{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "checkpoints").mkdir(parents=True)
    for relative in required:
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(files[relative], target)
    runtime_entries = [name for name in files if name.startswith("checkpoints/runtime_")]
    if len(runtime_entries) != 1:
        raise ValueError("v10 handoff must contain exactly one simulator runtime")
    runtime_relative = runtime_entries[0]
    shutil.copy2(files[runtime_relative], temporary / runtime_relative)

    metadata_path = temporary / "metadata.json"
    metadata = _json(metadata_path)
    metadata["arguments"].update(
        {
            "run_name": RUN_JOB["name"],
            "run_signature": RUN_SIGNATURE,
            "epochs": RUN_JOB["epochs"],
            "training_admission": RUN_JOB["training_admission"],
            "output_root": str(output_root),
            "resume": str(branch_dir / "checkpoints/latest.pt"),
        }
    )
    metadata["run_signature"] = RUN_SIGNATURE
    metadata["training_admission"] = _branch_training_admission()
    metadata["effective_schedule"]["online_epochs"] = RUN_JOB["epochs"]
    metadata["effective_schedule"]["simulator_decision_event_cap"] = RUN_JOB[
        "online_event_cap"
    ]
    metadata["v10_branch_handoff"] = {
        "schema_version": BRANCH_SCHEMA,
        "source_manifest": _descriptor(handoff_path),
        "source_prefix_run_signature": expected_prefix["run_signature"],
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "checkpoint_state_sha256": handoff["checkpoint_state_sha256"],
        "branch_run_signature": RUN_SIGNATURE,
        "variant": VARIANT,
    }
    _write_json(metadata_path, metadata)

    import torch

    checkpoint_path = temporary / "checkpoints/latest.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_before = _checkpoint_state_sha256(checkpoint_path)
    checkpoint_extra = dict(checkpoint["extra"])
    checkpoint_extra.update(
        {
            "phase": "online_training",
            "run_signature": RUN_SIGNATURE,
            "run_dir": str(branch_dir),
            "v10_branch_handoff": metadata["v10_branch_handoff"],
        }
    )
    checkpoint["extra"] = checkpoint_extra
    checkpoint_temporary = checkpoint_path.with_suffix(".pt.partial")
    torch.save(checkpoint, checkpoint_temporary)
    os.replace(checkpoint_temporary, checkpoint_path)
    state_after = _checkpoint_state_sha256(checkpoint_path)
    if state_before != state_after or state_after != handoff["checkpoint_state_sha256"]:
        raise RuntimeError("v10 branch materialization changed checkpoint state")
    branch_source = temporary / "v10_branch_source"
    branch_source.mkdir()
    shutil.copy2(checkpoint_path, branch_source / "latest.pt")
    shutil.copy2(temporary / runtime_relative, branch_source / Path(runtime_relative).name)
    record = {
        "schema_version": BRANCH_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed": True,
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
        "run_name": RUN_JOB["name"],
        "run_signature": RUN_SIGNATURE,
        "variant": VARIANT,
        "seed": RUN_JOB["seed"],
        "handoff_manifest_sha256": sha256(handoff_path),
        "source_checkpoint_sha256": sha256(source_checkpoint),
        "branch_initial_checkpoint_sha256": sha256(branch_source / "latest.pt"),
        "branch_initial_runtime_sha256": sha256(
            branch_source / Path(runtime_relative).name
        ),
        "checkpoint_state_sha256": state_after,
        "actor_tensor_sha256": handoff["actor_tensor_sha256"],
        "protocol_sha256": sha256(PROTOCOL_PATH),
    }
    _write_json(temporary / "v10_branch_materialization.json", record)
    branch_dir.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temporary, branch_dir)
    return branch_dir / "checkpoints/latest.pt"


def _v10_source_sha256() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        WORKSPACE / "SimpleSAC/transit_v10/__init__.py",
        WORKSPACE / "SimpleSAC/transit_v10/provenance.py",
        WORKSPACE / "SimpleSAC/transit_v10/trainer.py",
    )
    return {
        path.relative_to(WORKSPACE).as_posix(): sha256(path)
        for path in paths
    }


def validate_development_admission(
    path: Path,
    *,
    dataset_sha256: str,
    simulator: dict[str, Any],
    sources: dict[str, str],
) -> dict[str, Any]:
    admission_path = path.expanduser().resolve(strict=True)
    payload = _json(admission_path)
    wrapper = Path(__file__).resolve()
    expected = {
        "method": _argument_value("--method"),
        "seed": int(_argument_value("--seed")),
        "run_name": _argument_value("--run-name"),
        "run_signature": RUN_SIGNATURE,
    }
    if (
        payload.get("schema_version") != ADMISSION_SCHEMA
        or payload.get("study_stage") != "development"
        or payload.get("passed") is not True
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("dataset", {}).get("sha256") != dataset_sha256
        or payload.get("simulator", {}).get("aggregate_sha256")
        != simulator.get("aggregate_sha256")
        or payload.get("simulator", {}).get("runtime_sha256")
        != simulator.get("runtime_sha256")
        or payload.get("training_source_sha256") != sources
        or payload.get("v10_source_sha256") != _v10_source_sha256()
        or payload.get("v10_protocol", {}).get("sha256")
        != sha256(PROTOCOL_PATH)
        or payload.get("v10_wrapper", {}).get("sha256") != sha256(wrapper)
        or payload.get("variant") != VARIANT
        or payload.get("stage") != STAGE
        or any(payload.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("v10 development admission is stale or misbound")
    if admission_path != Path(RUN_JOB["training_admission"]).resolve():
        raise ValueError("runtime admission path differs from the registered job")
    return {
        "path": str(admission_path),
        "sha256": sha256(admission_path),
        "bytes": admission_path.stat().st_size,
        "schema_version": payload["schema_version"],
        "study_stage": "development",
        "selected_arguments": payload.get("selected_arguments"),
        "v10_protocol_sha256": sha256(PROTOCOL_PATH),
        "v10_source_sha256": _v10_source_sha256(),
        "variant": VARIANT,
        "stage": STAGE,
    }


base.validate_v7_training_admission_runtime = validate_development_admission


def validate_registered_offline_initializer(
    path: Path,
    extra: dict[str, Any],
    *,
    method: str,
    seed: int,
    offline_updates: int,
    trainer_total_updates: int,
    dataset_sha256: str,
    simulator: dict[str, Any],
    sources: dict[str, str],
    contract_config: dict[str, Any],
    training_admission: dict[str, Any] | None,
    expected_source_signature: str,
) -> dict[str, Any]:
    del training_admission
    initializer = path.expanduser().resolve(strict=True)
    registered = RUN_JOB["offline_initializer"]
    expected = {
        "phase": "offline_pretraining",
        "source_method": "h2oplus",
        "source_run_name": f"h2oplus_seed{seed}",
        "source_run_signature": expected_source_signature,
        "source_training_seed": seed,
        "offline_updates_completed": offline_updates,
        "trainer_total_updates": trainer_total_updates,
        "dataset_sha256": dataset_sha256,
        "simulator_runtime_sha256": simulator["runtime_sha256"],
        "simulator_aggregate_sha256": simulator["aggregate_sha256"],
        "source_sha256": sources,
        "contract_config": contract_config,
        "mdp_version": "v7",
    }
    if (
        method != "h2oplus"
        or initializer != Path(registered["path"]).resolve()
        or sha256(initializer) != registered["sha256"]
        or _stable_object_sha256(extra) != registered["extra_sha256"]
        or any(extra.get(key) != value for key, value in expected.items())
        or not isinstance(extra.get("offline_metrics"), dict)
    ):
        raise ValueError("v10 offline initializer is stale or misbound")
    return {
        "path": str(initializer),
        "sha256": sha256(initializer),
        "bytes": initializer.stat().st_size,
        "schema_version": "transit-offline-initializer-v7-v1",
        "source_method": "h2oplus",
        "source_run_name": extra["source_run_name"],
        "source_run_signature": expected_source_signature,
        "source_training_seed": seed,
        "offline_updates": offline_updates,
        "v10_reuse_authorized": True,
    }


base.validate_v7_offline_initializer_runtime = (
    validate_registered_offline_initializer
)
authorize_registered_initializer_bridge(
    base,
    method=_argument_value("--method"),
    validator=validate_registered_offline_initializer,
)
v7_trainer_module.TransitH2OTrainerV7 = TRAINER_CLASS


def _milestone(*, phase: str, epoch: int, metadata: dict[str, Any]) -> tuple[int, int] | None:
    arguments = metadata["arguments"]
    warmup = int(arguments["warmup_events"])
    per_epoch = int(arguments["events_per_epoch"])
    target_epochs = int(arguments["epochs"])
    target_events = warmup + target_epochs * per_epoch
    if phase == "offline_initializer_loaded":
        return 0, int(arguments["offline_updates"])
    if phase == "online_warmup":
        return warmup, int(arguments["offline_updates"])
    if phase == "online_training":
        nominal = warmup + epoch * per_epoch
        if nominal == target_events:
            return None
        allowed = set(PROTOCOL["policy_budgets"][VARIANT]["export_online_events"])
        if nominal in allowed:
            return nominal, int(arguments["offline_updates"]) + epoch * int(
                arguments["updates_per_epoch"]
            )
    if phase == "complete":
        allowed = set(PROTOCOL["policy_budgets"][VARIANT]["export_online_events"])
        if target_events in allowed:
            return target_events, int(arguments["offline_updates"]) + target_epochs * int(
                arguments["updates_per_epoch"]
            )
    return None


def _actual_events(run_dir: Path, phase: str, nominal: int) -> int:
    if phase in {"initialized", "offline_initializer_loaded"}:
        return 0
    if phase == "online_warmup":
        phase_metrics = _json(run_dir / "phase_metrics.json")
        return int(phase_metrics.get("warmup", {}).get("events", nominal))
    metrics_path = run_dir / "metrics.jsonl"
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line]
    if not rows:
        raise ValueError("online milestone lacks committed epoch metrics")
    return int(rows[-1]["actual_cumulative_simulator_decision_events"])


def _event_cap_evidence(
    run_dir: Path, phase: str, nominal: int
) -> dict[str, Any]:
    actual = _actual_events(run_dir, phase, nominal)
    if nominal == 0:
        if actual != 0:
            raise RuntimeError("zero-event milestone contains online interaction")
        return {
            "actual": 0,
            "deviation": 0,
            "maximum_simultaneous_decisions": 0,
            "hard_event_cap_reached": False,
            "blocked_simultaneous_batch_size": 0,
        }
    phase_metrics = _json(run_dir / "phase_metrics.json")
    warmup = phase_metrics.get("warmup", {})
    rows = []
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.is_file():
        rows = [
            json.loads(line)
            for line in metrics_path.read_text().splitlines()
            if line
        ]
    maximum_batch = max(
        [int(warmup.get("maximum_simultaneous_decisions", 0))]
        + [
            int(row.get("rollout", {}).get("maximum_simultaneous_decisions", 0))
            for row in rows
        ]
    )
    latest_rollout = rows[-1].get("rollout", {}) if rows else warmup
    deviation = actual - nominal
    if deviation > 0 and (maximum_batch <= 0 or deviation >= maximum_batch):
        raise RuntimeError("milestone event overshoot is not bounded by one atomic batch")
    if phase == "complete":
        underfill = nominal - actual
        blocked = int(latest_rollout.get("blocked_simultaneous_batch_size", 0))
        hard_cap = bool(latest_rollout.get("hard_event_cap_reached", False))
        if actual > nominal:
            raise RuntimeError("completed milestone exceeded its hard event cap")
        if underfill and (not hard_cap or blocked <= underfill):
            raise RuntimeError("completed milestone underfill lacks atomic-batch evidence")
    return {
        "actual": actual,
        "deviation": deviation,
        "maximum_simultaneous_decisions": maximum_batch,
        "hard_event_cap_reached": bool(
            latest_rollout.get("hard_event_cap_reached", False)
        ),
        "blocked_simultaneous_batch_size": int(
            latest_rollout.get("blocked_simultaneous_batch_size", 0)
        ),
    }


def _ratio_optimizer_steps(nominal_events: int) -> int:
    warmup = int(RUN_JOB["warmup_events"])
    if nominal_events <= warmup:
        return 0
    online_learner_steps = nominal_events - warmup
    interval = int(_argument_value("--ratio-update-interval"))
    pretrain = int(_argument_value("--ratio-pretrain-updates"))
    return pretrain + (online_learner_steps + interval - 1) // interval


def _existing_actor_digest(path: Path) -> tuple[str, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _actor_digest(payload["actor"]), dict(payload.get("extra", {}))


def _update_policy_manifest(run_dir: Path) -> None:
    policy_root = run_dir / "v10_policies"
    entries = []
    for actor in sorted(policy_root.glob("online_events_*.pt")):
        digest, extra = _existing_actor_digest(actor)
        checkpoint_descriptor = extra.get("v10_model_checkpoint", {})
        checkpoint = run_dir / str(checkpoint_descriptor.get("path", ""))
        if (
            not checkpoint.is_file()
            or sha256(checkpoint) != checkpoint_descriptor.get("sha256")
        ):
            raise RuntimeError(f"v10 source checkpoint drifted: {checkpoint}")
        entries.append(
            {
                "path": actor.relative_to(run_dir).as_posix(),
                "sha256": sha256(actor),
                "bytes": actor.stat().st_size,
                "actor_tensor_sha256": digest,
                "online_events_nominal": extra["v10_online_events_nominal"],
                "online_events_actual": extra["v10_online_events_actual"],
                "event_budget_deviation": extra["v10_event_budget_deviation"],
                "learner_update_steps": extra["v10_learner_update_steps"],
                "ratio_optimizer_steps": extra["v10_ratio_optimizer_steps"],
                "training_checkpoint_sha256": extra["training_checkpoint_sha256"],
                "model_checkpoint": checkpoint_descriptor,
                "phase": extra["phase"],
                "epoch": extra["epoch"],
                "stage": extra["stage"],
            }
        )
    _write_json(
        policy_root / "policy_manifest.json",
        {
            "schema_version": POLICY_MANIFEST_SCHEMA,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "method": RUN_JOB["method"],
            "variant": VARIANT,
            "stage": STAGE,
            "seed": RUN_JOB["seed"],
            "run_signature": RUN_SIGNATURE,
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "entries": entries,
        },
    )


ORIGINAL_SAVE = TRAINER_CLASS.save


def save_with_fewshot_export(
    self,
    path: str | Path,
    extra: dict[str, Any] | None = None,
    *,
    include_online_replay: bool = True,
) -> None:
    ORIGINAL_SAVE(
        self,
        path,
        extra=extra,
        include_online_replay=include_online_replay,
    )
    if not extra:
        return
    run_dir = Path(extra["run_dir"]).resolve()
    metadata = _json(run_dir / "metadata.json")
    phase = str(extra["phase"])
    epoch = int(extra["epoch"])
    resolved = _milestone(phase=phase, epoch=epoch, metadata=metadata)
    if resolved is None:
        return
    nominal_events, learner_steps = resolved
    event_evidence = _event_cap_evidence(run_dir, phase, nominal_events)
    runtime_checkpoint_path = Path(path).resolve(strict=True)
    actor_state = self.actor.state_dict()
    actor_digest = _actor_digest(actor_state)
    compact_checkpoint = (
        run_dir
        / "v10_checkpoints"
        / f"online_events_{nominal_events:09d}.pt"
    )
    compact_extra = dict(extra)
    compact_extra.update(
        {
            "runtime_state": None,
            "v10_checkpoint_scope": (
                "model, optimizers, RNG and counters; online replay and simulator "
                "runtime state intentionally excluded; not a resume checkpoint"
            ),
            "v10_online_events_nominal": nominal_events,
            "v10_online_events_actual": event_evidence["actual"],
            "v10_learner_update_steps": learner_steps,
            "v10_ratio_optimizer_steps": _ratio_optimizer_steps(nominal_events),
            "v10_intervention": self.v10_intervention,
        }
    )
    if compact_checkpoint.exists():
        import torch

        compact_payload = torch.load(
            compact_checkpoint, map_location="cpu", weights_only=False
        )
        if (
            _actor_digest(compact_payload.get("actor", {})) != actor_digest
            or compact_payload.get("extra", {}).get(
                "v10_online_events_nominal"
            )
            != nominal_events
            or int(compact_payload.get("total_updates", -1)) != learner_steps
        ):
            raise RuntimeError(
                f"existing v10 model checkpoint drifted: {compact_checkpoint}"
            )
    else:
        ORIGINAL_SAVE(
            self,
            compact_checkpoint,
            extra=compact_extra,
            include_online_replay=False,
        )
    compact_checkpoint_descriptor = {
        "path": compact_checkpoint.relative_to(run_dir).as_posix(),
        "sha256": sha256(compact_checkpoint),
        "bytes": compact_checkpoint.stat().st_size,
        "scope": compact_extra["v10_checkpoint_scope"],
    }
    destination = (
        run_dir
        / "v10_policies"
        / f"online_events_{nominal_events:09d}.pt"
    )
    policy_extra = {
        "epoch": epoch,
        "phase": phase,
        "run_signature": extra["run_signature"],
        "method": extra["method"],
        "variant": VARIANT,
        "stage": STAGE,
        "method_profile": extra["method_profile"],
        "training_seed": extra["training_seed"],
        "run_dir": str(run_dir),
        "dataset_sha256": extra["dataset_sha256"],
        "simulator_runtime_sha256": extra["simulator_runtime_sha256"],
        "simulator_aggregate_sha256": extra["simulator_aggregate_sha256"],
        "source_sha256": extra["source_sha256"],
        "contract_config": extra["contract_config"],
        "mdp_version": extra["mdp_version"],
        "trainer_config": extra["trainer_config"],
        "total_updates": int(self.total_updates),
        "training_checkpoint_sha256": compact_checkpoint_descriptor["sha256"],
        "v10_model_checkpoint": compact_checkpoint_descriptor,
        "v10_runtime_checkpoint_sha256_at_export": sha256(
            runtime_checkpoint_path
        ),
        "training_admission": metadata["training_admission"],
        "offline_initializer_consumed": extra["offline_initializer_consumed"],
        "v10_online_events_nominal": nominal_events,
        "v10_online_events_actual": event_evidence["actual"],
        "v10_event_budget_deviation": event_evidence["deviation"],
        "v10_event_cap_evidence": event_evidence,
        "v10_learner_update_steps": learner_steps,
        "v10_ratio_optimizer_steps": _ratio_optimizer_steps(nominal_events),
        "v10_actor_tensor_sha256": actor_digest,
        "v10_intervention": self.v10_intervention,
        "v10_source_sha256": _v10_source_sha256(),
        "v10_protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": sha256(PROTOCOL_PATH),
        },
        "v10_wrapper": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "selection_rule": (
            "all preregistered v10 online-event milestones; no evaluation outcome "
            "selects a checkpoint"
        ),
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    if learner_steps != int(self.total_updates):
        raise RuntimeError("v10 learner-step accounting disagrees with trainer")
    if destination.exists():
        existing_digest, existing_extra = _existing_actor_digest(destination)
        if (
            existing_digest != actor_digest
            or existing_extra.get("v10_online_events_nominal") != nominal_events
            or existing_extra.get("run_signature") != extra["run_signature"]
        ):
            raise RuntimeError(f"existing v10 milestone actor drifted: {destination}")
    else:
        self.save_policy_artifact(destination, extra=policy_extra)
    if nominal_events == int(metadata["arguments"]["warmup_events"]):
        zero = run_dir / "v10_policies" / "online_events_000000000.pt"
        zero_digest, _ = _existing_actor_digest(zero)
        if zero_digest != actor_digest:
            raise RuntimeError("v10 0-event and post-warmup actors differ")
    _update_policy_manifest(run_dir)
    if STAGE == "prefix" and phase == "complete" and nominal_events == 10_000:
        _create_prefix_handoff(
            run_dir=run_dir,
            runtime_checkpoint=runtime_checkpoint_path,
            checkpoint_extra=extra,
            actor_tensor_sha256=actor_digest,
        )


TRAINER_CLASS.save = save_with_fewshot_export


def _verify_complete_export() -> None:
    output_root = Path(_argument_value("--output-root")).resolve()
    run_dir = output_root / _argument_value("--run-name")
    manifest_path = run_dir / "v10_policies" / "policy_manifest.json"
    manifest = _json(manifest_path)
    expected_events = set(
        PROTOCOL["policy_budgets"][VARIANT]["export_online_events"]
    )
    entries = manifest.get("entries", [])
    actual_events = {entry["online_events_nominal"] for entry in entries}
    summary = _json(run_dir / "summary.json")
    zero_warmup_ok = True
    if STAGE == "prefix":
        zero = next(entry for entry in entries if entry["online_events_nominal"] == 0)
        warmup = next(
            entry
            for entry in entries
            if entry["online_events_nominal"] == RUN_JOB["warmup_events"]
        )
        zero_warmup_ok = (
            zero["actor_tensor_sha256"] == warmup["actor_tensor_sha256"]
        )
    branch_record_ok = True
    if STAGE == "branch":
        record = _json(run_dir / "v10_branch_materialization.json")
        handoff = _json(Path(RUN_JOB["branch_handoff_manifest"]))
        branch_record_ok = (
            record.get("run_signature") == RUN_SIGNATURE
            and record.get("variant") == VARIANT
            and record.get("checkpoint_state_sha256")
            == handoff.get("checkpoint_state_sha256")
        )
    if (
        manifest.get("schema_version") != POLICY_MANIFEST_SCHEMA
        or actual_events != expected_events
        or len(entries) != len(expected_events)
        or not zero_warmup_ok
        or not branch_record_ok
        or summary.get("completed_epochs") != RUN_JOB["epochs"]
        or summary.get("total_updates") != RUN_JOB["expected_total_updates"]
        or summary.get("simulator_decision_events") > RUN_JOB["online_event_cap"]
        or any(
            not (run_dir / entry["model_checkpoint"]["path"]).is_file()
            or sha256(run_dir / entry["model_checkpoint"]["path"])
            != entry["model_checkpoint"]["sha256"]
            or entry["learner_update_steps"]
            != next(
                value
                for value in PROTOCOL["policy_budgets"][VARIANT][
                    "milestones"
                ]
                if value["online_events"] == entry["online_events_nominal"]
            )["learner_update_steps"]
            for entry in entries
        )
    ):
        raise RuntimeError("completed run failed the frozen v10 export contract")
    _write_json(
        run_dir / "v10_export_summary.json",
        {
            "schema_version": EXPORT_SUMMARY_SCHEMA,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "method": RUN_JOB["method"],
            "variant": VARIANT,
            "stage": STAGE,
            "seed": RUN_JOB["seed"],
            "run_signature": RUN_SIGNATURE,
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "policy_manifest": {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
                "entries": len(entries),
            },
            "zero_and_warmup_actor_identical": (
                True if STAGE == "prefix" else None
            ),
            "shared_prefix_handoff": (
                _descriptor(run_dir / "v10_handoff/manifest.json")
                if STAGE == "prefix"
                else None
            ),
            "branch_materialization": (
                _descriptor(run_dir / "v10_branch_materialization.json")
                if STAGE == "branch"
                else None
            ),
            "summary_sha256": sha256(run_dir / "summary.json"),
            "formal_held_out_outcomes_read": False,
        },
    )


def main() -> None:
    branch_checkpoint = _materialize_branch()
    if branch_checkpoint is not None:
        if "--resume" in sys.argv:
            supplied = Path(_argument_value("--resume")).resolve()
            if supplied != branch_checkpoint.resolve():
                raise ValueError("scheduler resume path differs from v10 branch")
        else:
            sys.argv.extend(["--resume", str(branch_checkpoint)])
    base.main("v7")
    _verify_complete_export()


if __name__ == "__main__":
    main()
