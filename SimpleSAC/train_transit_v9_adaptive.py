#!/usr/bin/env python3
"""Run one registered v9 H2O+ development intervention.

The wrapper preserves the v7 SMDP, rollout and hard-event-cap implementation,
selects one isolated v9 trainer subclass, consumes a sealed v8 offline
initializer, and exports every preregistered few-shot milestone.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ENV = "H2OPLUS_V9_WORKSPACE_ROOT"
PROTOCOL_ENV = "H2OPLUS_V9_PROTOCOL"
EXPECTED_PROTOCOL_ENV = "H2OPLUS_V9_PROTOCOL_SHA256"
BASE_DEPLOYMENT_ENV = "H2OPLUS_V9_BASE_DEPLOYMENT_MANIFEST"
EXPECTED_BASE_DEPLOYMENT_ENV = (
    "H2OPLUS_V9_BASE_DEPLOYMENT_MANIFEST_SHA256"
)
ADMISSION_SCHEMA = "transit-training-admission-v9-adaptive-development-v2"
PROTOCOL_SCHEMA = "transit-h2oplus-v9-adaptive-protocol-v2"
POLICY_MANIFEST_SCHEMA = "transit-h2oplus-v9-adaptive-policy-manifest-v1"
EXPORT_SUMMARY_SCHEMA = "transit-h2oplus-v9-adaptive-export-summary-v1"


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
        raise ValueError(f"v9 command must explicitly provide {name}") from error
    if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
        raise ValueError(f"v9 command lacks the value for {name}")
    return sys.argv[index + 1]


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
        raise ValueError("v9 protocol hash drifted")
    protocol = _json(protocol_path)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("study_stage") != "development"
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("local_training_or_sumo_allowed") is not False
    ):
        raise ValueError("v9 protocol scope drifted")
    deployment_path = _required_path(BASE_DEPLOYMENT_ENV)
    expected_deployment = _required_sha(EXPECTED_BASE_DEPLOYMENT_ENV)
    if sha256(deployment_path) != expected_deployment:
        raise ValueError("frozen v7 deployment manifest hash drifted")
    bound = protocol.get("base_v7_training_deployment", {})
    if bound.get("sha256") != expected_deployment:
        raise ValueError("v9 protocol does not bind the base v7 deployment")
    return protocol_path, protocol


PROTOCOL_PATH, PROTOCOL = _load_protocol()
WORKSPACE = _required_path(WORKSPACE_ENV)
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import SimpleSAC.train_transit_v3 as base  # noqa: E402
import SimpleSAC.transit_v7.trainer as v7_trainer_module  # noqa: E402
from SimpleSAC.transit_v9.trainer import (  # noqa: E402
    V9_TRAINER_VARIANTS,
    authorize_registered_initializer_bridge,
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
    raise ValueError("run signature is not registered in the v9 protocol")
if _command_without_resume(sys.argv[1:]) != RUN_JOB.get("command_arguments"):
    raise ValueError("runtime command differs from the frozen v9 job")
VARIANT = str(RUN_JOB["variant"])
try:
    TRAINER_CLASS = V9_TRAINER_VARIANTS[VARIANT]
except KeyError as error:
    raise ValueError(f"unregistered v9 variant: {VARIANT}") from error


def _stable_object_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _v9_source_sha256() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        WORKSPACE / "SimpleSAC/transit_v9/__init__.py",
        WORKSPACE / "SimpleSAC/transit_v9/trainer.py",
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
        or payload.get("v9_source_sha256") != _v9_source_sha256()
        or payload.get("v9_protocol", {}).get("sha256")
        != sha256(PROTOCOL_PATH)
        or payload.get("v9_wrapper", {}).get("sha256") != sha256(wrapper)
        or payload.get("variant") != VARIANT
        or any(payload.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("v9 development admission is stale or misbound")
    if admission_path != Path(RUN_JOB["training_admission"]).resolve():
        raise ValueError("runtime admission path differs from the registered job")
    return {
        "path": str(admission_path),
        "sha256": sha256(admission_path),
        "bytes": admission_path.stat().st_size,
        "schema_version": payload["schema_version"],
        "study_stage": "development",
        "selected_arguments": payload.get("selected_arguments"),
        "v9_protocol_sha256": sha256(PROTOCOL_PATH),
        "v9_source_sha256": _v9_source_sha256(),
        "variant": VARIANT,
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
        raise ValueError("v9 offline initializer is stale or misbound")
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
        "v9_reuse_authorized": True,
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
    policy_root = run_dir / "v9_policies"
    entries = []
    for actor in sorted(policy_root.glob("online_events_*.pt")):
        digest, extra = _existing_actor_digest(actor)
        checkpoint_descriptor = extra.get("v9_model_checkpoint", {})
        checkpoint = run_dir / str(checkpoint_descriptor.get("path", ""))
        if (
            not checkpoint.is_file()
            or sha256(checkpoint) != checkpoint_descriptor.get("sha256")
        ):
            raise RuntimeError(f"v9 source checkpoint drifted: {checkpoint}")
        entries.append(
            {
                "path": actor.relative_to(run_dir).as_posix(),
                "sha256": sha256(actor),
                "bytes": actor.stat().st_size,
                "actor_tensor_sha256": digest,
                "online_events_nominal": extra["v9_online_events_nominal"],
                "online_events_actual": extra["v9_online_events_actual"],
                "event_budget_deviation": extra["v9_event_budget_deviation"],
                "learner_update_steps": extra["v9_learner_update_steps"],
                "ratio_optimizer_steps": extra["v9_ratio_optimizer_steps"],
                "training_checkpoint_sha256": extra["training_checkpoint_sha256"],
                "model_checkpoint": checkpoint_descriptor,
                "phase": extra["phase"],
                "epoch": extra["epoch"],
            }
        )
    _write_json(
        policy_root / "policy_manifest.json",
        {
            "schema_version": POLICY_MANIFEST_SCHEMA,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "method": RUN_JOB["method"],
            "variant": VARIANT,
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
        / "v9_checkpoints"
        / f"online_events_{nominal_events:09d}.pt"
    )
    compact_extra = dict(extra)
    compact_extra.update(
        {
            "runtime_state": None,
            "v9_checkpoint_scope": (
                "model, optimizers, RNG and counters; online replay and simulator "
                "runtime state intentionally excluded; not a resume checkpoint"
            ),
            "v9_online_events_nominal": nominal_events,
            "v9_online_events_actual": event_evidence["actual"],
            "v9_learner_update_steps": learner_steps,
            "v9_ratio_optimizer_steps": _ratio_optimizer_steps(nominal_events),
            "v9_intervention": self.v9_intervention,
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
                "v9_online_events_nominal"
            )
            != nominal_events
            or int(compact_payload.get("total_updates", -1)) != learner_steps
        ):
            raise RuntimeError(
                f"existing v9 model checkpoint drifted: {compact_checkpoint}"
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
        "scope": compact_extra["v9_checkpoint_scope"],
    }
    destination = (
        run_dir
        / "v9_policies"
        / f"online_events_{nominal_events:09d}.pt"
    )
    policy_extra = {
        "epoch": epoch,
        "phase": phase,
        "run_signature": extra["run_signature"],
        "method": extra["method"],
        "variant": VARIANT,
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
        "v9_model_checkpoint": compact_checkpoint_descriptor,
        "v9_runtime_checkpoint_sha256_at_export": sha256(
            runtime_checkpoint_path
        ),
        "training_admission": metadata["training_admission"],
        "offline_initializer_consumed": extra["offline_initializer_consumed"],
        "v9_online_events_nominal": nominal_events,
        "v9_online_events_actual": event_evidence["actual"],
        "v9_event_budget_deviation": event_evidence["deviation"],
        "v9_event_cap_evidence": event_evidence,
        "v9_learner_update_steps": learner_steps,
        "v9_ratio_optimizer_steps": _ratio_optimizer_steps(nominal_events),
        "v9_actor_tensor_sha256": actor_digest,
        "v9_intervention": self.v9_intervention,
        "v9_source_sha256": _v9_source_sha256(),
        "v9_protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": sha256(PROTOCOL_PATH),
        },
        "v9_wrapper": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "selection_rule": (
            "all preregistered v9 online-event milestones; no evaluation outcome "
            "selects a checkpoint"
        ),
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    if learner_steps != int(self.total_updates):
        raise RuntimeError("v9 learner-step accounting disagrees with trainer")
    if destination.exists():
        existing_digest, existing_extra = _existing_actor_digest(destination)
        if (
            existing_digest != actor_digest
            or existing_extra.get("v9_online_events_nominal") != nominal_events
            or existing_extra.get("run_signature") != extra["run_signature"]
        ):
            raise RuntimeError(f"existing v9 milestone actor drifted: {destination}")
    else:
        self.save_policy_artifact(destination, extra=policy_extra)
    if nominal_events == int(metadata["arguments"]["warmup_events"]):
        zero = run_dir / "v9_policies" / "online_events_000000000.pt"
        zero_digest, _ = _existing_actor_digest(zero)
        if zero_digest != actor_digest:
            raise RuntimeError("v9 0-event and post-warmup actors differ")
    _update_policy_manifest(run_dir)


TRAINER_CLASS.save = save_with_fewshot_export


def _verify_complete_export() -> None:
    output_root = Path(_argument_value("--output-root")).resolve()
    run_dir = output_root / _argument_value("--run-name")
    manifest_path = run_dir / "v9_policies" / "policy_manifest.json"
    manifest = _json(manifest_path)
    expected_events = set(
        PROTOCOL["policy_budgets"][VARIANT]["export_online_events"]
    )
    entries = manifest.get("entries", [])
    actual_events = {entry["online_events_nominal"] for entry in entries}
    zero = next(entry for entry in entries if entry["online_events_nominal"] == 0)
    warmup = next(
        entry
        for entry in entries
        if entry["online_events_nominal"] == RUN_JOB["warmup_events"]
    )
    summary = _json(run_dir / "summary.json")
    if (
        manifest.get("schema_version") != POLICY_MANIFEST_SCHEMA
        or actual_events != expected_events
        or len(entries) != len(expected_events)
        or zero["actor_tensor_sha256"] != warmup["actor_tensor_sha256"]
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
        raise RuntimeError("completed run failed the frozen v9 export contract")
    _write_json(
        run_dir / "v9_export_summary.json",
        {
            "schema_version": EXPORT_SUMMARY_SCHEMA,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "method": RUN_JOB["method"],
            "variant": VARIANT,
            "seed": RUN_JOB["seed"],
            "run_signature": RUN_SIGNATURE,
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "policy_manifest": {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
                "entries": len(entries),
            },
            "zero_and_warmup_actor_identical": True,
            "summary_sha256": sha256(run_dir / "summary.json"),
            "formal_held_out_outcomes_read": False,
        },
    )


def main() -> None:
    base.main("v7")
    _verify_complete_export()


if __name__ == "__main__":
    main()
