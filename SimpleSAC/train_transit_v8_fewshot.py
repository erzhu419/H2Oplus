#!/usr/bin/env python3
"""Run the frozen v7 trainer while exporting v8 few-shot policy milestones.

This wrapper does not change optimization, replay, rollout, or event-cap logic.
It only validates a development admission and snapshots the compact actor after
the v7 trainer atomically commits a registered checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


WORKSPACE_ENV = "H2OPLUS_V8_WORKSPACE_ROOT"
PROTOCOL_ENV = "H2OPLUS_V8_FEWSHOT_PROTOCOL"
EXPECTED_PROTOCOL_ENV = "H2OPLUS_V8_FEWSHOT_PROTOCOL_SHA256"
FORMAL_DEPLOYMENT_ENV = "H2OPLUS_V8_FORMAL_DEPLOYMENT_MANIFEST"
EXPECTED_FORMAL_DEPLOYMENT_ENV = (
    "H2OPLUS_V8_FORMAL_DEPLOYMENT_MANIFEST_SHA256"
)
ADMISSION_SCHEMA = "transit-training-admission-v8-fewshot-development-v1"
PROTOCOL_SCHEMA = "transit-h2oplus-v8-fewshot-protocol-v1"
POLICY_MANIFEST_SCHEMA = "transit-h2oplus-v8-fewshot-policy-manifest-v1"
EXPORT_SUMMARY_SCHEMA = "transit-h2oplus-v8-fewshot-export-summary-v1"


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
        raise ValueError(f"few-shot command must explicitly provide {name}") from error
    if index + 1 >= len(sys.argv) or sys.argv[index + 1].startswith("--"):
        raise ValueError(f"few-shot command lacks the value for {name}")
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
        raise ValueError("few-shot protocol hash drifted")
    protocol = _json(protocol_path)
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or protocol.get("study_stage") != "development"
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("local_training_or_sumo_allowed") is not False
    ):
        raise ValueError("few-shot protocol scope drifted")
    deployment_path = _required_path(FORMAL_DEPLOYMENT_ENV)
    expected_deployment = _required_sha(EXPECTED_FORMAL_DEPLOYMENT_ENV)
    if sha256(deployment_path) != expected_deployment:
        raise ValueError("frozen v7 deployment manifest hash drifted")
    bound = protocol.get("frozen_v7_training_deployment", {})
    if bound.get("sha256") != expected_deployment:
        raise ValueError("few-shot protocol does not bind the active v7 deployment")
    return protocol_path, protocol


PROTOCOL_PATH, PROTOCOL = _load_protocol()
WORKSPACE = _required_path(WORKSPACE_ENV)
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import SimpleSAC.train_transit_v3 as base  # noqa: E402
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7  # noqa: E402


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
    raise ValueError("run signature is not registered in the few-shot protocol")
if _command_without_resume(sys.argv[1:]) != RUN_JOB.get("command_arguments"):
    raise ValueError("runtime command differs from the frozen few-shot job")


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
        or payload.get("fewshot_protocol", {}).get("sha256")
        != sha256(PROTOCOL_PATH)
        or payload.get("fewshot_wrapper", {}).get("sha256") != sha256(wrapper)
        or any(payload.get(key) != value for key, value in expected.items())
    ):
        raise ValueError("few-shot development admission is stale or misbound")
    if admission_path != Path(RUN_JOB["training_admission"]).resolve():
        raise ValueError("runtime admission path differs from the registered job")
    return {
        "path": str(admission_path),
        "sha256": sha256(admission_path),
        "bytes": admission_path.stat().st_size,
        "schema_version": payload["schema_version"],
        "study_stage": "development",
        "selected_arguments": payload.get("selected_arguments"),
        "fewshot_protocol_sha256": sha256(PROTOCOL_PATH),
    }


base.validate_v7_training_admission_runtime = validate_development_admission


def _milestone(
    *, method: str, phase: str, epoch: int, metadata: dict[str, Any]
) -> tuple[int, int] | None:
    arguments = metadata["arguments"]
    warmup = int(arguments["warmup_events"])
    per_epoch = int(arguments["events_per_epoch"])
    target_epochs = int(arguments["epochs"])
    target_events = warmup + target_epochs * per_epoch
    if phase == "initialized" and method == "sac-online":
        return 0, 0
    if phase == "offline_initializer_frozen" and method == "h2oplus":
        return 0, int(arguments["offline_updates"])
    if phase == "online_warmup":
        return warmup, int(arguments["offline_updates"])
    if phase == "online_training":
        nominal = warmup + epoch * per_epoch
        if nominal == target_events:
            return None
        allowed = set(PROTOCOL["policy_budgets"][method]["export_online_events"])
        if nominal in allowed:
            return nominal, int(arguments["offline_updates"]) + epoch * int(
                arguments["updates_per_epoch"]
            )
    if phase == "complete":
        allowed = set(PROTOCOL["policy_budgets"][method]["export_online_events"])
        if target_events in allowed:
            return target_events, int(arguments["offline_updates"]) + target_epochs * int(
                arguments["updates_per_epoch"]
            )
    return None


def _actual_events(run_dir: Path, phase: str, nominal: int) -> int:
    if phase in {"initialized", "offline_initializer_frozen"}:
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


def _ratio_optimizer_steps(method: str, nominal_events: int) -> int:
    if method != "h2oplus" or nominal_events <= 5_000:
        return 0
    online_learner_steps = nominal_events - 5_000
    interval = int(_argument_value("--ratio-update-interval"))
    pretrain = int(_argument_value("--ratio-pretrain-updates"))
    return pretrain + (online_learner_steps + interval - 1) // interval


def _existing_actor_digest(path: Path) -> tuple[str, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    return _actor_digest(payload["actor"]), dict(payload.get("extra", {}))


def _update_policy_manifest(run_dir: Path) -> None:
    policy_root = run_dir / "fewshot_policies"
    entries = []
    for actor in sorted(policy_root.glob("online_events_*.pt")):
        digest, extra = _existing_actor_digest(actor)
        checkpoint_descriptor = extra.get("fewshot_model_checkpoint", {})
        checkpoint = run_dir / str(checkpoint_descriptor.get("path", ""))
        if (
            not checkpoint.is_file()
            or sha256(checkpoint) != checkpoint_descriptor.get("sha256")
        ):
            raise RuntimeError(f"few-shot source checkpoint drifted: {checkpoint}")
        entries.append(
            {
                "path": actor.relative_to(run_dir).as_posix(),
                "sha256": sha256(actor),
                "bytes": actor.stat().st_size,
                "actor_tensor_sha256": digest,
                "online_events_nominal": extra["fewshot_online_events_nominal"],
                "online_events_actual": extra["fewshot_online_events_actual"],
                "event_budget_deviation": extra["fewshot_event_budget_deviation"],
                "learner_update_steps": extra["fewshot_learner_update_steps"],
                "ratio_optimizer_steps": extra["fewshot_ratio_optimizer_steps"],
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
            "seed": RUN_JOB["seed"],
            "run_signature": RUN_SIGNATURE,
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "entries": entries,
        },
    )


ORIGINAL_SAVE = TransitH2OTrainerV7.save


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
    resolved = _milestone(
        method=str(extra["method"]), phase=phase, epoch=epoch, metadata=metadata
    )
    if resolved is None:
        return
    nominal_events, learner_steps = resolved
    event_evidence = _event_cap_evidence(run_dir, phase, nominal_events)
    runtime_checkpoint_path = Path(path).resolve(strict=True)
    actor_state = self.actor.state_dict()
    actor_digest = _actor_digest(actor_state)
    compact_checkpoint = (
        run_dir
        / "fewshot_checkpoints"
        / f"online_events_{nominal_events:09d}.pt"
    )
    compact_extra = dict(extra)
    compact_extra.update(
        {
            "runtime_state": None,
            "fewshot_checkpoint_scope": (
                "model, optimizers, RNG and counters; online replay and simulator "
                "runtime state intentionally excluded; not a resume checkpoint"
            ),
            "fewshot_online_events_nominal": nominal_events,
            "fewshot_online_events_actual": event_evidence["actual"],
            "fewshot_learner_update_steps": learner_steps,
            "fewshot_ratio_optimizer_steps": _ratio_optimizer_steps(
                str(extra["method"]), nominal_events
            ),
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
                "fewshot_online_events_nominal"
            )
            != nominal_events
            or int(compact_payload.get("total_updates", -1)) != learner_steps
        ):
            raise RuntimeError(
                f"existing few-shot model checkpoint drifted: {compact_checkpoint}"
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
        "scope": compact_extra["fewshot_checkpoint_scope"],
    }
    destination = (
        run_dir
        / "fewshot_policies"
        / f"online_events_{nominal_events:09d}.pt"
    )
    policy_extra = {
        "epoch": epoch,
        "phase": phase,
        "run_signature": extra["run_signature"],
        "method": extra["method"],
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
        "fewshot_model_checkpoint": compact_checkpoint_descriptor,
        "fewshot_runtime_checkpoint_sha256_at_export": sha256(
            runtime_checkpoint_path
        ),
        "training_admission": metadata["training_admission"],
        "fewshot_online_events_nominal": nominal_events,
        "fewshot_online_events_actual": event_evidence["actual"],
        "fewshot_event_budget_deviation": event_evidence["deviation"],
        "fewshot_event_cap_evidence": event_evidence,
        "fewshot_learner_update_steps": learner_steps,
        "fewshot_ratio_optimizer_steps": _ratio_optimizer_steps(
            str(extra["method"]), nominal_events
        ),
        "fewshot_actor_tensor_sha256": actor_digest,
        "fewshot_protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": sha256(PROTOCOL_PATH),
        },
        "fewshot_wrapper": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "selection_rule": (
            "all preregistered online-event milestones; no evaluation outcome "
            "selects a checkpoint"
        ),
        "study_stage": "development",
        "formal_held_out_outcomes_read": False,
    }
    if learner_steps != int(self.total_updates):
        raise RuntimeError("few-shot learner-step accounting disagrees with trainer")
    if destination.exists():
        existing_digest, existing_extra = _existing_actor_digest(destination)
        if (
            existing_digest != actor_digest
            or existing_extra.get("fewshot_online_events_nominal") != nominal_events
            or existing_extra.get("run_signature") != extra["run_signature"]
        ):
            raise RuntimeError(f"existing milestone actor drifted: {destination}")
    else:
        self.save_policy_artifact(destination, extra=policy_extra)
    if nominal_events == int(metadata["arguments"]["warmup_events"]):
        zero = run_dir / "fewshot_policies" / "online_events_000000000.pt"
        zero_digest, _ = _existing_actor_digest(zero)
        if zero_digest != actor_digest:
            raise RuntimeError("0-event and post-warmup actors differ")
    _update_policy_manifest(run_dir)


TransitH2OTrainerV7.save = save_with_fewshot_export


def _verify_complete_export() -> None:
    output_root = Path(_argument_value("--output-root")).resolve()
    run_dir = output_root / _argument_value("--run-name")
    manifest_path = run_dir / "fewshot_policies" / "policy_manifest.json"
    manifest = _json(manifest_path)
    expected_events = set(
        PROTOCOL["policy_budgets"][RUN_JOB["method"]]["export_online_events"]
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
                for value in PROTOCOL["policy_budgets"][RUN_JOB["method"]][
                    "milestones"
                ]
                if value["online_events"] == entry["online_events_nominal"]
            )["learner_update_steps"]
            for entry in entries
        )
    ):
        raise RuntimeError("completed run failed the frozen few-shot export contract")
    _write_json(
        run_dir / "fewshot_export_summary.json",
        {
            "schema_version": EXPORT_SUMMARY_SCHEMA,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "passed": True,
            "method": RUN_JOB["method"],
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
