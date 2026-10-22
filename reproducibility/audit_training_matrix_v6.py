#!/usr/bin/env python3
"""Independently audit a completed v6 training matrix and its checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.run_training_matrix_v4 import _is_complete, load_plan
from SimpleSAC.train_transit_v3 import sha256


AUDIT_SCHEMA = "transit-training-matrix-independent-audit-v1"


def _json_is_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_json_is_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_json_is_finite(item) for item in value)
    return False


def _checkpoint_models_are_finite(checkpoint: dict[str, Any]) -> bool:
    import torch

    for module_name in ("actor", "critic", "target_critic", "value", "ratio"):
        state = checkpoint.get(module_name)
        if state is None:
            continue
        for tensor in state.values():
            if torch.is_tensor(tensor) and not bool(torch.isfinite(tensor).all()):
                return False
    log_alpha = checkpoint.get("log_alpha")
    return not torch.is_tensor(log_alpha) or bool(torch.isfinite(log_alpha).all())


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _job_at_matrix_root(
    job: dict[str, Any],
    matrix_root: Path,
    initializer_root: Path | None = None,
) -> dict[str, Any]:
    """Bind a signed job identity to synchronized run and initializer roots."""

    candidate = {**job, "run_dir": matrix_root / job["name"]}
    if job.get("offline_initializer") is None or initializer_root is None:
        return candidate
    dependency = job.get("offline_initializer_dependency", {})
    expected_source = f"h2oplus_seed{job['seed']}"
    if dependency.get("source_job_name") != expected_source:
        raise ValueError(f"unexpected initializer source for {job['name']}")
    initializer = (
        initializer_root
        / expected_source
        / "checkpoints"
        / "offline_initializer.pt"
    )
    if not initializer.is_file():
        raise FileNotFoundError(
            f"canonical initializer is missing for {job['name']}: {initializer}"
        )
    candidate["offline_initializer"] = initializer
    return candidate


def audit(
    plan_path: Path,
    matrix_root: Path,
    initializer_root: Path | None = None,
) -> dict[str, Any]:
    import torch

    plan_path = plan_path.expanduser().resolve()
    matrix_root = matrix_root.expanduser().resolve()
    if initializer_root is not None:
        initializer_root = initializer_root.expanduser().resolve()
    jobs, current_evidence = load_plan(plan_path)
    plan = json.loads(plan_path.read_text())
    manifest_path = matrix_root / "matrix_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    warnings: list[str] = []
    job_reports: dict[str, dict[str, Any]] = {}

    expected_manifest_fields = {
        "plan_sha256": current_evidence["plan_sha256"],
        "dataset_sha256": current_evidence["dataset_sha256"],
        "source_sha256": current_evidence["source_sha256"],
        "runner_sha256": current_evidence["runner_sha256"],
        "trainer_entrypoint_sha256": current_evidence[
            "trainer_entrypoint_sha256"
        ],
        "simulator": current_evidence["simulator"],
    }
    for key, expected in expected_manifest_fields.items():
        if manifest.get(key) != expected:
            errors.append(f"matrix manifest {key} does not match current locked input")
    if Path(manifest.get("output_root", "")).resolve() != matrix_root:
        errors.append("matrix manifest output_root does not match the audited directory")
    if manifest.get("matrix_schema", manifest.get("schema_version")) != current_evidence[
        "matrix_schema"
    ] and manifest.get("schema_version") != current_evidence["matrix_schema"]:
        errors.append("matrix manifest schema is not the plan's v6 matrix schema")

    manifest_jobs = {job["name"]: job for job in manifest.get("jobs", [])}
    manifest_results = manifest.get("results", {})
    expected_names = {job["name"] for job in jobs}
    if set(manifest_jobs) != expected_names:
        errors.append("matrix manifest job set differs from the signed plan")
    if set(manifest_results) != expected_names:
        errors.append("matrix result set differs from the signed plan")

    registered_methods = plan.get("registration", {}).get("methods", {})
    expected_checkpoint_version = plan.get("registration", {}).get(
        "checkpoint_version"
    )
    if not isinstance(expected_checkpoint_version, str):
        errors.append("training registration lacks a checkpoint-version contract")
    for job in jobs:
        name = job["name"]
        run_dir = matrix_root / name
        # A synchronized matrix is intentionally relocatable.  Completeness must
        # be checked against the audited tree, not the plan's original host path.
        job = _job_at_matrix_root(job, matrix_root, initializer_root)
        prefix = f"{name}: "
        result = manifest_results.get(name, {})
        if result.get("status") not in {"completed", "skipped"}:
            errors.append(prefix + f"matrix status is {result.get('status')!r}")
            continue
        if not _is_complete(job, current_evidence):
            errors.append(prefix + "signed run completeness check failed")
            continue

        summary = json.loads((run_dir / "summary.json").read_text())
        metadata = json.loads((run_dir / "metadata.json").read_text())
        phase = json.loads((run_dir / "phase_metrics.json").read_text())
        metrics = _read_metrics(run_dir / "metrics.jsonl")
        checkpoint_path = run_dir / "checkpoints" / "latest.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        profile = metadata["method_profile"]
        arguments = metadata["arguments"]

        for label, payload in (
            ("summary", summary),
            ("metadata", metadata),
            ("phase metrics", phase),
            ("epoch metrics", metrics),
        ):
            if not _json_is_finite(payload):
                errors.append(prefix + f"{label} contains a non-finite value")
        if not _checkpoint_models_are_finite(checkpoint):
            errors.append(prefix + "checkpoint model parameters are non-finite")
        if checkpoint.get("checkpoint_version") != expected_checkpoint_version:
            errors.append(prefix + "checkpoint version is not the formal v6 contract")
        if checkpoint.get("total_updates") != summary.get("total_updates"):
            errors.append(prefix + "checkpoint and summary update counters disagree")
        if checkpoint.get("extra", {}).get("run_signature") != job["signature"]:
            errors.append(prefix + "checkpoint run signature disagrees with the plan")
        if checkpoint.get("extra", {}).get("phase") != "complete":
            errors.append(prefix + "latest checkpoint is not marked complete")
        if checkpoint.get("online_replay", {}).get("size") != summary.get(
            "online_buffer_size"
        ):
            errors.append(prefix + "checkpoint replay size disagrees with summary")

        registered = registered_methods.get(job["method"], {})
        if registered and registered.get("profile") != profile:
            errors.append(prefix + "metadata profile differs from preregistration")
        if summary.get("method_profile") != profile:
            errors.append(prefix + "summary and metadata method profiles disagree")
        if metadata.get("run_signature") != job["signature"]:
            errors.append(prefix + "metadata run signature disagrees with the plan")
        if summary.get("checkpoint", {}).get("sha256") != sha256(checkpoint_path):
            errors.append(prefix + "summary checkpoint SHA-256 is stale")

        uses_offline = profile["offline_objective"] != "none"
        uses_online = profile["online_update_mode"] != "none"
        expected_offline_updates = arguments["offline_updates"] if uses_offline else 0
        expected_online_updates = (
            arguments["epochs"] * arguments["updates_per_epoch"] if uses_online else 0
        )
        expected_total_updates = expected_offline_updates + expected_online_updates
        if summary.get("total_updates") != expected_total_updates:
            errors.append(
                prefix
                + f"total updates {summary.get('total_updates')} != {expected_total_updates}"
            )
        if checkpoint.get("online_critic_updates") != expected_online_updates:
            errors.append(prefix + "online critic update counter is incorrect")
        if phase.get("offline_updates_completed", 0) != expected_offline_updates:
            errors.append(prefix + "offline update counter is incorrect")

        expected_epochs = arguments["epochs"] if uses_online else 0
        if len(metrics) != expected_epochs:
            errors.append(prefix + f"found {len(metrics)} epoch rows, expected {expected_epochs}")
        if [row.get("epoch") for row in metrics] != list(
            range(1, expected_epochs + 1)
        ):
            errors.append(prefix + "epoch metric indices are incomplete or unordered")
        warmup_events = int(summary.get("warmup_metrics", {}).get("events", 0))
        rollout_transitions = sum(
            int(row.get("rollout", {}).get("transitions", 0)) for row in metrics
        )
        warmup_transitions = int(
            summary.get("warmup_metrics", {}).get("transitions", 0)
        )
        simulator_decision_event_cap = (
            arguments["warmup_events"]
            + arguments["epochs"] * arguments["events_per_epoch"]
            if uses_online
            else 0
        )
        actual_events = warmup_events
        maximum_batch = int(
            summary.get("warmup_metrics", {}).get(
                "maximum_simultaneous_decisions", 0
            )
        )
        for epoch_index, row in enumerate(metrics, start=1):
            rollout = row.get("rollout", {})
            cumulative_cap = (
                arguments["warmup_events"]
                + epoch_index * arguments["events_per_epoch"]
            )
            expected_request = cumulative_cap - actual_events
            if row.get("requested_simulator_decision_events") != expected_request:
                errors.append(prefix + "epoch event request did not carry overshoot")
            actual_events += int(rollout.get("events", 0))
            maximum_batch = max(
                maximum_batch,
                int(rollout.get("maximum_simultaneous_decisions", 0)),
            )
            if row.get("cumulative_simulator_decision_event_cap") != cumulative_cap:
                errors.append(prefix + "epoch cumulative event cap is incorrect")
            if row.get("actual_cumulative_simulator_decision_events") != actual_events:
                errors.append(prefix + "epoch cumulative event count is incorrect")
            if epoch_index < expected_epochs and actual_events < cumulative_cap:
                errors.append(prefix + "an intermediate epoch underfilled its event target")
            overshoot = actual_events - cumulative_cap
            if overshoot >= maximum_batch and overshoot > 0:
                errors.append(prefix + "atomic-batch event overshoot is not bounded")

        event_underfill = simulator_decision_event_cap - actual_events
        final_rollout = (
            metrics[-1].get("rollout", {})
            if metrics
            else summary.get("warmup_metrics", {})
        )
        final_blocked_batch = int(
            final_rollout.get("blocked_simultaneous_batch_size", 0)
        )
        if event_underfill < 0:
            errors.append(prefix + "simulator decision-event hard cap was exceeded")
        elif event_underfill:
            if (
                not final_rollout.get("hard_event_cap_reached", False)
                or final_blocked_batch <= event_underfill
            ):
                errors.append(
                    prefix
                    + "event-cap underfill lacks a larger blocked simultaneous batch"
                )
        elif final_blocked_batch:
            errors.append(prefix + "exact event-cap closure recorded a blocked batch")
        expected_summary_budget = {
            "simulator_decision_event_cap": simulator_decision_event_cap,
            "simulator_decision_events": actual_events,
            "simulator_decision_event_underfill": event_underfill,
            "maximum_simultaneous_decisions": maximum_batch,
            "final_blocked_simultaneous_batch_size": final_blocked_batch,
        }
        for key, expected in expected_summary_budget.items():
            if summary.get(key) != expected:
                errors.append(prefix + f"summary {key} is incorrect")
        observed_transitions = warmup_transitions + rollout_transitions
        if observed_transitions != summary.get("online_buffer_size"):
            errors.append(prefix + "recorded transitions disagree with online replay size")
        if registered:
            registered_events = registered.get("budget", {}).get(
                "total_simulator_events"
            )
            registered_cap = registered.get("budget", {}).get(
                "simulator_decision_event_cap"
            )
            if (
                registered_events != simulator_decision_event_cap
                or registered_cap != simulator_decision_event_cap
            ):
                errors.append(prefix + "event cap differs from preregistration")

        runtime = checkpoint.get("extra", {}).get("runtime_state")
        if uses_online:
            if not runtime:
                errors.append(prefix + "online checkpoint lacks simulator runtime state")
            else:
                runtime_path = run_dir / runtime["path"]
                if not runtime_path.is_file() or sha256(runtime_path) != runtime.get(
                    "sha256"
                ):
                    errors.append(prefix + "simulator runtime snapshot hash is invalid")
        elif runtime is not None:
            warnings.append(prefix + "offline-only checkpoint unexpectedly stores runtime state")

        compile_expected = bool(arguments.get("compile_networks", False))
        compiled_modules = metadata.get("hardware", {}).get("compiled_modules", [])
        if compile_expected != bool(compiled_modules):
            errors.append(prefix + "compiled-module provenance disagrees with the plan")
        job_reports[name] = {
            "method": job["method"],
            "seed": job["seed"],
            "run_signature": job["signature"],
            "checkpoint_sha256": sha256(checkpoint_path),
            "offline_updates": expected_offline_updates,
            "online_critic_updates": expected_online_updates,
            "simulator_decision_event_cap": simulator_decision_event_cap,
            "simulator_decision_events": actual_events,
            "simulator_decision_event_underfill": event_underfill,
            "maximum_simultaneous_decisions": maximum_batch,
            "final_blocked_simultaneous_batch_size": final_blocked_batch,
            "stored_online_transitions": summary.get("online_buffer_size"),
            "timings_sec": summary.get("timings_sec", {}),
            "max_rss_mb": summary.get("max_rss_mb"),
            "max_cuda_memory_mb": summary.get("max_cuda_memory_mb"),
            "compiled_modules": compiled_modules,
        }
        del checkpoint

    initializer_evidence = None
    if initializer_root is not None:
        initializer_evidence = {
            "path": str(initializer_root),
            "files": {
                f"h2oplus_seed{seed}/checkpoints/offline_initializer.pt": sha256(
                    initializer_root
                    / f"h2oplus_seed{seed}"
                    / "checkpoints"
                    / "offline_initializer.pt"
                )
                for seed in sorted(
                    {
                        int(job["seed"])
                        for job in jobs
                        if job.get("offline_initializer") is not None
                    }
                )
            },
        }

    return {
        "schema_version": AUDIT_SCHEMA,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
        "matrix_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "matrix_root": str(matrix_root),
        "initializer_root": initializer_evidence,
        "jobs_expected": len(jobs),
        "jobs_audited": len(job_reports),
        "jobs": job_reports,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--initializer-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit(args.plan, args.matrix_root, args.initializer_root)
    _write_json(args.output, report)
    print(args.output.expanduser().resolve())
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
