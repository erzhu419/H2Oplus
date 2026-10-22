#!/usr/bin/env python3
"""Audit v4 hurdle-policy checkpoints on a fixed canonical state sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.transit_v4.trainer import load_actor_checkpoint_v4


PERCENTILES = (0, 1, 5, 25, 50, 75, 95, 99, 100)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = np.percentile(values, PERCENTILES)
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "percentiles": {
            str(percentile): float(value)
            for percentile, value in zip(PERCENTILES, quantiles)
        },
    }


def _parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("checkpoint must use NAME=PATH syntax")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise ValueError("checkpoint must use non-empty NAME=PATH syntax")
    return name, Path(raw_path).expanduser().resolve()


def _state_sample(dataset: Path, rows: int) -> tuple[np.ndarray, dict]:
    with h5py.File(dataset, "r") as handle:
        total = int(handle["observations"].shape[0])
        selected_rows = min(int(rows), total)
        if selected_rows <= 0:
            raise ValueError("rows must be positive")
        indices = np.linspace(0, total - 1, selected_rows, dtype=np.int64)
        observations = handle["observations"][indices]
        provenance = {
            "schema_version": handle.attrs["schema_version"],
            "content_signature": handle.attrs.get("content_signature"),
            "total_rows": total,
            "sampled_rows": selected_rows,
            "sampling": "equally spaced row indices over immutable HDF5 order",
        }
    return np.asarray(observations, dtype=np.float32), provenance


@torch.inference_mode()
def audit_actor(
    checkpoint: Path,
    observations: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> dict:
    actor, extra = load_actor_checkpoint_v4(checkpoint, device=device)
    actor.eval()
    actions = []
    hold_probabilities = []
    positive_actions = []
    positive_log_stds = []
    for start in range(0, len(observations), batch_size):
        states = torch.as_tensor(
            observations[start : start + batch_size],
            dtype=torch.float32,
            device=device,
        )
        gate_logit, positive_mean, positive_log_std = actor.distribution_parameters(
            states
        )
        deterministic_actions, _ = actor(states, deterministic=True)
        actions.append(deterministic_actions[:, 0].cpu().numpy())
        hold_probabilities.append(torch.sigmoid(gate_logit).cpu().numpy())
        positive_actions.append(
            actor._positive_transform(positive_mean).cpu().numpy()
        )
        positive_log_stds.append(positive_log_std.cpu().numpy())

    action = np.concatenate(actions)
    hold_probability = np.concatenate(hold_probabilities)
    conditional_positive_action = np.concatenate(positive_actions)
    log_std = np.concatenate(positive_log_stds)
    arrays = (action, hold_probability, conditional_positive_action, log_std)
    errors = []
    if not all(np.isfinite(values).all() for values in arrays):
        errors.append("actor emitted non-finite policy parameters or actions")
    if np.any(action < -1.0) or np.any(action > 1.0):
        errors.append("deterministic action lies outside [-1, 1]")
    if np.any(hold_probability < 0.0) or np.any(hold_probability > 1.0):
        errors.append("hold probability lies outside [0, 1]")

    no_hold = action == -1.0
    deterministic_hold_seconds = 30.0 * (action + 1.0)
    conditional_hold_seconds = 30.0 * (conditional_positive_action + 1.0)
    disclosures = []
    no_hold_rate = float(np.mean(no_hold))
    if no_hold_rate >= 0.999:
        disclosures.append("deterministic policy is effectively all zero-hold")
    if no_hold_rate <= 0.001:
        disclosures.append("deterministic policy is effectively always-hold")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_extra": {
            key: extra.get(key)
            for key in ("method", "training_seed", "run_signature", "phase", "epoch")
        },
        "deterministic_no_hold_rate": no_hold_rate,
        "deterministic_hold_seconds": summary(deterministic_hold_seconds),
        "hold_probability": summary(hold_probability),
        "conditional_positive_hold_seconds": summary(conditional_hold_seconds),
        "positive_log_standard_deviation": summary(log_std),
        "exact_action_bounds": {
            "minimum": float(action.min()),
            "maximum": float(action.max()),
            "max_hold_rate": float(np.mean(action == 1.0)),
        },
        "errors": errors,
        "disclosures": disclosures,
        "passed": not errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    torch.set_num_threads(args.cpu_threads)
    dataset = args.dataset.expanduser().resolve()
    observations, sample = _state_sample(dataset, args.rows)
    checkpoints = [_parse_checkpoint(value) for value in args.checkpoint]
    names = [name for name, _ in checkpoints]
    if len(names) != len(set(names)):
        raise ValueError("checkpoint names must be unique")
    device = torch.device(args.device)
    policies = {
        name: audit_actor(
            path,
            observations,
            device=device,
            batch_size=args.batch_size,
        )
        for name, path in checkpoints
    }
    report = {
        "schema_version": "transit-policy-checkpoint-audit-v1",
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "state_sample": sample,
        "policies": policies,
        "passed": all(policy["passed"] for policy in policies.values()),
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    print(json.dumps({
        "output": str(output),
        "passed": report["passed"],
        "policies": {
            name: {
                "no_hold_rate": result["deterministic_no_hold_rate"],
                "mean_hold_seconds": result["deterministic_hold_seconds"]["mean"],
                "disclosures": result["disclosures"],
            }
            for name, result in policies.items()
        },
    }, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
