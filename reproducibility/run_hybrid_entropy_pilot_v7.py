#!/usr/bin/env python3
"""Run a paired numerical calibration of the v7 hybrid entropy target."""

from __future__ import annotations

import argparse
import atexit
import gc
import json
import math
import os
import resource
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import SCHEMA_VERSION as DATASET_SCHEMA
from bus_h2o.sumo_provenance import sha256
from reproducibility.benchmark_training_kernel_v4 import trainer_config_for_method
from reproducibility.audit_transit_v7_dataset import (
    AUDIT_SCHEMA as DATASET_AUDIT_SCHEMA,
    dataset_audit_source_hashes,
)
from reproducibility.run_reward_loss_pilot_v7 import (
    CRITIC_LOSSES,
    SCHEMA_VERSION as REWARD_LOSS_PILOT_SCHEMA,
    _finite_metrics,
    _parameter_summary,
    reward_loss_source_hashes,
)
from SimpleSAC.train_transit_v3 import (
    GPULease,
    collect_runtime_fingerprint,
    validate_v7_hybrid_target_entropy,
)
from SimpleSAC.transit_v7.replay_buffer import TransitReplayBufferV7
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7


SCHEMA_VERSION = "transit-v7-hybrid-entropy-numerical-pilot-v1"
TARGET_ENTROPY_CANDIDATES = (-0.5, 0.0, 0.5)
PILOT_SEEDS = (9401, 9402, 9403)
MINIMUM_OFFLINE_UPDATES = 2_000
MINIMUM_ONLINE_UPDATES = 2_000
DEFAULT_OFFLINE_UPDATES = MINIMUM_OFFLINE_UPDATES
DEFAULT_ONLINE_UPDATES = MINIMUM_ONLINE_UPDATES
MINIMUM_ONLINE_ROWS = 4_096
DEFAULT_ONLINE_ROWS = MINIMUM_ONLINE_ROWS
NUMERICAL_LIMIT = 1e6
ALPHA_RANGE = (1e-4, 100.0)
HOLD_PROBABILITY_RANGE = (0.01, 0.99)


def entropy_pilot_source_hashes() -> dict[str, str]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "reproducibility" / "run_reward_loss_pilot_v7.py",
        ROOT / "reproducibility" / "benchmark_training_kernel_v4.py",
        ROOT / "SimpleSAC" / "train_transit_v3.py",
        ROOT / "SimpleSAC" / "transit_v3" / "trainer.py",
        ROOT / "SimpleSAC" / "transit_v7" / "networks.py",
        ROOT / "SimpleSAC" / "transit_v7" / "replay_buffer.py",
        ROOT / "SimpleSAC" / "transit_v7" / "trainer.py",
        ROOT / "bus_h2o" / "mdp_contract_v7.py",
    )
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sources
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-audit", type=Path, required=True)
    parser.add_argument("--reward-loss-pilot", type=Path, required=True)
    parser.add_argument("--reward-scale", type=float, required=True)
    parser.add_argument("--critic-loss", choices=CRITIC_LOSSES, required=True)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--offline-updates", type=int, default=DEFAULT_OFFLINE_UPDATES)
    parser.add_argument("--online-updates", type=int, default=DEFAULT_ONLINE_UPDATES)
    parser.add_argument("--online-rows", type=int, default=DEFAULT_ONLINE_ROWS)
    parser.add_argument("--max-offline-rows", type=int, default=100_000)
    parser.add_argument("--diagnostic-rows", type=int, default=2_048)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--device-cache", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument("--allow-external-gpu-processes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def entropy_target_gates(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Admit a target only when all registered seeds remain nondegenerate."""

    gates: dict[str, dict[str, Any]] = {}
    expected_seeds = set(PILOT_SEEDS)
    for target in TARGET_ENTROPY_CANDIDATES:
        selected = [run for run in runs if float(run["target_entropy"]) == target]
        seeds = [int(run["seed"]) for run in selected]
        passed = sorted(
            int(run["seed"])
            for run in selected
            if run.get("numerical_gate", {}).get("eligible") is True
        )
        complete = (
            len(selected) == len(PILOT_SEEDS)
            and len(seeds) == len(set(seeds))
            and set(seeds) == expected_seeds
        )
        gates[f"{target:g}"] = {
            "target_entropy": target,
            "expected_seeds": list(PILOT_SEEDS),
            "observed_seeds": sorted(seeds),
            "passed_seeds": passed,
            "complete": complete,
            "eligible": complete and set(passed) == expected_seeds,
        }
    preference = sorted(TARGET_ENTROPY_CANDIDATES, key=lambda value: (abs(value), value))
    chosen = next(
        (target for target in preference if gates[f"{target:g}"]["eligible"]),
        None,
    )
    return {
        "preference_order": preference,
        "gates": gates,
        "selected_target_entropy": chosen,
        "passed": chosen is not None,
    }


def _load_inputs(
    dataset: Path,
    dataset_audit_path: Path,
    reward_loss_pilot_path: Path,
    *,
    reward_scale: float,
    critic_loss: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with h5py.File(dataset, "r") as handle:
        if handle.attrs.get("schema_version") != DATASET_SCHEMA:
            raise ValueError("entropy pilot requires transit-smdp-v7 data")
    dataset_digest = sha256(dataset)
    dataset_audit = json.loads(dataset_audit_path.read_text())
    if (
        dataset_audit.get("schema_version") != DATASET_AUDIT_SCHEMA
        or dataset_audit.get("passed") is not True
        or dataset_audit.get("dataset_sha256") != dataset_digest
        or dataset_audit.get("source_sha256") != dataset_audit_source_hashes()
    ):
        raise ValueError("entropy pilot dataset lacks a matching passed audit")
    reward_loss_pilot = json.loads(reward_loss_pilot_path.read_text())
    key = f"reward_scale={reward_scale:.12g}/critic_loss={critic_loss}"
    gate = reward_loss_pilot.get("configuration_gates", {}).get(key, {})
    selected = reward_loss_pilot.get("configuration_selection", {}).get(
        "selected", {}
    )
    if (
        reward_loss_pilot.get("schema_version") != REWARD_LOSS_PILOT_SCHEMA
        or reward_loss_pilot.get("dataset", {}).get("sha256") != dataset_digest
        or reward_loss_pilot.get("source_sha256") != reward_loss_source_hashes()
        or gate.get("eligible") is not True
        or float(selected.get("reward_scale", math.nan)) != reward_scale
        or selected.get("critic_loss") != critic_loss
    ):
        raise ValueError(
            "entropy pilot requires the selected all-seed-eligible reward/loss "
            "configuration"
        )
    return dataset_audit, reward_loss_pilot


def _append_synthetic_online_rows(replay: TransitReplayBufferV7, count: int) -> None:
    """Create deterministic compute-only online partitions from immutable rows."""

    if count <= 0 or count > replay.online_capacity:
        raise ValueError("synthetic online row count is outside replay capacity")
    source_indices = np.linspace(
        0, replay.offline_size - 1, num=count, dtype=np.int64
    )
    rows_per_trajectory = 128
    for index, source in enumerate(source_indices):
        trajectory = 10**12 + index // rows_per_trajectory
        replay.append(
            replay.observations[source],
            replay.actions[source],
            float(replay.rewards[source]),
            replay.next_observations[source],
            float(replay.discounts[source]),
            float(replay.terminals[source]),
            episode_id=trajectory,
            trajectory_id=trajectory,
            delta_t_seconds=float(replay.delta_t_seconds[source]),
        )


def _checkpoints(updates: int) -> set[int]:
    return {
        value
        for value in (1, 5, 10, 25, 50, 100, 250, updates)
        if 1 <= value <= updates
    }


def _run_candidate(
    replay: TransitReplayBufferV7,
    diagnostic_observations: torch.Tensor,
    *,
    reward_scale: float,
    critic_loss: str,
    huber_delta: float,
    target_entropy: float,
    seed: int,
    offline_updates: int,
    online_updates: int,
    device: torch.device,
) -> dict[str, Any]:
    config = trainer_config_for_method(
        "h2oplus",
        reward_scale=reward_scale,
        critic_loss=critic_loss,
        huber_delta=huber_delta,
        target_entropy=target_entropy,
    )
    trainer = TransitH2OTrainerV7(replay, config=config, device=device, seed=seed)
    initial = _parameter_summary(trainer)
    for _ in range(offline_updates):
        trainer.update(online=False, collect_metrics=False)
    online_start = _parameter_summary(trainer)
    reports = []
    all_metrics_finite = True
    checkpoints = _checkpoints(online_updates)
    started = time.perf_counter()
    for update in range(1, online_updates + 1):
        metrics = trainer.update(
            online=True, collect_metrics=update in checkpoints
        )
        if metrics:
            all_metrics_finite = all_metrics_finite and _finite_metrics(metrics)
            reports.append({"update": update, "metrics": metrics})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final = _parameter_summary(trainer)
    with torch.no_grad():
        components = trainer.actor.soft_action_components(
            diagnostic_observations, deterministic_positive=True
        )
    mean_hold_probability = float(components["hold_probability"].mean().cpu())
    mean_positive_action = float(components["positive_action"].mean().cpu())
    alpha = float(trainer.alpha.detach().cpu())
    reported_max_abs_td = max(
        (
            abs(float(report["metrics"].get("train/offline_max_abs_td", 0.0)))
            for report in reports
        ),
        default=0.0,
    )
    reported_max_grad = max(
        (
            abs(float(report["metrics"].get("train/critic_grad_norm", 0.0)))
            for report in reports
        ),
        default=0.0,
    )
    parameters_finite = all(row["all_finite"] for row in final.values())
    parameter_limit = max(row["maximum_absolute_value"] for row in final.values())
    eligible = bool(
        all_metrics_finite
        and parameters_finite
        and math.isfinite(alpha)
        and ALPHA_RANGE[0] <= alpha <= ALPHA_RANGE[1]
        and HOLD_PROBABILITY_RANGE[0]
        <= mean_hold_probability
        <= HOLD_PROBABILITY_RANGE[1]
        and abs(mean_positive_action) < 0.98
        and reported_max_abs_td <= NUMERICAL_LIMIT
        and reported_max_grad <= NUMERICAL_LIMIT
        and parameter_limit <= NUMERICAL_LIMIT
    )
    result = {
        "target_entropy": target_entropy,
        "seed": seed,
        "trainer_config": asdict(config),
        "offline_updates": offline_updates,
        "online_updates": online_updates,
        "online_seconds": elapsed,
        "online_updates_per_second": online_updates / elapsed,
        "initial_parameters": initial,
        "online_start_parameters": online_start,
        "final_parameters": final,
        "reports": reports,
        "final_policy": {
            "alpha": alpha,
            "mean_hold_probability": mean_hold_probability,
            "mean_deterministic_positive_action": mean_positive_action,
        },
        "numerical_gate": {
            "eligible": eligible,
            "all_reported_metrics_finite": all_metrics_finite,
            "all_parameters_finite": parameters_finite,
            "alpha_range": list(ALPHA_RANGE),
            "hold_probability_range": list(HOLD_PROBABILITY_RANGE),
            "maximum_reported_abs_td": reported_max_abs_td,
            "maximum_reported_critic_gradient_norm": reported_max_grad,
            "maximum_final_parameter_absolute_value": parameter_limit,
            "numerical_limit": NUMERICAL_LIMIT,
        },
    }
    del trainer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if (
        args.offline_updates < MINIMUM_OFFLINE_UPDATES
        or args.online_updates < MINIMUM_ONLINE_UPDATES
        or args.online_rows < MINIMUM_ONLINE_ROWS
        or args.max_offline_rows <= 1
        or args.diagnostic_rows <= 0
    ):
        raise ValueError("entropy-pilot sizes are invalid")
    if not math.isfinite(args.huber_delta) or args.huber_delta <= 0.0:
        raise ValueError("huber delta must be finite and positive")
    for target in TARGET_ENTROPY_CANDIDATES:
        validate_v7_hybrid_target_entropy(target)
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite entropy pilot {output}")
    dataset = args.dataset.expanduser().resolve()
    dataset_audit_path = args.dataset_audit.expanduser().resolve()
    reward_loss_pilot_path = args.reward_loss_pilot.expanduser().resolve()
    dataset_audit, reward_loss_pilot = _load_inputs(
        dataset,
        dataset_audit_path,
        reward_loss_pilot_path,
        reward_scale=args.reward_scale,
        critic_loss=args.critic_loss,
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA entropy pilot requested but unavailable")
    lease = GPULease(
        args.device,
        allow_external_processes=args.allow_external_gpu_processes,
        max_canonical_processes_per_gpu=1,
    )
    atexit.register(lease.close)
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_float32_matmul_precision("high")
    replay = TransitReplayBufferV7(
        dataset,
        split="train",
        device=device,
        online_capacity=args.online_rows,
        max_offline_rows=args.max_offline_rows,
        subsample_seed=9399,
        device_cache_observation_dtype=args.device_cache,
    )
    _append_synthetic_online_rows(replay, args.online_rows)
    cache = replay.materialize_device_cache(include_online=True)
    diagnostic_rows = min(args.diagnostic_rows, replay.offline_size)
    diagnostic_observations = replay.sample(
        diagnostic_rows,
        scope="offline",
        rng=np.random.default_rng(9398),
    )["observations"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    runs = [
        _run_candidate(
            replay,
            diagnostic_observations,
            reward_scale=args.reward_scale,
            critic_loss=args.critic_loss,
            huber_delta=args.huber_delta,
            target_entropy=target,
            seed=seed,
            offline_updates=args.offline_updates,
            online_updates=args.online_updates,
            device=device,
        )
        for target in TARGET_ENTROPY_CANDIDATES
        for seed in PILOT_SEEDS
    ]
    paired_initialization = {}
    paired_online_start = {}
    for seed in PILOT_SEEDS:
        selected = [run for run in runs if run["seed"] == seed]
        for module in ("actor", "critic", "target_critic", "value", "ratio"):
            paired_initialization[f"seed{seed}/{module}"] = len(
                {
                    run["initial_parameters"][module]["state_sha256"]
                    for run in selected
                }
            ) == 1
            paired_online_start[f"seed{seed}/{module}"] = len(
                {
                    run["online_start_parameters"][module]["state_sha256"]
                    for run in selected
                }
            ) == 1
    if not all(paired_initialization.values()) or not all(
        paired_online_start.values()
    ):
        raise RuntimeError("entropy candidates did not share paired training states")
    gates = entropy_target_gates(runs)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "numerical-pilot",
        "selection_boundary": (
            "uses immutable offline data and deterministic compute-only online rows; "
            "does not read reserved evaluation cells or select on policy return"
        ),
        "design": {
            "target_entropy_candidates": list(TARGET_ENTROPY_CANDIDATES),
            "seeds": list(PILOT_SEEDS),
            "offline_updates": args.offline_updates,
            "online_updates": args.online_updates,
            "minimum_offline_updates": MINIMUM_OFFLINE_UPDATES,
            "minimum_online_updates": MINIMUM_ONLINE_UPDATES,
            "online_rows": args.online_rows,
            "minimum_online_rows": MINIMUM_ONLINE_ROWS,
            "preference_rule": (
                "choose the all-seed-eligible candidate nearest zero; break an "
                "equal-distance tie toward the lower entropy"
            ),
        },
        "dataset": {
            "path": str(dataset),
            "sha256": sha256(dataset),
            "audit_path": str(dataset_audit_path),
            "audit_sha256": sha256(dataset_audit_path),
            "audit_schema_version": dataset_audit.get("schema_version"),
            "loaded_offline_rows": replay.offline_size,
        },
        "reward_loss_configuration": {
            "reward_scale": args.reward_scale,
            "critic_loss": args.critic_loss,
            "huber_delta": args.huber_delta,
            "pilot_path": str(reward_loss_pilot_path),
            "pilot_sha256": sha256(reward_loss_pilot_path),
            "pilot_schema_version": reward_loss_pilot.get("schema_version"),
        },
        "device_cache": cache,
        "runtime": collect_runtime_fingerprint(torch, str(device)),
        "resources": {
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
            "max_cuda_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        },
        "paired_initialization_checks": paired_initialization,
        "paired_online_start_checks": paired_online_start,
        "runs": runs,
        "target_entropy_gate": gates,
        "source_sha256": entropy_pilot_source_hashes(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    lease.close()
    print(
        json.dumps(
            {
                "output": str(output),
                "passed": gates["passed"],
                "selected_target_entropy": gates["selected_target_entropy"],
            },
            sort_keys=True,
        )
    )
    if not gates["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
