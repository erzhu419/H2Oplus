#!/usr/bin/env python3
"""Run paired H2O+ offline numerical pilots for registered reward/loss candidates."""

from __future__ import annotations

import argparse
import atexit
import gc
import hashlib
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
from reproducibility.audit_reward_conditioning_v7 import (
    AUDIT_SCHEMA as REWARD_AUDIT_SCHEMA,
    DEFAULT_REWARD_SCALES,
    reward_audit_source_hashes,
)
from reproducibility.audit_transit_v7_dataset import (
    AUDIT_SCHEMA as DATASET_AUDIT_SCHEMA,
    dataset_audit_source_hashes,
)
from reproducibility.benchmark_training_kernel_v4 import trainer_config_for_method
from SimpleSAC.train_transit_v3 import (
    GPULease,
    collect_runtime_fingerprint,
    validate_v7_hybrid_target_entropy,
)
from SimpleSAC.transit_v7.replay_buffer import TransitReplayBufferV7
from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7


SCHEMA_VERSION = "transit-v7-reward-loss-numerical-pilot-v1"
PILOT_SEEDS = (9301, 9302, 9303)
CRITIC_LOSSES = ("mse", "huber")
HUBER_DELTA = 1.0
MINIMUM_UPDATES = 2_000
DEFAULT_UPDATES = MINIMUM_UPDATES
NUMERICAL_LIMIT = 1e6
CONDITIONING_LIMITS = {
    "absolute_scaled_reward_q0.99": 25.0,
    "mse_absolute_scaled_reward_maximum": 100.0,
    "huber_absolute_scaled_reward_maximum": 1_000.0,
    "terminal_loss_fraction": 0.5,
}


def reward_loss_source_hashes() -> dict[str, str]:
    sources = (
        Path(__file__).resolve(),
        ROOT / "reproducibility" / "audit_reward_conditioning_v7.py",
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
    parser.add_argument("--reward-audit", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--max-offline-rows", type=int, default=100_000)
    parser.add_argument("--diagnostic-rows", type=int, default=2_048)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument(
        "--device-cache", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument("--target-entropy", type=float, default=0.0)
    parser.add_argument("--allow-external-gpu-processes", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _state_digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _parameter_summary(trainer: TransitH2OTrainerV7) -> dict[str, Any]:
    result = {}
    for name in ("actor", "critic", "target_critic", "value", "ratio"):
        module = getattr(trainer, name)
        values = torch.cat(
            [parameter.detach().reshape(-1) for parameter in module.parameters()]
        )
        result[name] = {
            "parameters": int(values.numel()),
            "all_finite": bool(torch.isfinite(values).all().item()),
            "maximum_absolute_value": float(values.abs().max().cpu()),
            "root_mean_square": float(values.square().mean().sqrt().cpu()),
            "state_sha256": _state_digest(module),
        }
    return result


def _checkpoints(updates: int) -> set[int]:
    candidates = {1, 5, 10, 25, 50, 100, 250, 500, updates}
    return {value for value in candidates if 1 <= value <= updates}


def _finite_metrics(metrics: dict[str, float]) -> bool:
    return all(math.isfinite(float(value)) for value in metrics.values())


def configuration_gates(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Require every preregistered seed to pass for a configuration."""

    result: dict[str, dict[str, Any]] = {}
    for reward_scale in DEFAULT_REWARD_SCALES:
        for critic_loss in CRITIC_LOSSES:
            selected = [
                run
                for run in runs
                if float(run["reward_scale"]) == reward_scale
                and run["critic_loss"] == critic_loss
            ]
            seeds = [int(run["seed"]) for run in selected]
            seed_set = set(seeds)
            expected_seeds = set(PILOT_SEEDS)
            complete = (
                len(selected) == len(PILOT_SEEDS)
                and len(seeds) == len(seed_set)
                and seed_set == expected_seeds
            )
            passed_seeds = sorted(
                int(run["seed"])
                for run in selected
                if run.get("numerical_gate", {}).get("eligible") is True
            )
            key = f"reward_scale={reward_scale:.12g}/critic_loss={critic_loss}"
            result[key] = {
                "reward_scale": reward_scale,
                "critic_loss": critic_loss,
                "expected_seeds": list(PILOT_SEEDS),
                "observed_seeds": sorted(seeds),
                "passed_seeds": passed_seeds,
                "complete": complete,
                "eligible": complete and set(passed_seeds) == expected_seeds,
            }
    return result


def select_reward_loss_configuration(
    reward_audit: dict[str, Any], gates: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Select a stable loss/scale pair from registered conditioning bounds."""

    configuration_rows = []
    candidates = reward_audit.get("reward_scale_candidates", {})
    for reward_scale in DEFAULT_REWARD_SCALES:
        scale_key = f"{reward_scale:.12g}"
        candidate = candidates.get(scale_key, {})
        absolute = candidate.get("absolute_scaled_reward", {})
        quantiles = absolute.get("quantiles", {})
        q95 = float(quantiles.get("q0.95", math.nan))
        q99 = float(quantiles.get("q0.99", math.nan))
        maximum = float(absolute.get("maximum", math.nan))
        proxy = candidate.get("reward_only_initial_td_proxy", {})
        for critic_loss in CRITIC_LOSSES:
            numerical_gate = gates.get(
                f"reward_scale={reward_scale:.12g}/critic_loss={critic_loss}", {}
            )
            loss_proxy = (
                proxy.get("mse", {})
                if critic_loss == "mse"
                else proxy.get("huber", {}).get(f"delta_{HUBER_DELTA:g}", {})
            )
            terminal_fraction = float(
                loss_proxy.get("terminal_fraction", math.nan)
            )
            maximum_limit = CONDITIONING_LIMITS[
                f"{critic_loss}_absolute_scaled_reward_maximum"
            ]
            conditioning_passed = bool(
                math.isfinite(q95)
                and q95 > 0.0
                and math.isfinite(q99)
                and q99 <= CONDITIONING_LIMITS["absolute_scaled_reward_q0.99"]
                and math.isfinite(maximum)
                and maximum <= maximum_limit
                and math.isfinite(terminal_fraction)
                and terminal_fraction
                <= CONDITIONING_LIMITS["terminal_loss_fraction"]
            )
            configuration_rows.append(
                {
                    "reward_scale": reward_scale,
                    "critic_loss": critic_loss,
                    "huber_delta": HUBER_DELTA,
                    "absolute_scaled_reward_q0.95": q95,
                    "absolute_scaled_reward_q0.99": q99,
                    "absolute_scaled_reward_maximum": maximum,
                    "absolute_scaled_reward_maximum_limit": maximum_limit,
                    "terminal_loss_fraction": terminal_fraction,
                    "numerical_gate_passed": numerical_gate.get("eligible") is True,
                    "conditioning_passed": conditioning_passed,
                    "log_distance_from_unit_q0.95": (
                        abs(math.log10(q95)) if q95 > 0.0 else math.inf
                    ),
                }
            )
    admissible = [
        row
        for row in configuration_rows
        if row["conditioning_passed"] and row["numerical_gate_passed"]
    ]
    if not admissible:
        return {
            "passed": False,
            "selected": None,
            "reason": "no all-seed-stable configuration satisfies conditioning limits",
            "conditioning_limits": CONDITIONING_LIMITS,
            "configurations": configuration_rows,
        }
    selected = min(
        admissible,
        key=lambda row: (
            row["log_distance_from_unit_q0.95"],
            0 if row["critic_loss"] == "mse" else 1,
            row["reward_scale"],
        ),
    )
    return {
        "passed": True,
        "selected": {
            "reward_scale": selected["reward_scale"],
            "critic_loss": selected["critic_loss"],
            "huber_delta": HUBER_DELTA,
        },
        "reason": (
            "closest admissible q0.95 to one after loss-specific tail and "
            "terminal-concentration checks; MSE wins an exact conditioning tie"
        ),
        "conditioning_limits": CONDITIONING_LIMITS,
        "configurations": configuration_rows,
    }


def _load_gate_inputs(
    dataset: Path, dataset_audit_path: Path, reward_audit_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    with h5py.File(dataset, "r") as handle:
        if handle.attrs.get("schema_version") != DATASET_SCHEMA:
            raise ValueError("reward/loss pilot requires transit-smdp-v7 data")
    dataset_digest = sha256(dataset)
    dataset_audit = json.loads(dataset_audit_path.read_text())
    if (
        dataset_audit.get("schema_version") != DATASET_AUDIT_SCHEMA
        or dataset_audit.get("passed") is not True
        or dataset_audit.get("dataset_sha256") != dataset_digest
        or dataset_audit.get("source_sha256") != dataset_audit_source_hashes()
    ):
        raise ValueError("reward/loss pilot dataset lacks a matching passed audit")
    reward_audit = json.loads(reward_audit_path.read_text())
    audited_scales = {
        float(row["reward_scale"])
        for row in reward_audit.get("reward_scale_candidates", {}).values()
    }
    if (
        reward_audit.get("schema_version") != REWARD_AUDIT_SCHEMA
        or reward_audit.get("dataset_sha256") != dataset_digest
        or reward_audit.get("source_sha256") != reward_audit_source_hashes()
        or audited_scales != set(DEFAULT_REWARD_SCALES)
    ):
        raise ValueError("reward/loss pilot reward audit is missing or stale")
    return dataset_audit, reward_audit


def run_candidate(
    replay: TransitReplayBufferV7,
    diagnostic_observations: torch.Tensor,
    *,
    reward_scale: float,
    critic_loss: str,
    target_entropy: float,
    seed: int,
    updates: int,
    device: torch.device,
) -> dict[str, Any]:
    config = trainer_config_for_method(
        "h2oplus",
        reward_scale=reward_scale,
        critic_loss=critic_loss,
        huber_delta=HUBER_DELTA,
        target_entropy=target_entropy,
    )
    trainer = TransitH2OTrainerV7(
        replay, config=config, device=device, seed=seed
    )
    initial = _parameter_summary(trainer)
    reports = []
    checkpoints = _checkpoints(updates)
    started = time.perf_counter()
    all_metrics_finite = True
    for update in range(1, updates + 1):
        report = update in checkpoints
        metrics = trainer.update(online=False, collect_metrics=report)
        if metrics:
            all_metrics_finite = all_metrics_finite and _finite_metrics(metrics)
            reports.append({"update": update, "metrics": metrics})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final = _parameter_summary(trainer)
    with torch.no_grad():
        actions, log_probability = trainer.actor(
            diagnostic_observations, deterministic=True
        )
    action_finite = bool(
        torch.isfinite(actions).all().item()
        and torch.isfinite(log_probability).all().item()
    )
    action_bounds = bool(
        torch.all(actions >= -1.0).item() and torch.all(actions <= 1.0).item()
    )
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
    parameter_limit = max(
        row["maximum_absolute_value"] for row in final.values()
    )
    eligible = bool(
        all_metrics_finite
        and parameters_finite
        and action_finite
        and action_bounds
        and reported_max_abs_td <= NUMERICAL_LIMIT
        and reported_max_grad <= NUMERICAL_LIMIT
        and parameter_limit <= NUMERICAL_LIMIT
    )
    result = {
        "reward_scale": reward_scale,
        "critic_loss": critic_loss,
        "huber_delta": HUBER_DELTA,
        "seed": seed,
        "trainer_config": asdict(config),
        "updates": updates,
        "seconds": elapsed,
        "updates_per_second": updates / elapsed,
        "reports": reports,
        "initial_parameters": initial,
        "final_parameters": final,
        "diagnostic_policy": {
            "rows": int(len(actions)),
            "all_finite": action_finite,
            "within_action_bounds": action_bounds,
            "exact_zero_rate": float((actions == -1.0).float().mean().cpu()),
            "mean_action": float(actions.mean().cpu()),
            "mean_log_probability": float(log_probability.mean().cpu()),
        },
        "numerical_gate": {
            "eligible": eligible,
            "all_reported_metrics_finite": all_metrics_finite,
            "all_parameters_finite": parameters_finite,
            "maximum_reported_abs_td": reported_max_abs_td,
            "maximum_reported_critic_gradient_norm": reported_max_grad,
            "maximum_final_parameter_absolute_value": parameter_limit,
            "limit": NUMERICAL_LIMIT,
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
        args.updates < MINIMUM_UPDATES
        or args.max_offline_rows <= 1
        or args.diagnostic_rows <= 0
    ):
        raise ValueError("reward/loss pilot sizes are invalid")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite numerical pilot {output}")
    dataset = args.dataset.expanduser().resolve()
    dataset_audit_path = args.dataset_audit.expanduser().resolve()
    reward_audit_path = args.reward_audit.expanduser().resolve()
    dataset_audit, reward_audit = _load_gate_inputs(
        dataset, dataset_audit_path, reward_audit_path
    )
    target_entropy = validate_v7_hybrid_target_entropy(args.target_entropy)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA reward/loss pilot requested but unavailable")
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
    runtime = collect_runtime_fingerprint(torch, str(device))
    replay = TransitReplayBufferV7(
        dataset,
        split="train",
        device=device,
        online_capacity=1,
        max_offline_rows=args.max_offline_rows,
        subsample_seed=9299,
        device_cache_observation_dtype=args.device_cache,
    )
    cache = replay.materialize_device_cache(include_online=False)
    diagnostic_rows = min(args.diagnostic_rows, replay.offline_size)
    diagnostic_observations = replay.sample(
        diagnostic_rows,
        scope="offline",
        rng=np.random.default_rng(9298),
    )["observations"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    runs = [
        run_candidate(
            replay,
            diagnostic_observations,
            reward_scale=reward_scale,
            critic_loss=critic_loss,
            target_entropy=target_entropy,
            seed=seed,
            updates=args.updates,
            device=device,
        )
        for reward_scale in DEFAULT_REWARD_SCALES
        for critic_loss in CRITIC_LOSSES
        for seed in PILOT_SEEDS
    ]
    paired_initialization = {}
    for seed in PILOT_SEEDS:
        seed_runs = [run for run in runs if run["seed"] == seed]
        for module in ("actor", "critic", "target_critic", "value", "ratio"):
            digests = {
                run["initial_parameters"][module]["state_sha256"]
                for run in seed_runs
            }
            paired_initialization[f"seed{seed}/{module}"] = len(digests) == 1
    if not all(paired_initialization.values()):
        raise RuntimeError("candidate pilots did not share paired initialization")
    gates = configuration_gates(runs)
    selection = select_reward_loss_configuration(reward_audit, gates)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": "numerical-pilot",
        "selection_boundary": (
            "uses immutable offline pilot data only; reserved evaluation cells are "
            "not read and these runs are not policy-effectiveness evidence"
        ),
        "design": {
            "reward_scales": list(DEFAULT_REWARD_SCALES),
            "critic_losses": list(CRITIC_LOSSES),
            "huber_delta": HUBER_DELTA,
            "seeds": list(PILOT_SEEDS),
            "updates_per_run": args.updates,
            "minimum_updates_per_run": MINIMUM_UPDATES,
            "numerical_limit": NUMERICAL_LIMIT,
            "paired_initialization_and_minibatch_rng": True,
        },
        "dataset": {
            "path": str(dataset),
            "sha256": sha256(dataset),
            "audit_path": str(dataset_audit_path),
            "audit_sha256": sha256(dataset_audit_path),
            "audit_schema_version": dataset_audit.get("schema_version"),
            "loaded_offline_rows": replay.offline_size,
        },
        "reward_audit": {
            "path": str(reward_audit_path),
            "sha256": sha256(reward_audit_path),
            "rows": reward_audit["rows"],
        },
        "target_entropy_inactive_during_iql_pilot": target_entropy,
        "device_cache": cache,
        "runtime": runtime,
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
        "runs": runs,
        "configuration_gates": gates,
        "configuration_selection": selection,
        "eligible_configurations": sorted(
            [
                [gate["reward_scale"], gate["critic_loss"]]
                for gate in gates.values()
                if gate["eligible"]
            ]
        ),
        "source_sha256": reward_loss_source_hashes(),
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
                "runs": len(runs),
                "eligible_configurations": payload["eligible_configurations"],
                "selected_configuration": selection["selected"],
            },
            sort_keys=True,
        )
    )
    if not selection["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
