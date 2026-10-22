#!/usr/bin/env python3
"""Quantify FP16 replay-storage error and paired learner drift for v6/v7."""

from __future__ import annotations

import argparse
import atexit
import json
import math
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility.benchmark_training_kernel_v4 import (
    _append_synthetic_online_rows,
    trainer_config_for_method,
)
from reproducibility.run_hybrid_entropy_pilot_v7 import (
    SCHEMA_VERSION as ENTROPY_PILOT_SCHEMA,
    entropy_pilot_source_hashes,
)
from reproducibility.run_reward_loss_pilot_v7 import (
    SCHEMA_VERSION as REWARD_LOSS_PILOT_SCHEMA,
    reward_loss_source_hashes,
)
from SimpleSAC.train_transit_v3 import (
    GPULease,
    collect_runtime_fingerprint,
    validate_v7_hybrid_target_entropy,
)
from SimpleSAC.transit_v3.protocols import get_method_profile


SCHEMA_VERSION = "transit-device-cache-precision-audit-v5"
MINIMUM_PAIRED_UPDATES = 2_000
MINIMUM_PAIRED_OFFLINE_ROWS = 100_000
MINIMUM_ONLINE_ROWS = 4_096
MINIMUM_SAMPLE_ROWS = 8_192
PRECISION_THRESHOLDS = {
    "input_maximum_absolute_error": 0.002,
    "actor_encoded_state_normalized_root_mean_square_error": 0.001,
    "initial_action_maximum_absolute_error": 0.005,
    "parameter_normalized_root_mean_square_error": 0.025,
    "parameter_maximum_absolute_error": 0.1,
    "final_policy_normalized_root_mean_square_error": 0.05,
    "final_policy_maximum_absolute_error": 0.01,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--mdp-version", choices=("v6", "v7"), default="v6")
    parser.add_argument("--method", default="h2oplus")
    parser.add_argument("--phase", choices=("offline", "online"), default="online")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=9201)
    parser.add_argument(
        "--max-offline-rows", type=int, default=MINIMUM_PAIRED_OFFLINE_ROWS
    )
    parser.add_argument("--online-rows", type=int, default=MINIMUM_ONLINE_ROWS)
    parser.add_argument("--sample-rows", type=int, default=MINIMUM_SAMPLE_ROWS)
    parser.add_argument(
        "--paired-updates", type=int, default=MINIMUM_PAIRED_UPDATES
    )
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--critic-loss", choices=("mse", "huber"))
    parser.add_argument("--huber-delta", type=float)
    parser.add_argument("--target-entropy", type=float)
    parser.add_argument("--reward-loss-pilot", type=Path)
    parser.add_argument("--entropy-pilot", type=Path)
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--allow-external-gpu-processes", action="store_true")
    parser.add_argument("--max-canonical-processes-per-gpu", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def error_statistics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float().reshape(-1).cpu()
    candidate = candidate.detach().float().reshape(-1).cpu()
    if reference.shape != candidate.shape:
        raise ValueError("precision-audit tensors have different shapes")
    difference = candidate - reference
    absolute = difference.abs()
    rmse = difference.square().mean().sqrt()
    scale = reference.std(unbiased=False).clamp_min(1e-12)
    return {
        "elements": int(reference.numel()),
        "maximum_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "p99_absolute_error": float(torch.quantile(absolute, 0.99)),
        "root_mean_square_error": float(rmse),
        "normalized_root_mean_square_error": float(rmse / scale),
        "reference_minimum": float(reference.min()),
        "reference_maximum": float(reference.max()),
        "reference_standard_deviation": float(scale),
        "exact_fraction": float((difference == 0).float().mean()),
    }


def full_float16_storage_error(
    dataset: Path,
    *,
    split_id: int = 0,
    chunk_rows: int = 32_768,
) -> dict[str, Any]:
    """Scan every selected HDF5 state without materializing two full caches."""

    if chunk_rows <= 0:
        raise ValueError("full precision scan chunk size must be positive")
    arrays: dict[str, dict[str, float | int | bool]] = {}
    selected_rows = 0
    with h5py.File(dataset, "r") as handle:
        split = handle["split_id"]
        for name in ("observations", "next_observations"):
            source = handle[name]
            elements = 0
            exact = 0
            absolute_sum = 0.0
            squared_sum = 0.0
            reference_sum = 0.0
            reference_squared_sum = 0.0
            maximum = 0.0
            reference_minimum = float("inf")
            reference_maximum = float("-inf")
            finite = True
            rows_for_array = 0
            for start in range(0, len(source), chunk_rows):
                stop = min(start + chunk_rows, len(source))
                mask = np.asarray(split[start:stop]) == split_id
                if not mask.any():
                    continue
                reference = np.asarray(source[start:stop][mask], dtype=np.float32)
                candidate = reference.astype(np.float16).astype(np.float32)
                difference = candidate - reference
                absolute = np.abs(difference)
                finite = finite and bool(
                    np.isfinite(reference).all() and np.isfinite(candidate).all()
                )
                rows_for_array += len(reference)
                elements += reference.size
                exact += int(np.count_nonzero(difference == 0.0))
                absolute_sum += float(absolute.sum(dtype=np.float64))
                squared_sum += float(
                    np.square(difference, dtype=np.float64).sum(dtype=np.float64)
                )
                reference_sum += float(reference.sum(dtype=np.float64))
                reference_squared_sum += float(
                    np.square(reference, dtype=np.float64).sum(dtype=np.float64)
                )
                maximum = max(maximum, float(absolute.max(initial=0.0)))
                reference_minimum = min(reference_minimum, float(reference.min()))
                reference_maximum = max(reference_maximum, float(reference.max()))
            if elements <= 0:
                raise ValueError(f"full precision scan selected no rows for {name}")
            mean = reference_sum / elements
            variance = max(reference_squared_sum / elements - mean * mean, 0.0)
            scale = max(variance**0.5, 1e-12)
            rmse = (squared_sum / elements) ** 0.5
            arrays[name] = {
                "rows": rows_for_array,
                "elements": elements,
                "all_finite": finite,
                "maximum_absolute_error": maximum,
                "mean_absolute_error": absolute_sum / elements,
                "root_mean_square_error": rmse,
                "normalized_root_mean_square_error": rmse / scale,
                "reference_minimum": reference_minimum,
                "reference_maximum": reference_maximum,
                "reference_standard_deviation": scale,
                "exact_fraction": exact / elements,
            }
            if selected_rows and selected_rows != rows_for_array:
                raise RuntimeError("full precision scans selected inconsistent rows")
            selected_rows = rows_for_array
    return {
        "split_id": split_id,
        "training_rows": selected_rows,
        "chunk_rows": chunk_rows,
        "quantization": "IEEE float32 to float16 to float32",
        "arrays": arrays,
    }


def module_parameters(module: torch.nn.Module | None) -> torch.Tensor:
    if module is None:
        return torch.empty(0)
    parameters = [parameter.detach().reshape(-1).cpu() for parameter in module.parameters()]
    return torch.cat(parameters) if parameters else torch.empty(0)


def module_drift(
    reference_trainer: Any,
    candidate_trainer: Any,
) -> dict[str, dict[str, float]]:
    result = {}
    for name in ("actor", "critic", "target_critic", "value", "ratio"):
        reference = module_parameters(getattr(reference_trainer, name))
        candidate = module_parameters(getattr(candidate_trainer, name))
        result[name] = error_statistics(reference, candidate)
    return result


def capture_torch_rng(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    cuda_state = (
        torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    )
    return torch.get_rng_state(), cuda_state


def restore_torch_rng(
    state: tuple[torch.Tensor, torch.Tensor | None], device: torch.device
) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if device.type == "cuda" and cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def paired_batch(
    replay32: Any,
    replay16: Any,
    *,
    scope: str,
    rows: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    batch32 = replay32.sample(
        rows, scope=scope, rng=np.random.default_rng(seed)
    )
    batch16 = replay16.sample(
        rows, scope=scope, rng=np.random.default_rng(seed)
    )
    for key in batch32:
        if key not in {"observations", "next_observations"}:
            torch.testing.assert_close(batch32[key], batch16[key], rtol=0.0, atol=0.0)
    return batch32, batch16


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _same_float(first: Any, second: Any) -> bool:
    try:
        return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=0.0)
    except (TypeError, ValueError):
        return False


def validate_v7_selection_evidence(
    reward_loss_path: Path,
    entropy_path: Path,
    *,
    reward_scale: float,
    critic_loss: str | None,
    huber_delta: float | None,
    target_entropy: float | None,
) -> dict[str, Any]:
    """Bind the precision run to the exact upstream trainer selections."""

    reward_loss_path = reward_loss_path.expanduser().resolve(strict=True)
    entropy_path = entropy_path.expanduser().resolve(strict=True)
    reward_loss = json.loads(reward_loss_path.read_text())
    entropy = json.loads(entropy_path.read_text())
    selected = reward_loss.get("configuration_selection", {}).get("selected", {})
    entropy_configuration = entropy.get("reward_loss_configuration", {})
    entropy_gate = entropy.get("target_entropy_gate", {})
    valid = (
        reward_loss.get("schema_version") == REWARD_LOSS_PILOT_SCHEMA
        and reward_loss.get("study_stage") == "numerical-pilot"
        and reward_loss.get("configuration_selection", {}).get("passed") is True
        and reward_loss.get("source_sha256") == reward_loss_source_hashes()
        and entropy.get("schema_version") == ENTROPY_PILOT_SCHEMA
        and entropy.get("study_stage") == "numerical-pilot"
        and entropy_gate.get("passed") is True
        and entropy.get("source_sha256") == entropy_pilot_source_hashes()
        and entropy_configuration.get("pilot_sha256") == sha256(reward_loss_path)
        and _same_float(selected.get("reward_scale"), reward_scale)
        and selected.get("critic_loss") == critic_loss
        and _same_float(selected.get("huber_delta"), huber_delta)
        and _same_float(entropy_configuration.get("reward_scale"), reward_scale)
        and entropy_configuration.get("critic_loss") == critic_loss
        and _same_float(entropy_configuration.get("huber_delta"), huber_delta)
        and _same_float(entropy_gate.get("selected_target_entropy"), target_entropy)
    )
    if not valid:
        raise ValueError(
            "v7 precision audit requires the current selected reward/loss and "
            "entropy artifacts"
        )
    return {
        "reward_loss_pilot": {
            "path": str(reward_loss_path),
            "sha256": sha256(reward_loss_path),
            "schema_version": REWARD_LOSS_PILOT_SCHEMA,
        },
        "entropy_pilot": {
            "path": str(entropy_path),
            "sha256": sha256(entropy_path),
            "schema_version": ENTROPY_PILOT_SCHEMA,
        },
        "selected_arguments": {
            "reward_scale": float(reward_scale),
            "critic_loss": str(critic_loss),
            "huber_delta": float(huber_delta),
            "target_entropy": float(target_entropy),
        },
    }


def cache_precision_decision(
    *,
    input_error: dict[str, dict[str, float]],
    representation_error: dict[str, dict[str, float]],
    parameter_drift: dict[str, dict[str, float]],
    policy_drift: dict[str, float],
    paired_updates: int,
) -> dict[str, Any]:
    """Select FP16 only when every preregistered numerical bound passes."""

    input_maximum = max(
        row["maximum_absolute_error"] for row in input_error.values()
    )
    parameter_nrmse = max(
        row["normalized_root_mean_square_error"]
        for row in parameter_drift.values()
    )
    parameter_maximum = max(
        row["maximum_absolute_error"] for row in parameter_drift.values()
    )
    checks = {
        "minimum_paired_updates": paired_updates >= MINIMUM_PAIRED_UPDATES,
        "full_and_sampled_input_maximum_absolute_error": input_maximum
        <= PRECISION_THRESHOLDS["input_maximum_absolute_error"],
        "actor_encoded_state_normalized_root_mean_square_error": (
            representation_error["actor_encoded_state"][
                "normalized_root_mean_square_error"
            ]
            <= PRECISION_THRESHOLDS[
                "actor_encoded_state_normalized_root_mean_square_error"
            ]
        ),
        "initial_action_maximum_absolute_error": (
            representation_error["deterministic_action_from_quantized_input"][
                "maximum_absolute_error"
            ]
            <= PRECISION_THRESHOLDS["initial_action_maximum_absolute_error"]
        ),
        "parameter_normalized_root_mean_square_error": parameter_nrmse
        <= PRECISION_THRESHOLDS[
            "parameter_normalized_root_mean_square_error"
        ],
        "parameter_maximum_absolute_error": parameter_maximum
        <= PRECISION_THRESHOLDS["parameter_maximum_absolute_error"],
        "final_policy_normalized_root_mean_square_error": policy_drift[
            "normalized_root_mean_square_error"
        ]
        <= PRECISION_THRESHOLDS[
            "final_policy_normalized_root_mean_square_error"
        ],
        "final_policy_maximum_absolute_error": policy_drift[
            "maximum_absolute_error"
        ]
        <= PRECISION_THRESHOLDS["final_policy_maximum_absolute_error"],
    }
    float16_eligible = all(checks.values())
    return {
        "passed": True,
        "float16_eligible": float16_eligible,
        "selected_observation_storage_dtype": (
            "float16" if float16_eligible else "float32"
        ),
        "fallback_rule": "use float32 whenever any FP16 precision check fails",
        "thresholds": PRECISION_THRESHOLDS,
        "minimum_paired_updates": MINIMUM_PAIRED_UPDATES,
        "observed": {
            "paired_updates": paired_updates,
            "full_and_sampled_input_maximum_absolute_error": input_maximum,
            "parameter_normalized_root_mean_square_error": parameter_nrmse,
            "parameter_maximum_absolute_error": parameter_maximum,
        },
        "checks": checks,
    }


def precision_audit_source_hashes(mdp_version: str) -> dict[str, str]:
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "reproducibility" / "benchmark_training_kernel_v4.py",
        ROOT / "SimpleSAC" / "train_transit_v3.py",
        ROOT / "SimpleSAC" / "transit_v3" / "protocols.py",
        ROOT / "SimpleSAC" / "transit_v3" / "replay_buffer.py",
        ROOT / "SimpleSAC" / "transit_v3" / "trainer.py",
        ROOT / "SimpleSAC" / f"transit_{mdp_version}" / "networks.py",
        ROOT / "SimpleSAC" / f"transit_{mdp_version}" / "replay_buffer.py",
        ROOT / "SimpleSAC" / f"transit_{mdp_version}" / "trainer.py",
        ROOT / "bus_h2o" / f"mdp_contract_{mdp_version}.py",
    ]
    if mdp_version == "v7":
        source_paths.extend(
            (
                ROOT / "reproducibility" / "run_reward_loss_pilot_v7.py",
                ROOT / "reproducibility" / "run_hybrid_entropy_pilot_v7.py",
                ROOT / "SimpleSAC" / "transit_v4" / "networks.py",
                ROOT / "bus_h2o" / "mdp_contract_v5.py",
                ROOT / "bus_h2o" / "passenger_time.py",
            )
        )
    return {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in source_paths
    }


def main() -> None:
    args = parse_args()
    if (
        args.max_offline_rows < MINIMUM_PAIRED_OFFLINE_ROWS
        or args.sample_rows < MINIMUM_SAMPLE_ROWS
        or args.paired_updates < MINIMUM_PAIRED_UPDATES
        or args.online_rows < MINIMUM_ONLINE_ROWS
    ):
        raise ValueError("precision-audit coverage is below the registered minimum")
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite precision audit {output}")
    dataset = args.dataset.expanduser().resolve()
    selection_evidence = None
    if args.mdp_version == "v7":
        if args.reward_loss_pilot is None or args.entropy_pilot is None:
            raise ValueError(
                "v7 precision audit requires --reward-loss-pilot and --entropy-pilot"
            )
        selection_evidence = validate_v7_selection_evidence(
            args.reward_loss_pilot,
            args.entropy_pilot,
            reward_scale=args.reward_scale,
            critic_loss=args.critic_loss,
            huber_delta=args.huber_delta,
            target_entropy=args.target_entropy,
        )
    elif args.reward_loss_pilot is not None or args.entropy_pilot is not None:
        raise ValueError("v6 precision audit does not accept v7 selection evidence")
    full_storage_error = full_float16_storage_error(dataset)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA precision audit requested but CUDA is unavailable")
    profile = get_method_profile(args.method)
    if args.phase == "offline" and profile.offline_objective in {"none", "bc"}:
        raise ValueError("selected method has no gradient-based offline phase")
    if args.phase == "online" and profile.online_update_mode == "none":
        raise ValueError("selected method has no online phase")
    if args.mdp_version == "v7" and args.target_entropy is not None:
        args.target_entropy = validate_v7_hybrid_target_entropy(
            args.target_entropy
        )
    lease = GPULease(
        args.device,
        args.allow_shared_gpu,
        args.allow_external_gpu_processes,
        args.max_canonical_processes_per_gpu,
    )
    atexit.register(lease.close)

    if args.mdp_version == "v7":
        from SimpleSAC.transit_v7.replay_buffer import TransitReplayBufferV7
        from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7

        replay_class = TransitReplayBufferV7
        trainer_class = TransitH2OTrainerV7
    else:
        from SimpleSAC.transit_v6.replay_buffer import TransitReplayBufferV6
        from SimpleSAC.transit_v6.trainer import TransitH2OTrainerV6

        replay_class = TransitReplayBufferV6
        trainer_class = TransitH2OTrainerV6

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_float32_matmul_precision("high")

    replay_kwargs = {
        "split": "train",
        "device": device,
        "online_capacity": max(args.online_rows, 1),
        "max_offline_rows": args.max_offline_rows,
        "subsample_seed": args.seed,
    }
    replay32 = replay_class(
        dataset, device_cache_observation_dtype="float32", **replay_kwargs
    )
    replay16 = replay_class(
        dataset, device_cache_observation_dtype="float16", **replay_kwargs
    )
    if args.phase == "online":
        _append_synthetic_online_rows(replay32, args.online_rows)
        _append_synthetic_online_rows(replay16, args.online_rows)
    cache32 = replay32.materialize_device_cache(include_online=args.phase == "online")
    cache16 = replay16.materialize_device_cache(include_online=args.phase == "online")

    sampled_rows = min(args.sample_rows, replay32.offline_size)
    offline32, offline16 = paired_batch(
        replay32,
        replay16,
        scope="offline",
        rows=sampled_rows,
        seed=args.seed + 1,
    )
    input_error = {
        key: error_statistics(offline32[key], offline16[key])
        for key in ("observations", "next_observations")
    }
    if args.phase == "online":
        online_rows = min(args.sample_rows, replay32.online_size)
        online32, online16 = paired_batch(
            replay32,
            replay16,
            scope="online",
            rows=online_rows,
            seed=args.seed + 2,
        )
        input_error.update(
            {
                f"online_{key}": error_statistics(online32[key], online16[key])
                for key in ("observations", "next_observations")
            }
        )

    config = trainer_config_for_method(
        args.method,
        reward_scale=args.reward_scale,
        critic_loss=args.critic_loss,
        huber_delta=args.huber_delta,
        target_entropy=args.target_entropy,
    )
    trainer32 = trainer_class(
        replay32, config=config, device=device, seed=args.seed
    )
    trainer16 = trainer_class(
        replay16, config=config, device=device, seed=args.seed
    )
    initial_drift = module_drift(trainer32, trainer16)
    if any(stats["maximum_absolute_error"] for stats in initial_drift.values()):
        raise RuntimeError("paired trainers did not start from identical parameters")

    with torch.no_grad():
        encoded32 = trainer32.actor.encode_states(offline32["observations"])
        encoded16 = trainer32.actor.encode_states(offline16["observations"])
        initial_actions32, _ = trainer32.actor(
            offline32["observations"], deterministic=True
        )
        initial_actions16, _ = trainer32.actor(
            offline16["observations"], deterministic=True
        )
    representation_error = {
        "actor_encoded_state": error_statistics(encoded32, encoded16),
        "deterministic_action_from_quantized_input": error_statistics(
            initial_actions32, initial_actions16
        ),
    }

    for _ in range(args.paired_updates):
        rng_state = capture_torch_rng(device)
        trainer32.update(
            online=args.phase == "online",
            online_only=(
                args.phase == "online" and profile.online_update_mode == "online-only"
            ),
            collect_metrics=False,
        )
        restore_torch_rng(rng_state, device)
        trainer16.update(
            online=args.phase == "online",
            online_only=(
                args.phase == "online" and profile.online_update_mode == "online-only"
            ),
            collect_metrics=False,
        )

    drift = module_drift(trainer32, trainer16)
    with torch.no_grad():
        final_actions32, _ = trainer32.actor(
            offline32["observations"], deterministic=True
        )
        final_actions16, _ = trainer16.actor(
            offline32["observations"], deterministic=True
        )
    policy_drift = error_statistics(final_actions32, final_actions16)
    decision_input_error = dict(input_error)
    decision_input_error.update(
        {
            f"full_{name}": stats
            for name, stats in full_storage_error["arrays"].items()
        }
    )
    decision = cache_precision_decision(
        input_error=decision_input_error,
        representation_error=representation_error,
        parameter_drift=drift,
        policy_drift=policy_drift,
        paired_updates=args.paired_updates,
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "disclosure": (
            "numerical and performance audit only; synthetic online rows are copied "
            "from offline data and are not scientific policy evidence"
        ),
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "mdp_version": args.mdp_version,
        "method": args.method,
        "phase": args.phase,
        "seed": args.seed,
        "max_offline_rows": args.max_offline_rows,
        "sample_rows": sampled_rows,
        "online_rows": args.online_rows if args.phase == "online" else 0,
        "paired_updates": args.paired_updates,
        "registered_minimums": {
            "paired_offline_rows": MINIMUM_PAIRED_OFFLINE_ROWS,
            "sample_rows": MINIMUM_SAMPLE_ROWS,
            "online_rows": MINIMUM_ONLINE_ROWS,
            "paired_updates": MINIMUM_PAIRED_UPDATES,
        },
        "float32_cache": cache32,
        "float16_cache": cache16,
        "full_dataset_storage_error": full_storage_error,
        "sampled_input_error": input_error,
        "representation_error": representation_error,
        "parameter_drift_after_paired_updates": drift,
        "deterministic_policy_drift_on_identical_float32_inputs": policy_drift,
        "precision_decision": decision,
        "trainer_config": config.__dict__,
        "selection_evidence": selection_evidence,
        "gpu_lease": {
            "allow_shared_gpu": args.allow_shared_gpu,
            "allow_external_gpu_processes": args.allow_external_gpu_processes,
            "max_canonical_processes_per_gpu": (
                lease.max_canonical_processes_per_gpu
                if device.type == "cuda"
                else None
            ),
        },
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_cuda": str(torch.version.cuda),
            "device": str(device),
            "cpu_threads": torch.get_num_threads(),
        },
        "runtime_fingerprint": collect_runtime_fingerprint(torch, str(device)),
        "source_sha256": precision_audit_source_hashes(args.mdp_version),
    }
    write_json(output, payload)
    lease.close()
    print(
        json.dumps(
            {
                "output": str(output),
                "input_max_abs": decision["observed"][
                    "full_and_sampled_input_maximum_absolute_error"
                ],
                "encoded_state_nrmse": representation_error[
                    "actor_encoded_state"
                ]["normalized_root_mean_square_error"],
                "policy_drift_max_abs": policy_drift["maximum_absolute_error"],
                "selected_observation_storage_dtype": decision[
                    "selected_observation_storage_dtype"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
