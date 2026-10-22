#!/usr/bin/env python3
"""Benchmark canonical v4/v5/v6/v7 learner updates without policy evidence."""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import hash_files, sha256
from SimpleSAC.train_transit_v3 import (
    GPULease,
    collect_runtime_fingerprint,
    validate_v7_hybrid_target_entropy,
)
from SimpleSAC.transit_v3.protocols import get_method_profile
from SimpleSAC.transit_v3.trainer import TransitTrainerConfig
SCHEMA_VERSION = "transit-training-kernel-benchmark-v3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--mdp-version", choices=("v4", "v5", "v6", "v7"), default="v4"
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--phase", choices=("offline", "online"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--warmup-updates", type=int, default=100)
    parser.add_argument("--measured-updates", type=int, default=500)
    parser.add_argument("--chunk-updates", type=int, default=50)
    parser.add_argument("--online-rows", type=int, default=8192)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--max-offline-rows", type=int, default=0)
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--critic-loss", choices=("mse", "huber"))
    parser.add_argument("--huber-delta", type=float)
    parser.add_argument("--target-entropy", type=float)
    parser.add_argument(
        "--device-cache",
        choices=("none", "float32", "float16"),
        default="float32",
    )
    parser.add_argument("--compile-networks", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "max-autotune-no-cudagraphs"),
        default="default",
    )
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--allow-external-gpu-processes", action="store_true")
    parser.add_argument("--max-canonical-processes-per-gpu", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def trainer_config_for_method(
    method: str,
    *,
    reward_scale: float = 1.0,
    critic_loss: str | None = None,
    huber_delta: float | None = None,
    target_entropy: float | None = None,
) -> TransitTrainerConfig:
    profile = get_method_profile(method)
    return TransitTrainerConfig(
        ensemble_size=profile.ensemble_size,
        hidden_dim=256,
        batch_size=profile.batch_size,
        reward_scale=reward_scale,
        critic_loss=critic_loss or profile.critic_loss,
        huber_delta=(
            profile.huber_delta if huber_delta is None else huber_delta
        ),
        activation=profile.activation,
        actor_layer_norm=profile.actor_layer_norm,
        critic_layer_norm=profile.critic_layer_norm,
        value_layer_norm=profile.value_layer_norm,
        online_fraction=profile.online_fraction,
        actor_learning_rate=profile.actor_learning_rate,
        critic_learning_rate=profile.critic_learning_rate,
        value_learning_rate=profile.value_learning_rate,
        alpha_learning_rate=profile.alpha_learning_rate,
        target_subset_size=profile.target_subset_size,
        target_subset_replacement=profile.target_subset_replacement,
        cql_penalty_subset_size=profile.cql_penalty_subset_size,
        cql_penalty_subset_replacement=profile.cql_penalty_subset_replacement,
        target_entropy=(
            profile.target_entropy if target_entropy is None else target_entropy
        ),
        initial_temperature=profile.initial_temperature,
        backup_entropy=profile.backup_entropy,
        bootstrap_probability=profile.bootstrap_probability,
        offline_objective=profile.offline_objective,
        online_backup=profile.online_backup,
        actor_q_reduction=profile.actor_q_reduction,
        online_actor_update_interval=profile.actor_update_interval,
        online_actor_batch_mode=profile.actor_batch_mode,
        online_update_order=profile.online_update_order,
        offline_update_order=profile.offline_update_order,
        temperature_state_source=profile.temperature_state_source,
        offline_temperature_action_samples=(
            profile.offline_temperature_action_samples
        ),
        offline_cql_alpha=profile.offline_cql_alpha,
        cql_n_actions=10,
        cql_max_target_backup=profile.cql_max_target_backup,
        mixed_backup_lambda=profile.mixed_backup_lambda,
        iql_expectile=profile.iql_expectile,
        iql_advantage_temperature=profile.iql_advantage_temperature,
        use_ratio_correction=profile.use_ratio_correction,
        ratio_update_interval=profile.ratio_update_interval,
        gradient_clip_norm=profile.gradient_clip_norm,
    )


def _append_synthetic_online_rows(replay, count: int) -> None:
    """Copy immutable rows only to exercise the online-update compute path."""

    if count <= 0:
        raise ValueError("online benchmark rows must be positive")
    if count > replay.online_capacity:
        raise ValueError("online benchmark rows exceed replay capacity")
    for index in range(count):
        source = index % replay.offline_size
        duration_kwargs = (
            {"delta_t_seconds": float(replay.delta_t_seconds[source])}
            if replay.include_delta_t
            else {}
        )
        replay.append(
            replay.observations[source],
            replay.actions[source],
            float(replay.rewards[source]),
            replay.next_observations[source],
            float(replay.discounts[source]),
            float(replay.terminals[source]),
            episode_id=int(replay.episode_ids[source]) + 10**12,
            trajectory_id=int(replay.trajectory_ids[source]) + 10**12,
            **duration_kwargs,
        )


def _update_once(
    trainer, *, method: str, phase: str
) -> None:
    profile = get_method_profile(method)
    if phase == "offline":
        if profile.offline_objective == "none":
            raise ValueError(f"method {method!r} has no offline training phase")
        if profile.offline_objective == "bc":
            trainer.behavior_clone_step(collect_metrics=False)
        else:
            trainer.update(online=False, collect_metrics=False)
        return
    if profile.online_update_mode == "none":
        raise ValueError(f"method {method!r} has no online training phase")
    trainer.update(
        online=True,
        online_only=profile.online_update_mode == "online-only",
        collect_metrics=False,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.warmup_updates < 0 or args.measured_updates <= 0:
        raise ValueError("benchmark update counts are invalid")
    if args.chunk_updates <= 0:
        raise ValueError("chunk-updates must be positive")
    if args.mdp_version == "v7" and args.target_entropy is not None:
        args.target_entropy = validate_v7_hybrid_target_entropy(
            args.target_entropy
        )
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite benchmark {output}")
    dataset = args.dataset.expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    cuda_device_index = (
        torch.cuda.current_device() if device.index is None else device.index
    ) if device.type == "cuda" else None
    lease = GPULease(
        args.device,
        args.allow_shared_gpu,
        args.allow_external_gpu_processes,
        args.max_canonical_processes_per_gpu,
    )

    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.set_float32_matmul_precision("high")
    runtime_fingerprint = collect_runtime_fingerprint(torch, str(device))
    np.random.seed(args.seed)
    torch.random.default_generator.manual_seed(args.seed)
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.manual_seed(args.seed)
        initial_cuda_free_bytes, cuda_total_bytes = torch.cuda.mem_get_info(
            cuda_device_index
        )
        # PyTorch 2.4 requires a CUDA context query/allocation before resetting
        # peak counters after lazy seeding and thread-pool configuration.
        torch.cuda.reset_peak_memory_stats(cuda_device_index)
    else:
        initial_cuda_free_bytes = None
        cuda_total_bytes = None

    setup_started = time.perf_counter()
    if args.mdp_version == "v7":
        from SimpleSAC.transit_v7.replay_buffer import TransitReplayBufferV7
        from SimpleSAC.transit_v7.trainer import TransitH2OTrainerV7

        replay_class = TransitReplayBufferV7
        trainer_class = TransitH2OTrainerV7
    elif args.mdp_version == "v6":
        from SimpleSAC.transit_v6.replay_buffer import TransitReplayBufferV6
        from SimpleSAC.transit_v6.trainer import TransitH2OTrainerV6

        replay_class = TransitReplayBufferV6
        trainer_class = TransitH2OTrainerV6
    elif args.mdp_version == "v5":
        from SimpleSAC.transit_v5.replay_buffer import TransitReplayBufferV5
        from SimpleSAC.transit_v5.trainer import TransitH2OTrainerV5

        replay_class = TransitReplayBufferV5
        trainer_class = TransitH2OTrainerV5
    else:
        from SimpleSAC.transit_v4.replay_buffer import TransitReplayBufferV4
        from SimpleSAC.transit_v4.trainer import TransitH2OTrainerV4

        replay_class = TransitReplayBufferV4
        trainer_class = TransitH2OTrainerV4
    replay = replay_class(
        dataset,
        split="train",
        device=device,
        online_capacity=max(args.online_rows, 1),
        max_offline_rows=args.max_offline_rows,
        subsample_seed=args.seed,
        device_cache_observation_dtype=(
            args.device_cache if args.device_cache != "none" else "float32"
        ),
    )
    if args.phase == "online":
        _append_synthetic_online_rows(replay, args.online_rows)
    cache = (
        replay.materialize_device_cache(include_online=args.phase == "online")
        if args.device_cache != "none"
        else {"mode": "none"}
    )
    config = trainer_config_for_method(
        args.method,
        reward_scale=args.reward_scale,
        critic_loss=args.critic_loss,
        huber_delta=args.huber_delta,
        target_entropy=args.target_entropy,
    )
    trainer = trainer_class(
        replay,
        config=config,
        device=device,
        seed=args.seed,
    )
    compiled_modules = (
        trainer.compile_networks(mode=args.compile_mode)
        if args.compile_networks
        else []
    )
    if device.type == "cuda":
        _synchronize(device)
        setup_cuda_memory = {
            "allocated_bytes": torch.cuda.memory_allocated(cuda_device_index),
            "reserved_bytes": torch.cuda.memory_reserved(cuda_device_index),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                cuda_device_index
            ),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                cuda_device_index
            ),
        }
        torch.cuda.reset_peak_memory_stats(cuda_device_index)
    else:
        setup_cuda_memory = None
    setup_seconds = time.perf_counter() - setup_started

    warmup_started = time.perf_counter()
    for _ in range(args.warmup_updates):
        _update_once(trainer, method=args.method, phase=args.phase)
    _synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started

    chunk_seconds = []
    remaining = args.measured_updates
    while remaining:
        count = min(args.chunk_updates, remaining)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(count):
            _update_once(trainer, method=args.method, phase=args.phase)
        _synchronize(device)
        chunk_seconds.append(
            {"updates": count, "seconds": time.perf_counter() - started}
        )
        remaining -= count
    measured_seconds = sum(chunk["seconds"] for chunk in chunk_seconds)
    if device.type == "cuda":
        final_cuda_free_bytes, _ = torch.cuda.mem_get_info(cuda_device_index)
        training_cuda_memory = {
            "allocated_bytes": torch.cuda.memory_allocated(cuda_device_index),
            "reserved_bytes": torch.cuda.memory_reserved(cuda_device_index),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(
                cuda_device_index
            ),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(
                cuda_device_index
            ),
        }
    else:
        final_cuda_free_bytes = None
        training_cuda_memory = None

    source_paths = [
        Path(__file__).resolve(),
        ROOT / "SimpleSAC" / "transit_v3" / "domain_ratio.py",
        ROOT / "SimpleSAC" / "transit_v3" / "networks.py",
        ROOT / "SimpleSAC" / "transit_v3" / "protocols.py",
        ROOT / "SimpleSAC" / "transit_v3" / "replay_buffer.py",
        ROOT / "SimpleSAC" / "transit_v3" / "trainer.py",
        ROOT / "SimpleSAC" / "train_transit_v3.py",
    ]
    source_paths.extend(
        (
            ROOT / "bus_h2o" / f"mdp_contract_{args.mdp_version}.py",
            ROOT / "SimpleSAC" / f"transit_{args.mdp_version}" / "networks.py",
            ROOT
            / "SimpleSAC"
            / f"transit_{args.mdp_version}"
            / "replay_buffer.py",
            ROOT / "SimpleSAC" / f"transit_{args.mdp_version}" / "trainer.py",
        )
    )
    if args.mdp_version in {"v5", "v6", "v7"}:
        source_paths.extend(
            (
                ROOT / "bus_h2o" / "mdp_contract_v4.py",
                ROOT / "SimpleSAC" / "transit_v4" / "networks.py",
            )
        )
    if args.mdp_version == "v6":
        source_paths.extend(
            (
                ROOT / "bus_h2o" / "mdp_contract_v5.py",
                ROOT / "SimpleSAC" / "transit_v6" / "sampler.py",
            )
        )
    if args.mdp_version == "v7":
        source_paths.extend(
            (
                ROOT / "bus_h2o" / "mdp_contract_v5.py",
                ROOT / "bus_h2o" / "passenger_time.py",
                ROOT / "SimpleSAC" / "transit_v7" / "sampler.py",
            )
        )
    rates = [chunk["updates"] / chunk["seconds"] for chunk in chunk_seconds]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "disclosure": (
            "performance-only benchmark; online rows are copied offline rows and "
            "outputs are not valid policy checkpoints or scientific results"
        ),
        "method": args.method,
        "mdp_version": args.mdp_version,
        "phase": args.phase,
        "seed": args.seed,
        "dataset": str(dataset),
        "dataset_sha256": sha256(dataset),
        "trainer_config": asdict(config),
        "optimizer_config": trainer.optimizer_config,
        "device_cache": cache,
        "device_cache_mode": args.device_cache,
        "max_offline_rows": args.max_offline_rows,
        "compile_networks": args.compile_networks,
        "compile_mode": args.compile_mode,
        "allow_shared_gpu": args.allow_shared_gpu,
        "allow_external_gpu_processes": args.allow_external_gpu_processes,
        "max_canonical_processes_per_gpu": (
            lease.max_canonical_processes_per_gpu
            if device.type == "cuda"
            else None
        ),
        "compiled_modules": compiled_modules,
        "compiler_config": trainer.compiler_config,
        "warmup_updates": args.warmup_updates,
        "measured_updates": args.measured_updates,
        "setup_seconds": setup_seconds,
        "warmup_seconds": warmup_seconds,
        "measured_seconds": measured_seconds,
        "updates_per_second": args.measured_updates / measured_seconds,
        "chunk_updates_per_second": {
            "minimum": min(rates),
            "median": float(np.median(rates)),
            "maximum": max(rates),
        },
        "chunks": chunk_seconds,
        "resources": {
            "max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
            "cuda": (
                {
                    "total_bytes": cuda_total_bytes,
                    "initial_free_bytes": initial_cuda_free_bytes,
                    "final_free_bytes": final_cuda_free_bytes,
                    "setup": setup_cuda_memory,
                    "training": training_cuda_memory,
                }
                if device.type == "cuda"
                else None
            ),
        },
        "runtime": runtime_fingerprint,
        "source_sha256": hash_files(source_paths),
    }
    _write_json(output, payload)
    lease.close()
    print(json.dumps({
        "output": str(output),
        "updates_per_second": payload["updates_per_second"],
        "setup_seconds": setup_seconds,
        "warmup_seconds": warmup_seconds,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
