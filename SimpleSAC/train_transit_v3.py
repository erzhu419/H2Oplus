#!/usr/bin/env python3
"""Train the canonical all-lines transit controller with auditable phases."""

from __future__ import annotations

import argparse
import csv
import ctypes
import fcntl
import hashlib
from importlib import metadata as importlib_metadata
import json
import logging
import os
import pickle
import platform
import random
import re
import resource
import shlex
import shutil
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SimpleSAC.transit_v3.protocols import METHOD_PROFILES, MethodProfile


METHOD_PROTOCOLS = {
    name: profile.description for name, profile in METHOD_PROFILES.items()
}
OFFLINE_PRETRAIN_METHODS = {
    name
    for name, profile in METHOD_PROFILES.items()
    if profile.offline_objective != "none"
}
ONLINE_METHODS = {
    name
    for name, profile in METHOD_PROFILES.items()
    if profile.online_update_mode != "none"
}
ONLINE_ONLY_UPDATE_METHODS = {
    name
    for name, profile in METHOD_PROFILES.items()
    if profile.online_update_mode == "online-only"
}
NVIDIA_SMI_FALLBACKS = (
    Path("/usr/lib/wsl/lib/nvidia-smi"),
    Path("/usr/bin/nvidia-smi"),
    Path("/usr/local/bin/nvidia-smi"),
)
GPU_LEASE_ROOT = Path("/tmp")
MAX_CANONICAL_GPU_PROCESSES = 16
HYBRID_ACTION_REFERENCE_MEASURE_MASS = 3.0
V7_SHARED_IQL_BRANCH_METHODS = {
    "h2oplus-no-ratio",
    "h2oplus-lambda0",
    "h2oplus-lambda0p2",
    "h2oplus-lambda0p5",
    "h2oplus-lambda1",
}


def validate_v7_hybrid_target_entropy(value: float) -> float:
    """Validate entropy relative to the atom-plus-interval base measure.

    The reference measure has mass one at zero hold and Lebesgue mass two on
    the positive-action interval. Its maximum entropy is log(3), while
    differential entropy below that maximum has no finite lower bound.
    """

    target = float(value)
    maximum = float(np.log(HYBRID_ACTION_REFERENCE_MEASURE_MASS))
    if not np.isfinite(target) or target >= maximum:
        raise ValueError(
            "v7 hybrid-entropy target must be finite and less than log(3)"
        )
    return target


def validate_v7_training_admission_runtime(
    path: Path,
    *,
    dataset_sha256: str,
    simulator: dict[str, Any],
    sources: dict[str, str],
) -> dict[str, Any]:
    """Recheck the formal certificate at the trainer process boundary."""

    admission_path = path.expanduser().resolve()
    payload = json.loads(admission_path.read_text())
    if (
        payload.get("schema_version") != "transit-training-admission-v7-v2"
        or payload.get("study_stage") != "formal"
        or payload.get("passed") is not True
        or payload.get("dataset", {}).get("sha256") != dataset_sha256
        or payload.get("simulator", {}).get("aggregate_sha256")
        != simulator.get("aggregate_sha256")
        or payload.get("simulator", {}).get("runtime_sha256")
        != simulator.get("runtime_sha256")
        or payload.get("training_source_sha256") != sources
    ):
        raise ValueError("v7 training admission is stale or bound to another run")
    return {
        "path": str(admission_path),
        "sha256": sha256(admission_path),
        "bytes": admission_path.stat().st_size,
        "schema_version": payload["schema_version"],
        "selected_arguments": payload.get("selected_arguments"),
    }


def validate_v7_offline_initializer_runtime(
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
    if method not in V7_SHARED_IQL_BRANCH_METHODS | {"h2oplus"}:
        raise ValueError("method is not registered for shared IQL initialization")
    expected = {
        "phase": "offline_pretraining",
        "source_method": "h2oplus",
        "source_training_seed": seed,
        "offline_updates_completed": offline_updates,
        "trainer_total_updates": trainer_total_updates,
        "dataset_sha256": dataset_sha256,
        "simulator_runtime_sha256": simulator["runtime_sha256"],
        "simulator_aggregate_sha256": simulator["aggregate_sha256"],
        "source_sha256": sources,
        "contract_config": contract_config,
        "mdp_version": "v7",
        "training_admission": training_admission,
    }
    mismatches = {
        name: {"initializer": extra.get(name), "training": value}
        for name, value in expected.items()
        if extra.get(name) != value
    }
    source_signature = extra.get("source_run_signature")
    source_run_name = extra.get("source_run_name")
    if source_signature != expected_source_signature:
        mismatches["source_run_signature"] = {
            "initializer": source_signature,
            "training": expected_source_signature,
        }
    if source_run_name != f"h2oplus_seed{seed}":
        mismatches["source_run_name"] = {
            "initializer": source_run_name,
            "training": f"h2oplus_seed{seed}",
        }
    if not isinstance(extra.get("offline_metrics"), dict):
        mismatches["offline_metrics"] = {
            "initializer": type(extra.get("offline_metrics")).__name__,
            "training": "dict",
        }
    if mismatches:
        raise ValueError(f"v7 offline-initializer provenance mismatch: {mismatches}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "schema_version": "transit-offline-initializer-v7-v1",
        "source_method": "h2oplus",
        "source_run_name": source_run_name,
        "source_run_signature": source_signature,
        "source_training_seed": seed,
        "offline_updates": offline_updates,
    }


def make_warmup_policy(
    profile: MethodProfile,
    policy,
    *,
    seed: int,
):
    """Resolve the registered pre-update data-collection policy."""

    if profile.warmup_action_mode == "policy":
        return policy
    if profile.warmup_action_mode != "uniform-box":
        raise ValueError(
            f"unsupported warmup action mode {profile.warmup_action_mode!r}"
        )
    rng = np.random.default_rng(int(seed) ^ 0x524C5044)

    def uniform_box_policy(
        observations: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        del deterministic
        batch_size = np.asarray(observations).shape[0]
        return rng.uniform(-1.0, 1.0, size=(batch_size, 1)).astype(np.float32)

    return uniform_box_policy


def parse_args(mdp_version: str = "v3") -> argparse.Namespace:
    if mdp_version not in {"v3", "v4", "v5", "v6", "v7"}:
        raise ValueError("mdp_version must be one of v3, v4, v5, v6, or v7")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHOD_PROTOCOLS), default="h2oplus")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--run-signature", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument(
        "--max-canonical-processes-per-gpu",
        type=int,
        help=(
            "hard per-GPU process quota for canonical jobs; defaults to one. "
            "This is mutually exclusive with the legacy --allow-shared-gpu flag"
        ),
    )
    parser.add_argument(
        "--allow-external-gpu-processes",
        action="store_true",
        help=(
            "allow unrelated compute processes on the selected GPU while retaining "
            "the configured canonical-process quota"
        ),
    )
    parser.add_argument("--cpu-threads", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(
            ROOT
            / "bus_h2o"
            / f"datasets_{mdp_version}"
            / f"transit_smdp_{mdp_version}.h5"
        ),
    )
    parser.add_argument(
        "--sim-env",
        type=Path,
        default=ROOT / "bus_h2o" / "calibrated_env",
    )
    if mdp_version == "v7":
        parser.add_argument("--training-admission", type=Path)
        parser.add_argument(
            "--offline-initializer",
            type=Path,
            help=(
                "same-seed formal H2O+ IQL prefix used only by registered "
                "ablation and mixed-backup sensitivity branches"
            ),
        )
        parser.add_argument(
            "--offline-initializer-source-signature",
            help="registered run signature of the same-seed H2O+ prefix producer",
        )
        parser.add_argument(
            "--sim-demand-scales",
            type=float,
            nargs="+",
            default=[0.75, 1.0, 1.25],
            help=(
                "episode-level online demand schedule; formal v7 uses "
                "0.75, 1.0, and 1.25 in this order"
            ),
        )
    else:
        parser.add_argument("--sim-demand-scale", type=float, default=1.0)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT
        / (
            "experiment_output_smdp_v7"
            if mdp_version == "v7"
            else "experiment_output_v7"
            if mdp_version == "v6"
            else "experiment_output_v6"
            if mdp_version == "v5"
            else "experiment_output_v5"
            if mdp_version == "v4"
            else "experiment_output_v4"
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume an existing canonical run from run_dir/checkpoints/latest.pt",
    )
    parser.add_argument("--max-offline-rows", type=int, default=0)
    parser.add_argument("--online-capacity", type=int, default=500_000)
    parser.add_argument("--offline-updates", type=int)
    parser.add_argument("--warmup-events", type=int)
    parser.add_argument("--ratio-pretrain-updates", type=int)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--events-per-epoch", type=int, default=500)
    parser.add_argument("--interaction-chunk-events", type=int, default=16)
    parser.add_argument("--updates-per-epoch", type=int)
    parser.add_argument("--utd-ratio", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=None if mdp_version == "v7" else 1.0,
    )
    parser.add_argument("--critic-loss", choices=("mse", "huber"))
    parser.add_argument("--huber-delta", type=float)
    parser.add_argument("--ensemble-size", type=int)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--online-fraction", type=float)
    parser.add_argument("--target-entropy", type=float)
    parser.add_argument("--initial-temperature", type=float)
    parser.add_argument("--iql-advantage-temperature", type=float)
    parser.add_argument("--offline-cql-alpha", type=float)
    parser.add_argument("--cql-n-actions", type=int, default=10)
    parser.add_argument("--ratio-update-interval", type=int)
    parser.add_argument("--offline-checkpoint-every", type=int, default=10_000)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--resource-interval", type=float, default=5.0)
    parser.add_argument("--no-device-cache", action="store_true")
    parser.add_argument(
        "--device-cache-observation-dtype",
        choices=("float32", "float16"),
        default="float32",
        help="storage dtype only; sampled network inputs are always float32",
    )
    parser.add_argument(
        "--compile-networks",
        action="store_true",
        help="compile repeated actor, critic, value, and ratio MLP kernels",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "max-autotune-no-cudagraphs"),
        default="default",
        help="safe torch.compile mode; CUDA Graph modes are excluded",
    )
    return parser.parse_args()


def resolve_method_arguments(
    args: argparse.Namespace, profile: MethodProfile
) -> dict[str, dict[str, Any]]:
    """Apply method defaults while preserving every explicit CLI override."""

    defaults = {
        "offline_updates": profile.default_offline_updates,
        "warmup_events": profile.default_warmup_events,
        "ratio_pretrain_updates": profile.default_ratio_pretrain_updates,
        "utd_ratio": profile.utd_ratio,
        "batch_size": profile.batch_size,
        "ensemble_size": profile.ensemble_size,
        "online_fraction": profile.online_fraction,
        "offline_cql_alpha": profile.offline_cql_alpha,
        "target_entropy": profile.target_entropy,
        "initial_temperature": profile.initial_temperature,
        "critic_loss": profile.critic_loss,
        "huber_delta": profile.huber_delta,
        "iql_advantage_temperature": profile.iql_advantage_temperature,
        "ratio_update_interval": profile.ratio_update_interval,
    }
    overrides = {}
    applied = {}
    for name, default in defaults.items():
        supplied = getattr(args, name)
        if supplied is None:
            setattr(args, name, default)
            applied[name] = default
        else:
            overrides[name] = {"profile": default, "effective": supplied}
    if args.updates_per_epoch is None:
        args.updates_per_epoch = (
            args.events_per_epoch * args.utd_ratio
            if profile.online_update_mode != "none"
            else 0
        )
        applied["updates_per_epoch"] = args.updates_per_epoch
    else:
        expected = (
            args.events_per_epoch * args.utd_ratio
            if profile.online_update_mode != "none"
            else 0
        )
        overrides["updates_per_epoch"] = {
            "profile": expected,
            "effective": args.updates_per_epoch,
        }
    return {"defaults_applied": applied, "overrides": overrides}


def _run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def _git_provenance() -> dict[str, Any]:
    """Record optional Git context without making the Git CLI a runtime dependency."""

    git = shutil.which("git")
    if git is None:
        return {
            "cli_available": False,
            "commit": None,
            "status": None,
            "errors": ["git executable is unavailable"],
        }

    probes = {
        "commit": [git, "rev-parse", "HEAD"],
        "status": [git, "status", "--short"],
    }
    values: dict[str, str | None] = {}
    errors: list[str] = []
    for name, command in probes.items():
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            values[name] = None
            errors.append(f"{name}: {type(error).__name__}: {error}")
            continue
        if result.returncode == 0:
            values[name] = result.stdout.strip()
        else:
            values[name] = None
            detail = result.stderr.strip() or f"exit code {result.returncode}"
            errors.append(f"{name}: {detail}")
    return {
        "cli_available": True,
        "commit": values["commit"],
        "status": values["status"],
        "errors": errors,
    }


def resolve_nvidia_smi(
    fallbacks: tuple[Path, ...] = NVIDIA_SMI_FALLBACKS,
) -> str | None:
    """Resolve nvidia-smi even when WSL omits its driver directory from PATH."""

    discovered = shutil.which("nvidia-smi")
    if discovered is not None:
        return discovered
    for candidate in fallbacks:
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return str(candidate)
    return None


def _physical_gpu(device: str) -> str:
    index = 0
    if device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if index >= len(devices):
            raise ValueError("requested CUDA device is outside CUDA_VISIBLE_DEVICES")
        return devices[index]
    return str(index)


class GPULease:
    """Enforce a fail-closed, cross-process quota for canonical GPU jobs."""

    def __init__(
        self,
        requested_device: str,
        allow_shared: bool = False,
        allow_external_processes: bool = False,
        max_canonical_processes_per_gpu: int | None = None,
    ) -> None:
        self.handle = None
        self.physical_gpu = None
        self.slot_index = None
        self.nvidia_smi = resolve_nvidia_smi()
        if max_canonical_processes_per_gpu is None:
            max_processes = MAX_CANONICAL_GPU_PROCESSES if allow_shared else 1
        else:
            if allow_shared:
                raise ValueError(
                    "--allow-shared-gpu cannot be combined with an explicit "
                    "canonical-process quota"
                )
            max_processes = max_canonical_processes_per_gpu
        if (
            isinstance(max_processes, bool)
            or not isinstance(max_processes, int)
            or not 1 <= max_processes <= MAX_CANONICAL_GPU_PROCESSES
        ):
            raise ValueError(
                "max_canonical_processes_per_gpu must lie in "
                f"[1, {MAX_CANONICAL_GPU_PROCESSES}]"
            )
        self.max_canonical_processes_per_gpu = max_processes
        wants_cuda = requested_device.startswith("cuda") or (
            requested_device == "auto" and self.nvidia_smi is not None
        )
        if not wants_cuda:
            return
        if self.nvidia_smi is None:
            raise RuntimeError(
                "CUDA was requested but nvidia-smi was not found in PATH or a known "
                "Linux/WSL driver location"
            )
        self.physical_gpu = _physical_gpu(
            "cuda:0" if requested_device == "auto" else requested_device
        )
        lock_name = re.sub(r"[^A-Za-z0-9._-]", "_", str(self.physical_gpu))
        GPU_LEASE_ROOT.mkdir(parents=True, exist_ok=True)
        allocation_handle = open(
            GPU_LEASE_ROOT / f"transit_h2o_v3_gpu_{lock_name}.allocation.lock",
            "a+",
        )
        try:
            fcntl.flock(allocation_handle, fcntl.LOCK_EX)
            active_slots: dict[int, dict[str, Any]] = {}
            free_slots: list[int] = []
            for slot_index in range(MAX_CANONICAL_GPU_PROCESSES):
                slot_path = GPU_LEASE_ROOT / (
                    f"transit_h2o_v3_gpu_{lock_name}.slot{slot_index}.lock"
                )
                slot_handle = open(slot_path, "a+")
                try:
                    fcntl.flock(slot_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    slot_handle.seek(0)
                    try:
                        metadata = json.loads(slot_handle.read())
                    except (json.JSONDecodeError, TypeError) as error:
                        raise RuntimeError(
                            f"GPU {self.physical_gpu} has an occupied canonical "
                            f"lease slot with invalid metadata: {slot_path}"
                        ) from error
                    active_slots[slot_index] = metadata
                else:
                    free_slots.append(slot_index)
                    fcntl.flock(slot_handle, fcntl.LOCK_UN)
                finally:
                    slot_handle.close()

            incompatible = {
                slot: metadata.get("max_canonical_processes_per_gpu")
                for slot, metadata in active_slots.items()
                if metadata.get("max_canonical_processes_per_gpu") != max_processes
            }
            if incompatible:
                raise RuntimeError(
                    f"GPU {self.physical_gpu} has canonical jobs registered with "
                    f"a different per-GPU quota: {incompatible}"
                )
            available = [slot for slot in free_slots if slot < max_processes]
            if len(active_slots) >= max_processes or not available:
                raise RuntimeError(
                    f"GPU {self.physical_gpu} already has {len(active_slots)} "
                    f"canonical process(es), reaching its hard quota of {max_processes}"
                )

            self.slot_index = min(available)
            slot_path = GPU_LEASE_ROOT / (
                f"transit_h2o_v3_gpu_{lock_name}.slot{self.slot_index}.lock"
            )
            self.handle = open(slot_path, "a+")
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.handle.seek(0)
            self.handle.truncate()
            self.handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "started_utc": datetime.now(timezone.utc).isoformat(),
                        "physical_gpu": self.physical_gpu,
                        "slot_index": self.slot_index,
                        "max_canonical_processes_per_gpu": max_processes,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            self.handle.flush()

            existing = _run(
                [
                    self.nvidia_smi,
                    "-i",
                    self.physical_gpu,
                    "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits",
                ]
            )
            canonical_pids = {
                metadata.get("pid")
                for metadata in active_slots.values()
                if isinstance(metadata.get("pid"), int)
            }
            other_pids = {
                int(value)
                for value in existing.splitlines()
                if value.strip().isdigit() and int(value) != os.getpid()
            }
            external_pids = other_pids - canonical_pids
            if external_pids and not allow_external_processes:
                raise RuntimeError(
                    f"GPU {self.physical_gpu} is already used by external compute "
                    f"processes {sorted(external_pids)}"
                )
        except Exception:
            self.close()
            raise
        finally:
            fcntl.flock(allocation_handle, fcntl.LOCK_UN)
            allocation_handle.close()

    def close(self) -> None:
        if self.handle is not None:
            self.handle.seek(0)
            self.handle.truncate()
            self.handle.flush()
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


class _NvmlSampler:
    """Minimal in-process NVML reader without an optional Python dependency."""

    class _Utilization(ctypes.Structure):
        _fields_ = (("gpu", ctypes.c_uint), ("memory", ctypes.c_uint))

    class _Memory(ctypes.Structure):
        _fields_ = (
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        )

    def __init__(self, physical_gpu: str) -> None:
        if not str(physical_gpu).isdigit():
            raise ValueError("the dependency-free NVML reader requires a GPU index")
        self.library = ctypes.CDLL("libnvidia-ml.so.1")
        self.library.nvmlInit_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetHandleByIndex_v2.argtypes = (
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_void_p),
        )
        self.library.nvmlDeviceGetHandleByIndex_v2.restype = ctypes.c_int
        self.library.nvmlDeviceGetUtilizationRates.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(self._Utilization),
        )
        self.library.nvmlDeviceGetUtilizationRates.restype = ctypes.c_int
        self.library.nvmlDeviceGetMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(self._Memory),
        )
        self.library.nvmlDeviceGetMemoryInfo.restype = ctypes.c_int
        self.library.nvmlShutdown.restype = ctypes.c_int
        if self.library.nvmlInit_v2() != 0:
            raise RuntimeError("NVML initialization failed")
        self.handle = ctypes.c_void_p()
        if self.library.nvmlDeviceGetHandleByIndex_v2(
            int(physical_gpu), ctypes.byref(self.handle)
        ) != 0:
            self.library.nvmlShutdown()
            raise RuntimeError(f"NVML could not open GPU {physical_gpu}")
        self.initialized = True

    def sample(self) -> tuple[str, str, str]:
        utilization = self._Utilization()
        memory = self._Memory()
        if self.library.nvmlDeviceGetUtilizationRates(
            self.handle, ctypes.byref(utilization)
        ) != 0:
            raise RuntimeError("NVML utilization query failed")
        if self.library.nvmlDeviceGetMemoryInfo(
            self.handle, ctypes.byref(memory)
        ) != 0:
            raise RuntimeError("NVML memory query failed")
        return (
            str(int(utilization.gpu)),
            str(int(utilization.memory)),
            str(int(round(memory.used / 2**20))),
        )

    def close(self) -> None:
        if getattr(self, "initialized", False):
            self.library.nvmlShutdown()
            self.initialized = False


class ResourceMonitor:
    def __init__(self, path: Path, interval: float, physical_gpu: str | None) -> None:
        self.path = path
        self.interval = max(float(interval), 1.0)
        self.physical_gpu = physical_gpu
        self.phase = "setup"
        self.started = time.perf_counter()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self._psutil = None
        self._process = None
        self._children: dict[int, Any] = {}
        self._nvml = None
        self._nvidia_smi = (
            resolve_nvidia_smi() if self.physical_gpu is not None else None
        )
        if self.physical_gpu is not None:
            try:
                self._nvml = _NvmlSampler(self.physical_gpu)
            except Exception:
                self._nvml = None
        try:
            import psutil

            self._psutil = psutil
            self._process = psutil.Process()
            self._process.cpu_percent(None)
        except Exception:
            pass

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 2.0)
        if self._nvml is not None:
            self._nvml.close()

    def _process_stats(self) -> tuple[float, float, float, float]:
        if self._process is not None:
            try:
                process = self._process
                children = process.children(recursive=True)
                live_pids = {child.pid for child in children}
                self._children = {
                    pid: child for pid, child in self._children.items() if pid in live_pids
                }
                for child in children:
                    if child.pid not in self._children:
                        self._children[child.pid] = child
                        child.cpu_percent(None)
                child_cpu = sum(
                    child.cpu_percent(None) for child in self._children.values()
                )
                child_rss = sum(
                    child.memory_info().rss for child in self._children.values()
                )
                return (
                    process.cpu_percent(None),
                    process.memory_info().rss / 2**20,
                    child_cpu,
                    child_rss / 2**20,
                )
            except Exception:
                pass
        try:
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            return (
                0.0,
                rss,
                0.0,
                0.0,
            )
        except Exception:
            return 0.0, 0.0, 0.0, 0.0

    def _gpu_stats(self) -> tuple[str, str, str]:
        if self.physical_gpu is None:
            return "", "", ""
        if self._nvml is not None:
            try:
                return self._nvml.sample()
            except Exception:
                self._nvml.close()
                self._nvml = None
        if self._nvidia_smi is None:
            return "", "", ""
        output = _run(
            [
                self._nvidia_smi,
                "-i",
                self.physical_gpu,
                "--query-gpu=utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        values = [item.strip() for item in output.split(",")]
        return tuple((values + ["", "", ""])[:3])

    def _run(self) -> None:
        with self.path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                (
                    "utc",
                    "elapsed_sec",
                    "phase",
                    "process_cpu_percent",
                    "process_rss_mb",
                    "children_cpu_percent",
                    "children_rss_mb",
                    "gpu_util_percent",
                    "gpu_memory_util_percent",
                    "gpu_memory_used_mb",
                )
            )
            while not self.stop_event.is_set():
                cpu, rss, child_cpu, child_rss = self._process_stats()
                gpu_util, gpu_mem_util, gpu_mem = self._gpu_stats()
                writer.writerow(
                    (
                        datetime.now(timezone.utc).isoformat(),
                        f"{time.perf_counter() - self.started:.3f}",
                        self.phase,
                        f"{cpu:.1f}",
                        f"{rss:.1f}",
                        f"{child_cpu:.1f}",
                        f"{child_rss:.1f}",
                        gpu_util,
                        gpu_mem_util,
                        gpu_mem,
                    )
                )
                handle.flush()
                self.stop_event.wait(self.interval)


@contextmanager
def timed_phase(name: str, timings: dict[str, float], monitor: ResourceMonitor, torch):
    previous = monitor.phase
    monitor.phase = name
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - started
        monitor.phase = previous


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


RUNTIME_DISTRIBUTIONS = (
    "torch",
    "numpy",
    "h5py",
    "pandas",
    "gymnasium",
    "openpyxl",
    "matplotlib",
    "pytest",
    "psutil",
)


def _compiler_fingerprint() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return path-independent compiler identity plus host path observations."""

    configured = os.environ.get("CC") or shutil.which("cc")
    if not configured:
        return (
            {"sha256": None, "version": None},
            {"path": None, "configured": os.environ.get("CC")},
        )
    command = shlex.split(configured)
    if not command:
        raise ValueError("CC does not contain a compiler command")
    executable = shutil.which(command[0]) or command[0]
    resolved = Path(executable).expanduser().resolve(strict=True)
    result = subprocess.run(
        [str(resolved), "--version"],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    version_output = result.stdout.strip() or result.stderr.strip()
    version = version_output.splitlines()[0] if version_output else None
    required = {
        "sha256": sha256(resolved),
        "version": version,
    }
    host = {
        "path": str(resolved),
        "configured": os.environ.get("CC"),
    }
    cpath = os.environ.get("CPATH")
    if cpath:
        required_entries = []
        host_entries = []
        for raw_path in cpath.split(os.pathsep):
            if not raw_path:
                continue
            include_dir = Path(raw_path).expanduser().resolve(strict=True)
            crypt_header = include_dir / "crypt.h"
            crypt_h_sha256 = sha256(crypt_header) if crypt_header.is_file() else None
            required_entries.append(
                {
                    "crypt_h_sha256": crypt_h_sha256,
                }
            )
            host_entries.append(
                {
                    "path": str(include_dir),
                    "crypt_h_sha256": crypt_h_sha256,
                }
            )
        required["cpath"] = {"entries": required_entries}
        host["cpath"] = {
            "value": cpath,
            "entries": host_entries,
        }
    return required, host


def collect_runtime_fingerprint(torch, device: str) -> dict[str, Any]:
    """Separate continuation-critical software from host observations."""

    package_versions = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        if distribution == "torch":
            package_versions[distribution] = str(torch.__version__)
            continue
        try:
            package_versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            package_versions[distribution] = "MISSING"
    torch_root = Path(torch.__file__).resolve().parent
    torch_binary_paths = {
        "python_extension": Path(torch._C.__file__).resolve(),
        "cuda_library": torch_root / "lib" / "libtorch_cuda.so",
    }
    torch_binary_sha256 = {
        name: sha256(path) if path.is_file() else None
        for name, path in torch_binary_paths.items()
    }
    try:
        torch_distribution_version = importlib_metadata.version("torch")
    except importlib_metadata.PackageNotFoundError:
        torch_distribution_version = "MISSING"
    uses_cuda = str(device).startswith("cuda")
    libc_name, libc_version = platform.libc_ver()
    compiler_required, compiler_host = _compiler_fingerprint()
    return {
        "required": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_cache_tag": getattr(sys.implementation, "cache_tag", None),
            "python_executable_sha256": sha256(Path(sys.executable).resolve()),
            "packages": package_versions,
            "torch_cuda_version": str(torch.version.cuda),
            "torch_binary_sha256": torch_binary_sha256,
            "compiler": compiler_required,
        },
        "host": {
            "platform": platform.platform(),
            "libc": {"name": libc_name, "version": libc_version},
            "python_executable": sys.executable,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if uses_cuda else None,
            "cpu_threads": torch.get_num_threads(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cc_environment": os.environ.get("CC"),
            "cpath_environment": os.environ.get("CPATH"),
            "nvidia_smi": resolve_nvidia_smi(),
            "compiler": compiler_host,
            "torch_distribution_version": torch_distribution_version,
        },
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_pickle(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return sha256(path)


def read_runtime_state(run_dir: Path, descriptor: dict[str, Any]) -> Any:
    relative = Path(descriptor.get("path", ""))
    if not relative.parts or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("checkpoint runtime state path is not a safe relative path")
    path = (run_dir / relative).resolve()
    if run_dir.resolve() not in path.parents:
        raise ValueError("checkpoint runtime state escapes the run directory")
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint runtime state is missing: {path}")
    actual_hash = sha256(path)
    if actual_hash != descriptor.get("sha256"):
        raise ValueError(
            f"checkpoint runtime state hash mismatch: expected={descriptor.get('sha256')}, "
            f"actual={actual_hash}"
        )
    with path.open("rb") as handle:
        return pickle.load(handle)


def source_hashes(mdp_version: str = "v3") -> dict[str, str]:
    if mdp_version not in {"v3", "v4", "v5", "v6", "v7"}:
        raise ValueError("mdp_version must be one of v3, v4, v5, v6, or v7")
    paths = [
        ROOT / "bus_h2o" / "mdp_contract.py",
        ROOT / "bus_h2o" / "envs" / "all_lines_sim_env.py",
        ROOT / "SimpleSAC" / "transit_v3" / "replay_buffer.py",
        ROOT / "SimpleSAC" / "transit_v3" / "sampler.py",
        ROOT / "SimpleSAC" / "transit_v3" / "networks.py",
        ROOT / "SimpleSAC" / "transit_v3" / "domain_ratio.py",
        ROOT / "SimpleSAC" / "transit_v3" / "protocols.py",
        ROOT / "SimpleSAC" / "transit_v3" / "trainer.py",
        ROOT / "reproducibility" / "build_simulator_demand.py",
        ROOT / "reproducibility" / "build_simulator_runtime_calibration_v4.py",
        Path(__file__).resolve(),
    ]
    if mdp_version == "v7":
        paths.extend(
            [
                ROOT / "bus_h2o" / "mdp_contract_v4.py",
                ROOT / "bus_h2o" / "mdp_contract_v5.py",
                ROOT / "bus_h2o" / "mdp_contract_v7.py",
                ROOT / "bus_h2o" / "passenger_time.py",
                ROOT / "SimpleSAC" / "train_transit_v7.py",
                ROOT / "SimpleSAC" / "transit_v4" / "networks.py",
                ROOT / "SimpleSAC" / "transit_v7" / "networks.py",
                ROOT / "SimpleSAC" / "transit_v7" / "replay_buffer.py",
                ROOT / "SimpleSAC" / "transit_v7" / "sampler.py",
                ROOT / "SimpleSAC" / "transit_v7" / "trainer.py",
                ROOT
                / "reproducibility"
                / "build_simulator_runtime_calibration_v7.py",
                ROOT
                / "reproducibility"
                / "build_simulator_runtime_calibration_v7_v2.py",
                ROOT
                / "reproducibility"
                / "build_simulator_runtime_calibration_v7_v3.py",
                ROOT
                / "reproducibility"
                / "build_simulator_runtime_calibration_v7_v4.py",
                ROOT
                / "reproducibility"
                / "collect_observation_speed_trace_v7.py",
                ROOT
                / "reproducibility"
                / "run_observation_speed_calibration_v7.py",
            ]
        )
    elif mdp_version == "v4":
        paths.extend(
            [
                ROOT / "bus_h2o" / "mdp_contract_v4.py",
                ROOT / "SimpleSAC" / "train_transit_v4.py",
                ROOT / "SimpleSAC" / "transit_v4" / "networks.py",
                ROOT / "SimpleSAC" / "transit_v4" / "replay_buffer.py",
                ROOT / "SimpleSAC" / "transit_v4" / "sampler.py",
                ROOT / "SimpleSAC" / "transit_v4" / "trainer.py",
            ]
        )
    elif mdp_version == "v5":
        paths.extend(
            [
                ROOT / "bus_h2o" / "mdp_contract_v4.py",
                ROOT / "bus_h2o" / "mdp_contract_v5.py",
                ROOT / "bus_h2o" / "passenger_time.py",
                ROOT / "SimpleSAC" / "train_transit_v5.py",
                ROOT / "SimpleSAC" / "transit_v4" / "networks.py",
                ROOT / "SimpleSAC" / "transit_v5" / "networks.py",
                ROOT / "SimpleSAC" / "transit_v5" / "replay_buffer.py",
                ROOT / "SimpleSAC" / "transit_v5" / "sampler.py",
                ROOT / "SimpleSAC" / "transit_v5" / "trainer.py",
            ]
        )
    elif mdp_version == "v6":
        paths.extend(
            [
                ROOT / "bus_h2o" / "mdp_contract_v4.py",
                ROOT / "bus_h2o" / "mdp_contract_v5.py",
                ROOT / "bus_h2o" / "mdp_contract_v6.py",
                ROOT / "bus_h2o" / "passenger_time.py",
                ROOT / "SimpleSAC" / "train_transit_v6.py",
                ROOT / "SimpleSAC" / "transit_v4" / "networks.py",
                ROOT / "SimpleSAC" / "transit_v6" / "networks.py",
                ROOT / "SimpleSAC" / "transit_v6" / "replay_buffer.py",
                ROOT / "SimpleSAC" / "transit_v6" / "sampler.py",
                ROOT / "SimpleSAC" / "transit_v6" / "trainer.py",
            ]
        )
    paths.extend(sorted((ROOT / "bus_h2o" / "sim_core").glob("*.py")))
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(set(paths))
    }


def aggregate_hash(file_hashes: dict[str, str]) -> str:
    """Hash both artifact names and contents to make a stable tree identity."""

    digest = hashlib.sha256()
    for name, file_hash in sorted(file_hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def simulator_input_hashes(
    sim_env: Path,
    *,
    network_path: Path | None = None,
    runtime_calibration_version: str = "v4",
) -> dict[str, Any]:
    """Identify and hash exactly the files selected by ``MultiLineEnv``."""

    sim_env = sim_env.expanduser().resolve()
    data_dir = sim_env / "data"
    config_path = sim_env / "config.json"
    report_path = sim_env / "simulator_demand_calibration.json"
    if runtime_calibration_version not in {"v4", "v7_v2", "v7_v4"}:
        raise ValueError("runtime calibration version must be v4, v7_v2, or v7_v4")
    runtime_calibration_name = (
        f"segment_runtime_calibration_{runtime_calibration_version}.json"
    )
    runtime_calibration_path = sim_env / runtime_calibration_name
    network_path = (
        network_path.expanduser().resolve()
        if network_path is not None
        else (ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml").resolve()
    )
    for required in (
        config_path,
        report_path,
        runtime_calibration_path,
        network_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(f"canonical simulator input is missing: {required}")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"canonical simulator data directory is missing: {data_dir}")

    runtime_paths: dict[str, Path] = {
        "config.json": config_path,
        "repository:bus_h2o/network_data/a_sorted_busline_edge.xml": network_path,
        runtime_calibration_name: runtime_calibration_path,
    }
    line_ids = []
    discovery_files = {
        "stop_news.xlsx",
        "route_news.xlsx",
        "time_table.xlsx",
        "passenger_OD.xlsx",
    }
    for line_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        if not discovery_files.issubset({path.name for path in line_dir.iterdir()}):
            continue
        line_ids.append(line_dir.name)
        selected = {
            "stop_news.xlsx": line_dir / "stop_news.xlsx",
            "route_news.xlsx": line_dir / "route_news.xlsx",
            "passenger_OD.xlsx": (
                line_dir / "passenger_OD_sumo.xlsx"
                if (line_dir / "passenger_OD_sumo.xlsx").is_file()
                else line_dir / "passenger_OD.xlsx"
            ),
            "time_table.xlsx": (
                line_dir / "time_table_sumo.xlsx"
                if (line_dir / "time_table_sumo.xlsx").is_file()
                else line_dir / "time_table.xlsx"
            ),
        }
        for role, path in selected.items():
            if not path.is_file():
                raise FileNotFoundError(
                    f"canonical simulator line {line_dir.name} lacks {role}: {path}"
                )
            runtime_paths[f"data/{line_dir.name}/{role}:{path.name}"] = path
    if not line_ids:
        raise ValueError(f"no runnable simulator lines found under {data_dir}")

    report = json.loads(report_path.read_text())
    report_schema = report.get("schema_version")
    if report_schema not in {
        "transit-simulator-demand-calibration-v1",
        "transit-simulator-demand-calibration-v2",
        "transit-simulator-demand-calibration-v3",
    }:
        raise ValueError("simulator calibration report has an unsupported schema")
    if report_schema == "transit-simulator-demand-calibration-v3":
        journey_execution = report.get("journey_execution", {})
        if journey_execution.get("schema_version") != (
            "transit-simulator-journeys-v1"
        ) or not journey_execution.get("causal_transfer_chain"):
            raise ValueError("v3 simulator calibration lacks causal journey semantics")
        journey_relative = Path(str(journey_execution.get("artifact", "")))
        if (
            not journey_relative.parts
            or journey_relative.is_absolute()
            or ".." in journey_relative.parts
        ):
            raise ValueError("v3 simulator journey artifact path is unsafe")
        journey_path = (report_path.parent / journey_relative).resolve()
        if sim_env != journey_path and sim_env not in journey_path.parents:
            raise ValueError("v3 simulator journey artifact escapes the environment")
        if not journey_path.is_file():
            raise FileNotFoundError(
                f"canonical simulator journey artifact is missing: {journey_path}"
            )
        runtime_paths[journey_relative.as_posix()] = journey_path
    runtime_calibration = json.loads(runtime_calibration_path.read_text())
    expected_runtime_schema = {
        "v4": "transit-simulator-runtime-calibration-v4",
        "v7_v2": "transit-simulator-runtime-calibration-v7-v2",
        "v7_v4": "transit-simulator-runtime-calibration-v7-v4",
    }[runtime_calibration_version]
    if runtime_calibration.get("schema_version") != expected_runtime_schema:
        raise ValueError("simulator runtime calibration has an unsupported schema")
    if runtime_calibration_version in {"v7_v2", "v7_v4"} and not runtime_calibration.get(
        "coverage_gate", {}
    ).get("passed"):
        raise ValueError("v7 simulator runtime calibration failed its coverage gate")
    if runtime_calibration_version == "v7_v4":
        profile = runtime_calibration.get("backward_headway_observation", {}).get(
            "dynamic_speed_profiles", {}
        )
        profile_name = profile.get("artifact")
        if (
            profile.get("schema_version")
            != "transit-simulator-observation-speed-profiles-v7-v1"
            or profile.get("coverage_gate", {}).get("passed") is not True
            or not isinstance(profile_name, str)
            or not profile_name
            or Path(profile_name).name != profile_name
        ):
            raise ValueError("v7-v4 observation-speed profile descriptor is invalid")
        profile_path = (sim_env / profile_name).resolve()
        if profile_path.parent != sim_env or not profile_path.is_file():
            raise FileNotFoundError(
                f"v7-v4 observation-speed profile is missing: {profile_path}"
            )
        if sha256(profile_path) != profile.get("sha256"):
            raise ValueError("v7-v4 observation-speed profile hash changed")
        runtime_paths[profile_name] = profile_path
    declared_outputs = {}
    for raw_path, expected in report.get("output_sha256", {}).items():
        configured = Path(raw_path).expanduser()
        if report_schema in {
            "transit-simulator-demand-calibration-v2",
            "transit-simulator-demand-calibration-v3",
        }:
            if report.get("output_path_base") != "report_directory":
                raise ValueError(
                    "relocatable simulator calibration report has an invalid path base"
                )
            if configured.is_absolute() or ".." in configured.parts:
                raise ValueError(
                    "simulator calibration output is not a safe relative path: "
                    f"{raw_path}"
                )
            artifact = (report_path.parent / configured).resolve()
            if sim_env != artifact and sim_env not in artifact.parents:
                raise ValueError(
                    f"simulator calibration output escapes the environment: {raw_path}"
                )
        else:
            artifact = configured.resolve()
        resolved = str(artifact)
        if resolved in declared_outputs:
            raise ValueError(f"duplicate calibrated simulator output: {resolved}")
        declared_outputs[resolved] = expected
    selected_calibrated_paths = {
        str(path.resolve())
        for path in runtime_paths.values()
        if str(path.resolve()) in declared_outputs
    }
    if selected_calibrated_paths != set(declared_outputs):
        missing = sorted(selected_calibrated_paths - set(declared_outputs))
        extra = sorted(set(declared_outputs) - selected_calibrated_paths)
        raise ValueError(
            "calibration report does not match selected simulator artifacts: "
            f"missing={missing}, extra={extra}"
        )
    for raw_path, expected in declared_outputs.items():
        actual = sha256(Path(raw_path))
        if actual != expected:
            raise ValueError(
                f"calibrated simulator artifact hash mismatch: {raw_path}; "
                f"report={expected}, actual={actual}"
            )

    runtime_hashes = {
        name: sha256(path) for name, path in sorted(runtime_paths.items())
    }
    all_hashes = dict(runtime_hashes)
    all_hashes["simulator_demand_calibration.json"] = sha256(report_path)
    return {
        "schema_version": "transit-simulator-inputs-v1",
        "path": str(sim_env),
        "line_ids": line_ids,
        "runtime_files_sha256": runtime_hashes,
        "runtime_sha256": aggregate_hash(runtime_hashes),
        "calibration_report": {
            "path": str(report_path),
            "sha256": all_hashes["simulator_demand_calibration.json"],
            "source_manifest_sha256": report.get("source_manifest_sha256"),
            "source_bus_objects_sha256": report.get("source_bus_objects_sha256"),
        },
        "aggregate_sha256": aggregate_hash(all_hashes),
    }


RESUME_IMMUTABLE_ARGUMENTS = (
    "method",
    "run_name",
    "run_signature",
    "offline_initializer",
    "offline_initializer_source_signature",
    "seed",
    "max_offline_rows",
    "online_capacity",
    "offline_updates",
    "warmup_events",
    "ratio_pretrain_updates",
    "events_per_epoch",
    "interaction_chunk_events",
    "updates_per_epoch",
    "utd_ratio",
    "batch_size",
    "ensemble_size",
    "hidden_dim",
    "online_fraction",
    "offline_cql_alpha",
    "cql_n_actions",
    "ratio_update_interval",
    "sim_demand_scale",
    "sim_demand_scales",
    "no_device_cache",
    "device_cache_observation_dtype",
    "compile_networks",
    "compile_mode",
)


def validate_resume(
    metadata: dict[str, Any],
    *,
    arguments: dict[str, Any],
    trainer_config: dict[str, Any],
    contract_config: dict[str, Any],
    dataset_sha256: str,
    simulator: dict[str, Any],
    sources: dict[str, str],
    runtime_fingerprint: dict[str, Any],
    checkpoint_extra: dict[str, Any],
    target_epochs: int,
    run_schema_version: str = "transit-training-run-v4",
) -> int:
    """Reject a continuation whenever its scientific protocol has drifted."""

    if metadata.get("schema_version") != run_schema_version:
        raise ValueError(
            f"resume metadata does not implement {run_schema_version}"
        )
    previous_arguments = metadata.get("arguments", {})
    mismatches = {
        name: (previous_arguments.get(name), arguments.get(name))
        for name in RESUME_IMMUTABLE_ARGUMENTS
        if previous_arguments.get(name) != arguments.get(name)
    }
    if mismatches:
        raise ValueError(f"resume arguments changed: {mismatches}")
    if metadata.get("trainer_config") != trainer_config:
        raise ValueError("resume trainer configuration changed")
    if metadata.get("contract_config") != contract_config:
        raise ValueError("resume MDP contract changed")
    if metadata.get("dataset", {}).get("sha256") != dataset_sha256:
        raise ValueError("resume dataset hash changed")
    if metadata.get("source_sha256") != sources:
        raise ValueError("resume source-code hashes changed")
    if metadata.get("simulator", {}).get("aggregate_sha256") != simulator.get(
        "aggregate_sha256"
    ):
        raise ValueError("resume simulator inputs changed")
    stored_runtime = metadata.get("hardware", {}).get("runtime_fingerprint", {})
    if stored_runtime.get("required") != runtime_fingerprint.get("required"):
        raise ValueError(
            "resume required runtime changed: "
            f"stored={stored_runtime.get('required')}, "
            f"current={runtime_fingerprint.get('required')}"
        )

    expected_checkpoint_values = {
        "method": arguments["method"],
        "training_seed": arguments["seed"],
        "run_signature": arguments["run_signature"],
        "dataset_sha256": dataset_sha256,
        "simulator_aggregate_sha256": simulator["aggregate_sha256"],
    }
    checkpoint_mismatches = {
        name: (checkpoint_extra.get(name), expected)
        for name, expected in expected_checkpoint_values.items()
        if checkpoint_extra.get(name) != expected
    }
    if checkpoint_mismatches:
        raise ValueError(f"resume checkpoint provenance changed: {checkpoint_mismatches}")
    completed_epochs = int(checkpoint_extra.get("epoch", -1))
    if completed_epochs < 0:
        raise ValueError("resume checkpoint lacks a valid completed epoch")
    if target_epochs < completed_epochs:
        raise ValueError(
            f"target epochs {target_epochs} precede checkpoint epoch {completed_epochs}"
        )
    if completed_epochs and arguments["method"] in ONLINE_METHODS:
        runtime_state = checkpoint_extra.get("runtime_state")
        if not isinstance(runtime_state, dict) or not {
            "path",
            "sha256",
        }.issubset(runtime_state):
            raise ValueError("online resume checkpoint lacks exact simulator runtime state")
    offline_updates_completed = int(
        checkpoint_extra.get("offline_updates_completed", 0)
    )
    if not 0 <= offline_updates_completed <= int(arguments["offline_updates"]):
        raise ValueError("resume checkpoint has invalid offline-update progress")
    if checkpoint_extra.get("warmup_complete") and arguments["method"] in ONLINE_METHODS:
        runtime_state = checkpoint_extra.get("runtime_state")
        if not isinstance(runtime_state, dict) or not {
            "path",
            "sha256",
        }.issubset(runtime_state):
            raise ValueError("post-warmup checkpoint lacks exact simulator runtime state")
    return completed_epochs


def setup_logging(run_dir: Path, mdp_version: str = "v3") -> logging.Logger:
    logger = logging.getLogger(f"transit_{mdp_version}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(), logging.FileHandler(run_dir / "run.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def run_updates(
    trainer,
    count: int,
    *,
    mode: str,
    log_every: int,
    logger,
) -> dict[str, float]:
    if mode not in {"offline", "mixed", "online-only", "bc"}:
        raise ValueError(f"unknown update mode {mode!r}")
    latest: dict[str, float] = {}
    started = time.perf_counter()
    for step in range(count):
        report = step == count - 1 or (step + 1) % log_every == 0
        metrics = _run_update(trainer, mode=mode, collect_metrics=report)
        if metrics:
            latest = metrics
            elapsed = time.perf_counter() - started
            logger.info(
                "%s update %d/%d rate=%.1f/s critic=%.4f actor=%.4f alpha=%.4f",
                mode,
                step + 1,
                count,
                (step + 1) / max(elapsed, 1e-9),
                metrics.get("train/critic_loss", float("nan")),
                metrics.get("train/actor_loss", float("nan")),
                metrics.get("train/alpha", float("nan")),
            )
    return latest


def _run_update(trainer, *, mode: str, collect_metrics: bool) -> dict[str, float]:
    if mode == "bc":
        return trainer.behavior_clone_step(collect_metrics=collect_metrics)
    return trainer.update(
        online=mode != "offline",
        online_only=mode == "online-only",
        collect_metrics=collect_metrics,
    )


def cumulative_epoch_event_request(
    *,
    warmup_event_cap: int,
    events_per_epoch: int,
    epoch_number: int,
    actual_events_so_far: int,
) -> tuple[int, int]:
    """Return the next request after carrying atomic-batch overshoot forward."""

    cumulative_cap = warmup_event_cap + epoch_number * events_per_epoch
    request = cumulative_cap - actual_events_so_far
    if request <= 0:
        raise RuntimeError(
            "atomic decision-batch overshoot consumed a complete epoch budget"
        )
    return request, cumulative_cap


def run_interleaved_epoch(
    trainer,
    sampler,
    *,
    events: int,
    updates: int,
    chunk_events: int,
    mode: str,
    log_every: int,
    logger,
    timings: dict[str, float],
    monitor: ResourceMonitor,
    torch,
    hard_event_cap: bool = False,
) -> tuple[dict[str, float | int], dict[str, float]]:
    """Alternate collection and learning while preserving exact epoch budgets."""

    if mode not in {"mixed", "online-only"}:
        raise ValueError("interleaved epochs require an online update mode")
    if events <= 0 or updates < 0 or chunk_events <= 0:
        raise ValueError("invalid interleaved epoch budget")

    previous_phase = monitor.phase
    collected_events = 0
    completed_updates = 0
    total_transitions = 0
    total_episodes = 0
    reward_sum = 0.0
    duration_sum = 0.0
    hold_sum = 0.0
    maximum_batch = 0
    discarded_pending_total = 0
    trip_boundaries_total = 0
    blocked_batch_size = 0
    hard_cap_reached = False
    latest_rollout: dict[str, float | int] = {}
    latest_train: dict[str, float] = {}
    training_elapsed = 0.0
    next_request_floor = 0

    def train_to(target_updates: int) -> None:
        nonlocal completed_updates, latest_train, training_elapsed
        if completed_updates >= target_updates:
            return
        monitor.phase = "online_training"
        started = time.perf_counter()
        while completed_updates < target_updates:
            completed_updates += 1
            report = (
                completed_updates == updates
                or completed_updates % log_every == 0
            )
            metrics = _run_update(
                trainer,
                mode=mode,
                collect_metrics=report,
            )
            if metrics:
                latest_train = metrics
                elapsed = training_elapsed + time.perf_counter() - started
                logger.info(
                    "%s update %d/%d rate=%.1f/s critic=%.4f actor=%.4f "
                    "actor_updated=%d alpha=%.4f",
                    mode,
                    completed_updates,
                    updates,
                    completed_updates / max(elapsed, 1e-9),
                    metrics.get("train/critic_loss", float("nan")),
                    metrics.get("train/actor_loss", float("nan")),
                    int(metrics.get("train/actor_updated", 0.0)),
                    metrics.get("train/alpha", float("nan")),
                )
        chunk_training_elapsed = time.perf_counter() - started
        training_elapsed += chunk_training_elapsed
        timings["online_training"] = timings.get(
            "online_training", 0.0
        ) + chunk_training_elapsed

    try:
        while collected_events < events:
            remaining_events = events - collected_events
            request = min(
                remaining_events,
                max(chunk_events, next_request_floor),
            )
            next_request_floor = 0
            monitor.phase = "online_rollout"
            started = time.perf_counter()
            if hard_event_cap:
                rollout = sampler.collect(
                    trainer.act,
                    request,
                    deterministic=False,
                    hard_event_cap=True,
                )
            else:
                rollout = sampler.collect(
                    trainer.act, request, deterministic=False
                )
            timings["online_rollout"] = timings.get("online_rollout", 0.0) + (
                time.perf_counter() - started
            )
            actual_events = int(rollout["events"])
            transitions = int(rollout["transitions"])
            maximum_batch = max(
                maximum_batch,
                int(rollout.get("maximum_simultaneous_decisions", 0)),
            )
            discarded_pending_total += int(
                rollout.get("discarded_pending_total", 0)
            )
            trip_boundaries_total += int(
                rollout.get("trip_boundaries_total", 0)
            )
            call_blocked_batch_size = int(
                rollout.get("blocked_simultaneous_batch_size", 0)
            )
            call_hard_cap_reached = bool(
                rollout.get("hard_event_cap_reached", False)
            )
            remaining_after_call = events - (
                collected_events + actual_events
            )
            global_hard_cap_reached = bool(
                hard_event_cap
                and call_hard_cap_reached
                and call_blocked_batch_size > remaining_after_call
            )
            if (
                hard_event_cap
                and call_hard_cap_reached
                and call_blocked_batch_size
                and not global_hard_cap_reached
            ):
                # The sampler enforces the requested chunk atomically. A
                # simultaneous batch blocked at a chunk boundary may still fit
                # inside the epoch-wide event cap, so retry it in the next
                # interleaving chunk instead of terminating the final epoch.
                next_request_floor = call_blocked_batch_size
            if global_hard_cap_reached:
                blocked_batch_size = max(
                    blocked_batch_size,
                    call_blocked_batch_size,
                )
                hard_cap_reached = True
            latest_rollout = rollout
            if actual_events <= 0:
                if global_hard_cap_reached:
                    break
                if next_request_floor:
                    continue
                raise RuntimeError("online sampler returned no decision events")
            collected_events += actual_events
            total_transitions += transitions
            total_episodes += int(rollout["episodes_started"])
            reward_sum += float(rollout["mean_reward"]) * transitions
            duration_sum += float(rollout["mean_delta_t"]) * transitions
            hold_sum += float(rollout["mean_hold_seconds"]) * actual_events

            capped_events = min(collected_events, events)
            target_updates = round(updates * capped_events / events)
            train_to(target_updates)
            if global_hard_cap_reached:
                break
        train_to(updates)
    finally:
        monitor.phase = previous_phase

    if completed_updates != updates:
        raise RuntimeError(
            f"interleaved epoch completed {completed_updates} of {updates} updates"
        )
    rollout_metrics = {
        "events": collected_events,
        "transitions": total_transitions,
        "episodes_started": total_episodes,
        "discarded_pending_total": discarded_pending_total,
        "trip_boundaries_total": trip_boundaries_total,
        "maximum_simultaneous_decisions": maximum_batch,
        "blocked_simultaneous_batch_size": blocked_batch_size,
        "hard_event_cap_reached": hard_cap_reached,
        "mean_reward": reward_sum / total_transitions if total_transitions else 0.0,
        "mean_delta_t": duration_sum / total_transitions if total_transitions else 0.0,
        "mean_hold_seconds": hold_sum / collected_events if collected_events else 0.0,
        "online_buffer_size": int(latest_rollout["online_buffer_size"]),
    }
    return rollout_metrics, latest_train


def reconcile_metrics(
    path: Path,
    *,
    completed_epochs: int,
    archive_suffix: str,
) -> Path | None:
    """Discard metrics newer than the restored model, preserving them as evidence."""

    if not path.exists():
        if completed_epochs:
            raise ValueError(
                f"resume checkpoint reports epoch {completed_epochs}, but {path} is missing"
            )
        return None
    retained = []
    uncommitted = []
    seen_epochs = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            epoch = int(row["epoch"])
        except Exception as error:
            raise ValueError(f"invalid metrics row {path}:{line_number}") from error
        if epoch <= completed_epochs:
            if epoch in seen_epochs:
                raise ValueError(f"duplicate committed metrics for epoch {epoch}")
            seen_epochs.add(epoch)
            retained.append(line)
        else:
            uncommitted.append(line)
    expected_epochs = set(range(1, completed_epochs + 1))
    if seen_epochs != expected_epochs:
        raise ValueError(
            "committed metrics do not cover the checkpoint epoch range: "
            f"missing={sorted(expected_epochs - seen_epochs)}"
        )
    if not uncommitted:
        return None
    archive = path.with_name(f"metrics.uncommitted_{archive_suffix}.jsonl")
    archive.write_text("\n".join(uncommitted) + "\n")
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(("\n".join(retained) + "\n") if retained else "")
    os.replace(temporary, path)
    return archive


def main(mdp_version: str = "v3") -> None:
    if mdp_version not in {"v3", "v4", "v5", "v6", "v7"}:
        raise ValueError("mdp_version must be one of v3, v4, v5, v6, or v7")
    run_schema_version = {
        "v3": "transit-training-run-v4",
        "v4": "transit-training-run-v5",
        "v5": "transit-training-run-v6",
        "v6": "transit-training-run-v7",
        "v7": "transit-training-run-v8",
    }[mdp_version]
    args = parse_args(mdp_version)
    target_entropy_was_explicit = args.target_entropy is not None
    profile = METHOD_PROFILES[args.method]
    protocol_resolution = resolve_method_arguments(args, profile)
    if mdp_version == "v7":
        if args.reward_scale is None:
            raise ValueError("v7 requires an explicit physical-reward scale")
        if not target_entropy_was_explicit:
            raise ValueError("v7 requires an explicit hybrid-entropy target")
        args.target_entropy = validate_v7_hybrid_target_entropy(
            args.target_entropy
        )
        if args.run_signature and args.training_admission is None:
            raise ValueError("signed v7 runs require --training-admission")
    if any(
        value < 0
        for value in (
            args.epochs,
            args.offline_updates,
            args.warmup_events,
            args.ratio_pretrain_updates,
            args.updates_per_epoch,
        )
    ):
        raise ValueError("training lengths cannot be negative")
    if args.batch_size <= 0 or args.ensemble_size < 2 or args.utd_ratio <= 0:
        raise ValueError("batch size, ensemble size and UTD ratio are invalid")
    if args.events_per_epoch <= 0 or args.interaction_chunk_events <= 0:
        raise ValueError("online event counts must be positive")
    if mdp_version == "v7":
        args.sim_demand_scales = [
            float(value) for value in args.sim_demand_scales
        ]
        if (
            not args.sim_demand_scales
            or not all(
                np.isfinite(value) and value > 0.0
                for value in args.sim_demand_scales
            )
            or len(set(args.sim_demand_scales)) != len(args.sim_demand_scales)
        ):
            raise ValueError(
                "sim-demand-scales must contain unique positive values"
            )
        has_initializer = args.offline_initializer is not None
        has_initializer_signature = (
            args.offline_initializer_source_signature is not None
        )
        if has_initializer != has_initializer_signature:
            raise ValueError(
                "offline initializer and source signature must be provided together"
            )
        if has_initializer:
            if args.method not in V7_SHARED_IQL_BRANCH_METHODS:
                raise ValueError(
                    "offline initializer is restricted to registered H2O+ "
                    "ablation and sensitivity branches"
                )
            if not re.fullmatch(
                r"[0-9a-f]{64}", args.offline_initializer_source_signature
            ):
                raise ValueError(
                    "offline initializer source signature must be a lowercase SHA-256"
                )
    elif args.sim_demand_scale <= 0:
        raise ValueError("sim-demand-scale must be positive")
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+-]*", args.run_name):
        raise ValueError("run-name contains unsafe characters")
    if args.run_signature and not re.fullmatch(r"[0-9a-f]{64}", args.run_signature):
        raise ValueError("run-signature must be a lowercase SHA-256 digest")
    if not 0.0 <= args.online_fraction < 1.0:
        raise ValueError("online_fraction must lie in [0, 1)")
    if (
        args.log_every <= 0
        or args.checkpoint_every <= 0
        or args.offline_checkpoint_every <= 0
    ):
        raise ValueError("logging and checkpoint intervals must be positive")
    args.dataset = args.dataset.expanduser().resolve()
    args.sim_env = args.sim_env.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if mdp_version == "v7" and args.training_admission is not None:
        args.training_admission = args.training_admission.expanduser().resolve()
    if mdp_version == "v7" and args.offline_initializer is not None:
        args.offline_initializer = args.offline_initializer.expanduser().resolve()
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()

    lease = GPULease(
        args.device,
        args.allow_shared_gpu,
        args.allow_external_gpu_processes,
        args.max_canonical_processes_per_gpu,
    )
    try:
        import torch

        from bus_h2o.envs.all_lines_sim_env import AllLinesSimEnv
        from SimpleSAC.transit_v3.trainer import TransitTrainerConfig

        if mdp_version == "v7":
            from bus_h2o.mdp_contract_v7 import TransitMDPConfigV7 as ContractConfig
            from SimpleSAC.transit_v7.replay_buffer import (
                TransitReplayBufferV7 as ReplayBuffer,
            )
            from SimpleSAC.transit_v7.sampler import (
                AllLinesTransitSamplerV7 as TransitSampler,
            )
            from SimpleSAC.transit_v7.trainer import (
                OFFLINE_INITIALIZER_IGNORED_CONFIG_FIELDS,
                TransitH2OTrainerV7 as Trainer,
                inspect_offline_initializer_v7,
            )
        elif mdp_version == "v6":
            from bus_h2o.mdp_contract_v6 import TransitMDPConfigV6 as ContractConfig
            from SimpleSAC.transit_v6.replay_buffer import (
                TransitReplayBufferV6 as ReplayBuffer,
            )
            from SimpleSAC.transit_v6.sampler import (
                AllLinesTransitSamplerV6 as TransitSampler,
            )
            from SimpleSAC.transit_v6.trainer import (
                TransitH2OTrainerV6 as Trainer,
            )
        elif mdp_version == "v5":
            from bus_h2o.mdp_contract_v5 import TransitMDPConfigV5 as ContractConfig
            from SimpleSAC.transit_v5.replay_buffer import (
                TransitReplayBufferV5 as ReplayBuffer,
            )
            from SimpleSAC.transit_v5.sampler import (
                AllLinesTransitSamplerV5 as TransitSampler,
            )
            from SimpleSAC.transit_v5.trainer import (
                TransitH2OTrainerV5 as Trainer,
            )
        elif mdp_version == "v4":
            from bus_h2o.mdp_contract_v4 import TransitMDPConfigV4 as ContractConfig
            from SimpleSAC.transit_v4.replay_buffer import (
                TransitReplayBufferV4 as ReplayBuffer,
            )
            from SimpleSAC.transit_v4.sampler import (
                AllLinesTransitSamplerV4 as TransitSampler,
            )
            from SimpleSAC.transit_v4.trainer import (
                TransitH2OTrainerV4 as Trainer,
            )
        else:
            from bus_h2o.mdp_contract import TransitMDPConfig as ContractConfig
            from SimpleSAC.transit_v3.replay_buffer import (
                TransitReplayBuffer as ReplayBuffer,
            )
            from SimpleSAC.transit_v3.sampler import (
                AllLinesTransitSampler as TransitSampler,
            )
            from SimpleSAC.transit_v3.trainer import TransitH2OTrainer as Trainer

        device = args.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access it")
        torch.set_num_threads(args.cpu_threads)
        torch.set_float32_matmul_precision("high")
        runtime_fingerprint = collect_runtime_fingerprint(torch, device)
        np.random.seed(args.seed)
        random.seed(args.seed)
        uses_offline_pretraining = profile.offline_objective != "none"
        uses_online_interaction = profile.online_update_mode != "none"
        effective_offline_updates = args.offline_updates if uses_offline_pretraining else 0
        effective_epochs = args.epochs if uses_online_interaction else 0

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        resume_checkpoint = args.resume
        if resume_checkpoint is not None:
            if not resume_checkpoint.is_file():
                raise FileNotFoundError(f"resume checkpoint is missing: {resume_checkpoint}")
            if resume_checkpoint.parent.name != "checkpoints":
                raise ValueError("resume checkpoint must be inside run_dir/checkpoints")
            run_dir = resume_checkpoint.parent.parent
            metadata_path = run_dir / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"resume metadata is missing: {metadata_path}")
            metadata = json.loads(metadata_path.read_text())
            resource_path = run_dir / f"resources_resume_{timestamp}.csv"
        else:
            run_dir = (
                args.output_root
                / (
                    args.run_name
                    or f"{args.method}_seed{args.seed}_{timestamp}"
                )
            ).resolve()
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "checkpoints").mkdir()
            metadata_path = run_dir / "metadata.json"
            metadata = {}
            resource_path = run_dir / "resources.csv"
        logger = setup_logging(run_dir, mdp_version)
        monitor = ResourceMonitor(
            resource_path, args.resource_interval, lease.physical_gpu
        )
        monitor.start()
        timings: dict[str, float] = {}
        session_started_utc = datetime.now(timezone.utc).isoformat()

        try:
            with timed_phase("data_load", timings, monitor, torch):
                replay = ReplayBuffer(
                    args.dataset,
                    split="train",
                    device=device,
                    online_capacity=args.online_capacity,
                    max_offline_rows=args.max_offline_rows,
                    subsample_seed=args.seed,
                    device_cache_observation_dtype=(
                        args.device_cache_observation_dtype
                    ),
                )
                contract = ContractConfig(**replay.contract_config)
            with timed_phase("provenance_audit", timings, monitor, torch):
                dataset_sha256 = sha256(args.dataset)
                simulator = simulator_input_hashes(
                    args.sim_env,
                    runtime_calibration_version=(
                        "v7_v4" if mdp_version == "v7" else "v4"
                    ),
                )
                if mdp_version == "v7":
                    simulator["demand_scales"] = list(args.sim_demand_scales)
                    simulator["demand_scale_schedule"] = "cyclic-by-episode"
                else:
                    simulator["demand_scale"] = args.sim_demand_scale
                sources = source_hashes(mdp_version)
                training_admission = (
                    validate_v7_training_admission_runtime(
                        args.training_admission,
                        dataset_sha256=dataset_sha256,
                        simulator=simulator,
                        sources=sources,
                    )
                    if mdp_version == "v7" and args.training_admission is not None
                    else None
                )

            trainer_config = TransitTrainerConfig(
                ensemble_size=args.ensemble_size,
                hidden_dim=args.hidden_dim,
                batch_size=args.batch_size,
                reward_scale=args.reward_scale,
                critic_loss=args.critic_loss,
                huber_delta=args.huber_delta,
                activation=profile.activation,
                actor_layer_norm=profile.actor_layer_norm,
                critic_layer_norm=profile.critic_layer_norm,
                value_layer_norm=profile.value_layer_norm,
                online_fraction=args.online_fraction,
                actor_learning_rate=profile.actor_learning_rate,
                critic_learning_rate=profile.critic_learning_rate,
                value_learning_rate=profile.value_learning_rate,
                alpha_learning_rate=profile.alpha_learning_rate,
                target_subset_size=profile.target_subset_size,
                target_subset_replacement=profile.target_subset_replacement,
                cql_penalty_subset_size=profile.cql_penalty_subset_size,
                cql_penalty_subset_replacement=(
                    profile.cql_penalty_subset_replacement
                ),
                target_entropy=args.target_entropy,
                initial_temperature=args.initial_temperature,
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
                offline_cql_alpha=args.offline_cql_alpha,
                cql_n_actions=args.cql_n_actions,
                cql_max_target_backup=profile.cql_max_target_backup,
                mixed_backup_lambda=profile.mixed_backup_lambda,
                iql_expectile=profile.iql_expectile,
                iql_advantage_temperature=args.iql_advantage_temperature,
                use_ratio_correction=profile.use_ratio_correction,
                ratio_update_interval=args.ratio_update_interval,
                gradient_clip_norm=profile.gradient_clip_norm,
            )
            with timed_phase("model_init", timings, monitor, torch):
                trainer = Trainer(
                    replay, config=trainer_config, device=device, seed=args.seed
                )
                compiled_modules = (
                    trainer.compile_networks(mode=args.compile_mode)
                    if args.compile_networks
                    else []
                )
            current_arguments = {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            }
            trainer_config_dict = asdict(trainer_config)
            contract_config_dict = asdict(contract)
            start_epoch = 0
            checkpoint_extra: dict[str, Any] = {}
            offline_initializer_consumed = None
            offline_initializer_produced = None

            def audit_offline_initializer(
                path: Path,
                *,
                target_method: str,
                expected_source_signature: str,
                require_exact_source_config: bool,
            ) -> tuple[dict[str, Any], dict[str, Any]]:
                if not path.is_file():
                    raise FileNotFoundError(
                        f"v7 offline initializer is missing: {path}"
                    )
                inspected = inspect_offline_initializer_v7(path)
                expected_compatibility = {
                    name: value
                    for name, value in trainer_config_dict.items()
                    if name not in OFFLINE_INITIALIZER_IGNORED_CONFIG_FIELDS
                }
                if inspected["offline_compatibility_config"] != expected_compatibility:
                    raise ValueError(
                        "v7 offline initializer trainer configuration is incompatible"
                    )
                if (
                    require_exact_source_config
                    and inspected["source_config"] != trainer_config_dict
                ):
                    raise ValueError(
                        "produced v7 offline initializer source configuration drifted"
                    )
                descriptor = validate_v7_offline_initializer_runtime(
                    path,
                    inspected["extra"],
                    method=target_method,
                    seed=args.seed,
                    offline_updates=effective_offline_updates,
                    trainer_total_updates=inspected["total_updates"],
                    dataset_sha256=dataset_sha256,
                    simulator=simulator,
                    sources=sources,
                    contract_config=contract_config_dict,
                    training_admission=training_admission,
                    expected_source_signature=expected_source_signature,
                )
                return descriptor, inspected["extra"]

            if resume_checkpoint is not None:
                with timed_phase("checkpoint_restore", timings, monitor, torch):
                    checkpoint_extra = trainer.load(resume_checkpoint)
                start_epoch = validate_resume(
                    metadata,
                    arguments=current_arguments,
                    trainer_config=trainer_config_dict,
                    contract_config=contract_config_dict,
                    dataset_sha256=dataset_sha256,
                    simulator=simulator,
                    sources=sources,
                    runtime_fingerprint=runtime_fingerprint,
                    checkpoint_extra=checkpoint_extra,
                    target_epochs=effective_epochs,
                    run_schema_version=run_schema_version,
                )
                offline_initializer_consumed = checkpoint_extra.get(
                    "offline_initializer_consumed"
                )
                offline_initializer_produced = checkpoint_extra.get(
                    "offline_initializer_produced"
                )
                for field, restored in (
                    ("offline_initializer_consumed", offline_initializer_consumed),
                    ("offline_initializer_produced", offline_initializer_produced),
                ):
                    if metadata.get(field) != restored:
                        raise ValueError(
                            f"resume metadata and checkpoint disagree on {field}"
                        )
                if mdp_version == "v7" and args.offline_initializer is not None:
                    consumed, _ = audit_offline_initializer(
                        args.offline_initializer,
                        target_method=args.method,
                        expected_source_signature=(
                            args.offline_initializer_source_signature
                        ),
                        require_exact_source_config=False,
                    )
                    if consumed != offline_initializer_consumed:
                        raise ValueError(
                            "resume checkpoint is bound to another offline initializer"
                        )
                elif offline_initializer_consumed is not None:
                    raise ValueError(
                        "resume checkpoint unexpectedly consumed an offline initializer"
                    )
                if mdp_version == "v7" and offline_initializer_produced is not None:
                    produced_path = (
                        run_dir / "checkpoints" / "offline_initializer.pt"
                    )
                    produced, _ = audit_offline_initializer(
                        produced_path,
                        target_method="h2oplus",
                        expected_source_signature=args.run_signature,
                        require_exact_source_config=True,
                    )
                    if produced != offline_initializer_produced:
                        raise ValueError(
                            "resume checkpoint is bound to another produced initializer"
                        )
                session_timings = dict(timings)
                timings = {
                    key: float(value)
                    for key, value in checkpoint_extra.get("timings", {}).items()
                }
                for key, value in session_timings.items():
                    timings[key] = timings.get(key, 0.0) + value
                archived_metrics = reconcile_metrics(
                    run_dir / "metrics.jsonl",
                    completed_epochs=start_epoch,
                    archive_suffix=timestamp,
                )
                previous_target = metadata.get("effective_schedule", {}).get(
                    "online_epochs"
                )
                metadata.setdefault("resume_events", []).append(
                    {
                        "started_utc": session_started_utc,
                        "checkpoint": str(resume_checkpoint),
                        "checkpoint_sha256": sha256(resume_checkpoint),
                        "completed_epochs": start_epoch,
                        "previous_target_epochs": previous_target,
                        "new_target_epochs": effective_epochs,
                        "uncommitted_metrics_archive": (
                            str(archived_metrics) if archived_metrics is not None else None
                        ),
                        "arguments": current_arguments,
                        "hardware": {
                            "platform": platform.platform(),
                            "python": sys.version,
                            "torch": torch.__version__,
                            "device": device,
                            "gpu": (
                                torch.cuda.get_device_name(device)
                                if device.startswith("cuda")
                                else None
                            ),
                            "cpu_threads": torch.get_num_threads(),
                            "runtime_fingerprint": runtime_fingerprint,
                        },
                    }
                )
                metadata["effective_schedule"]["online_epochs"] = effective_epochs
                logger.info(
                    "restored checkpoint=%s completed_epochs=%d target_epochs=%d",
                    resume_checkpoint,
                    start_epoch,
                    effective_epochs,
                )
            elif mdp_version == "v7" and args.offline_initializer is not None:
                with timed_phase(
                    "offline_initializer_restore", timings, monitor, torch
                ):
                    initializer_extra = trainer.load_offline_initializer(
                        args.offline_initializer
                    )
                    offline_initializer_consumed, audited_extra = (
                        audit_offline_initializer(
                            args.offline_initializer,
                            target_method=args.method,
                            expected_source_signature=(
                                args.offline_initializer_source_signature
                            ),
                            require_exact_source_config=False,
                        )
                    )
                if initializer_extra != audited_extra:
                    raise RuntimeError(
                        "offline initializer changed during the restore audit"
                    )
                if trainer.total_updates != effective_offline_updates:
                    raise ValueError(
                        "offline initializer does not close the registered update budget"
                    )
                checkpoint_extra = {
                    "offline_updates_completed": effective_offline_updates,
                    "warmup_complete": False,
                    "ratio_pretraining_complete": False,
                    "offline_metrics": initializer_extra["offline_metrics"],
                    "offline_initializer_consumed": offline_initializer_consumed,
                }
                logger.info(
                    "loaded shared IQL initializer=%s source_signature=%s",
                    args.offline_initializer,
                    offline_initializer_consumed["source_run_signature"],
                )

            cache_stats = {}
            if not args.no_device_cache:
                with timed_phase("device_cache", timings, monitor, torch):
                    cache_stats = replay.materialize_device_cache(
                        include_online=bool(replay.online_size)
                    )

            if resume_checkpoint is None:
                metadata = {
                    "schema_version": run_schema_version,
                    "mdp_version": mdp_version,
                    "started_utc": session_started_utc,
                    "method_protocol": METHOD_PROTOCOLS[args.method],
                    "run_signature": args.run_signature or None,
                    "method_profile": asdict(profile),
                    "protocol_resolution": protocol_resolution,
                    "protocol_fidelity": (
                        "profile-default"
                        if not protocol_resolution["overrides"]
                        else "explicit-overrides"
                    ),
                    "effective_schedule": {
                        "offline_updates": effective_offline_updates,
                        "online_epochs": effective_epochs,
                        "events_per_epoch": args.events_per_epoch,
                        "simulator_decision_event_cap": (
                            args.warmup_events
                            + effective_epochs * args.events_per_epoch
                            if uses_online_interaction
                            else 0
                        ),
                        "atomic_simultaneous_batch_policy": (
                            "carry intermediate overshoot forward; do not split or "
                            "execute a final batch that exceeds the hard event cap"
                        ),
                        "interaction_chunk_events": args.interaction_chunk_events,
                        "critic_updates_per_epoch": args.updates_per_epoch,
                        "actor_updates_per_epoch": (
                            args.updates_per_epoch
                            // profile.actor_update_interval
                            if uses_online_interaction
                            else 0
                        ),
                        "utd_ratio": args.utd_ratio,
                        "update_mode": profile.online_update_mode,
                    },
                    "arguments": current_arguments,
                    "trainer_config": trainer_config_dict,
                    "contract_config": contract_config_dict,
                    "dataset": {
                        "path": str(args.dataset),
                        "sha256": dataset_sha256,
                        "offline_rows": replay.offline_size,
                    },
                    "simulator": simulator,
                    "source_sha256": sources,
                    "training_admission": training_admission,
                    "offline_initializer_consumed": offline_initializer_consumed,
                    "offline_initializer_produced": offline_initializer_produced,
                    "git": _git_provenance(),
                    "hardware": {
                        "platform": platform.platform(),
                        "python": sys.version,
                        "torch": torch.__version__,
                        "device": device,
                        "gpu": (
                            torch.cuda.get_device_name(device)
                            if device.startswith("cuda")
                            else None
                        ),
                        "cpu_threads": torch.get_num_threads(),
                        "runtime_fingerprint": runtime_fingerprint,
                        "device_cache": cache_stats,
                        "optimizer_config": trainer.optimizer_config,
                        "compiled_modules": compiled_modules,
                        "compiler_config": trainer.compiler_config,
                    },
                }
            write_json(metadata_path, metadata)
            logger.info(
                "run=%s method=%s device=%s offline_rows=%d simulator=%s",
                run_dir,
                args.method,
                device,
                replay.offline_size,
                simulator["aggregate_sha256"],
            )

            phase_metrics_path = run_dir / "phase_metrics.json"
            if phase_metrics_path.is_file():
                phase_metrics = json.loads(phase_metrics_path.read_text())
            else:
                phase_metrics = {}
            sampler = None
            offline_updates_completed = int(
                checkpoint_extra.get("offline_updates_completed", 0)
            )
            warmup_complete = bool(checkpoint_extra.get("warmup_complete", False))
            ratio_pretraining_complete = bool(
                checkpoint_extra.get("ratio_pretraining_complete", False)
            )
            offline_metrics = dict(
                phase_metrics.get(
                    "offline", checkpoint_extra.get("offline_metrics", {})
                )
            )
            if offline_updates_completed:
                phase_metrics["offline"] = offline_metrics
                phase_metrics["offline_updates_completed"] = (
                    offline_updates_completed
                )
                if offline_initializer_consumed is not None:
                    phase_metrics["offline_initializer_consumed"] = (
                        offline_initializer_consumed
                    )
                write_json(phase_metrics_path, phase_metrics)

            def save_training_checkpoint(epoch: int, *, phase: str) -> None:
                runtime_descriptor = None
                runtime_path = None
                if sampler is not None:
                    runtime_path = (
                        run_dir
                        / "checkpoints"
                        / f"runtime_epoch{epoch:06d}.pkl"
                    )
                    runtime_state = sampler.runtime_state_dict()
                    runtime_hash = write_pickle(runtime_path, runtime_state)
                    runtime_descriptor = {
                        "schema_version": runtime_state["schema_version"],
                        "path": str(runtime_path.relative_to(run_dir)),
                        "sha256": runtime_hash,
                        "bytes": runtime_path.stat().st_size,
                    }
                trainer.save(
                    run_dir / "checkpoints" / "latest.pt",
                    extra={
                        "epoch": epoch,
                        "phase": phase,
                        "run_signature": args.run_signature,
                        "offline_updates_completed": offline_updates_completed,
                        "offline_metrics": offline_metrics,
                        "warmup_complete": warmup_complete,
                        "ratio_pretraining_complete": ratio_pretraining_complete,
                        "offline_initializer_consumed": (
                            offline_initializer_consumed
                        ),
                        "offline_initializer_produced": (
                            offline_initializer_produced
                        ),
                        "method": args.method,
                        "method_profile": asdict(profile),
                        "training_seed": args.seed,
                        "run_dir": str(run_dir),
                        "dataset_sha256": dataset_sha256,
                        "simulator_runtime_sha256": simulator["runtime_sha256"],
                        "simulator_aggregate_sha256": simulator[
                            "aggregate_sha256"
                        ],
                        "source_sha256": sources,
                        "contract_config": contract_config_dict,
                        "mdp_version": mdp_version,
                        "trainer_config": trainer_config_dict,
                        "runtime_state": runtime_descriptor,
                        "timings": dict(timings),
                    },
                )
                if runtime_path is not None:
                    for obsolete in (run_dir / "checkpoints").glob(
                        "runtime_epoch*.pkl"
                    ):
                        if obsolete != runtime_path:
                            obsolete.unlink()

            if resume_checkpoint is None:
                with timed_phase("checkpoint", timings, monitor, torch):
                    save_training_checkpoint(
                        0,
                        phase=(
                            "offline_initializer_loaded"
                            if offline_initializer_consumed is not None
                            else "initialized"
                        ),
                    )

            if offline_updates_completed > effective_offline_updates:
                raise ValueError("checkpoint exceeds the configured offline-update budget")
            while offline_updates_completed < effective_offline_updates:
                update_count = min(
                    args.offline_checkpoint_every,
                    effective_offline_updates - offline_updates_completed,
                )
                with timed_phase("offline_pretraining", timings, monitor, torch):
                    offline_metrics = run_updates(
                        trainer,
                        update_count,
                        mode="bc" if profile.offline_objective == "bc" else "offline",
                        log_every=args.log_every,
                        logger=logger,
                    )
                offline_updates_completed += update_count
                phase_metrics["offline"] = offline_metrics
                phase_metrics["offline_updates_completed"] = offline_updates_completed
                write_json(phase_metrics_path, phase_metrics)
                with timed_phase("checkpoint", timings, monitor, torch):
                    save_training_checkpoint(0, phase="offline_pretraining")

            produces_shared_initializer = (
                mdp_version == "v7"
                and args.method == "h2oplus"
                and args.run_name == f"h2oplus_seed{args.seed}"
                and effective_offline_updates == 50_000
                and bool(args.run_signature)
                and training_admission is not None
            )
            if produces_shared_initializer and offline_initializer_produced is None:
                if trainer.online_critic_updates or replay.online_size:
                    raise RuntimeError(
                        "cannot recover a missing offline initializer after online state"
                    )
                if trainer.total_updates != effective_offline_updates:
                    raise RuntimeError(
                        "shared initializer must be frozen at the exact IQL budget"
                    )
                initializer_path = (
                    run_dir / "checkpoints" / "offline_initializer.pt"
                )
                initializer_extra = {
                    "phase": "offline_pretraining",
                    "source_method": "h2oplus",
                    "source_run_name": args.run_name,
                    "source_run_signature": args.run_signature,
                    "source_training_seed": args.seed,
                    "offline_updates_completed": offline_updates_completed,
                    "trainer_total_updates": trainer.total_updates,
                    "dataset_sha256": dataset_sha256,
                    "simulator_runtime_sha256": simulator["runtime_sha256"],
                    "simulator_aggregate_sha256": simulator[
                        "aggregate_sha256"
                    ],
                    "source_sha256": sources,
                    "contract_config": contract_config_dict,
                    "mdp_version": mdp_version,
                    "training_admission": training_admission,
                    "offline_metrics": offline_metrics,
                }
                if not initializer_path.exists():
                    with timed_phase(
                        "offline_initializer_save", timings, monitor, torch
                    ):
                        trainer.save_offline_initializer(
                            initializer_path, extra=initializer_extra
                        )
                offline_initializer_produced, audited_extra = (
                    audit_offline_initializer(
                        initializer_path,
                        target_method="h2oplus",
                        expected_source_signature=args.run_signature,
                        require_exact_source_config=True,
                    )
                )
                if audited_extra != initializer_extra:
                    raise RuntimeError(
                        "existing offline initializer disagrees with its source run"
                    )
                metadata["offline_initializer_produced"] = (
                    offline_initializer_produced
                )
                phase_metrics["offline_initializer_produced"] = (
                    offline_initializer_produced
                )
                write_json(metadata_path, metadata)
                write_json(phase_metrics_path, phase_metrics)
                with timed_phase("checkpoint", timings, monitor, torch):
                    save_training_checkpoint(0, phase="offline_initializer_frozen")

            warmup_metrics = dict(phase_metrics.get("warmup", {}))
            online_work_remaining = uses_online_interaction and start_epoch < effective_epochs
            if online_work_remaining:
                python_rng_state = random.getstate()
                numpy_rng_state = np.random.get_state()
                torch_rng_state = torch.get_rng_state()
                cuda_rng_state = (
                    torch.cuda.get_rng_state(device)
                    if device.startswith("cuda")
                    else None
                )
                try:
                    with timed_phase("simulator_init", timings, monitor, torch):
                        env = AllLinesSimEnv(
                            str(args.sim_env),
                            episode_seconds=(
                                contract.horizon_seconds
                                if mdp_version == "v7"
                                else contract.episode_seconds
                            ),
                            demand_scale=(
                                args.sim_demand_scales[0]
                                if mdp_version == "v7"
                                else args.sim_demand_scale
                            ),
                            observation_schema=mdp_version,
                            observation_speed_seed=(
                                args.seed if mdp_version == "v7" else None
                            ),
                        )
                        sampler = TransitSampler(
                            env,
                            replay,
                            station_max_by_line=replay.station_max_by_line,
                            contract_config=contract,
                            **(
                                {
                                    "demand_scale_schedule": tuple(
                                        args.sim_demand_scales
                                    )
                                }
                                if mdp_version == "v7"
                                else {}
                            ),
                        )
                        if checkpoint_extra.get("runtime_state") is not None:
                            runtime_state = read_runtime_state(
                                run_dir, checkpoint_extra["runtime_state"]
                            )
                            sampler.load_runtime_state_dict(runtime_state)
                finally:
                    random.setstate(python_rng_state)
                    np.random.set_state(numpy_rng_state)
                    torch.set_rng_state(torch_rng_state.cpu())
                    if cuda_rng_state is not None and device.startswith("cuda"):
                        torch.cuda.set_rng_state(cuda_rng_state.cpu(), device)
            if (
                online_work_remaining
                and args.warmup_events
                and not warmup_complete
            ):
                assert sampler is not None
                warmup_policy = make_warmup_policy(
                    profile, trainer.act, seed=args.seed
                )
                with timed_phase("online_warmup_rollout", timings, monitor, torch):
                    if mdp_version in {"v6", "v7"} and effective_epochs == 0:
                        warmup_metrics = sampler.collect(
                            warmup_policy,
                            args.warmup_events,
                            deterministic=False,
                            hard_event_cap=True,
                        )
                    else:
                        warmup_metrics = sampler.collect(
                            warmup_policy,
                            args.warmup_events,
                            deterministic=False,
                        )
                warmup_metrics["action_mode"] = profile.warmup_action_mode
                logger.info("warmup rollout %s", warmup_metrics)
                warmup_complete = True
                phase_metrics["warmup"] = warmup_metrics
                phase_metrics["warmup_complete"] = True
                write_json(phase_metrics_path, phase_metrics)
                with timed_phase("checkpoint", timings, monitor, torch):
                    save_training_checkpoint(0, phase="online_warmup")
            elif not args.warmup_events:
                warmup_complete = True
            if not args.no_device_cache and online_work_remaining and replay.online_size:
                with timed_phase("device_cache", timings, monitor, torch):
                    cache_stats = replay.materialize_device_cache()

            ratio_metrics = dict(phase_metrics.get("ratio", {}))
            if (
                online_work_remaining
                and trainer_config.use_ratio_correction
                and args.ratio_pretrain_updates
                and not ratio_pretraining_complete
            ):
                with timed_phase("ratio_pretraining", timings, monitor, torch):
                    ratio_metrics = trainer.pretrain_ratio(args.ratio_pretrain_updates)
                logger.info("ratio pretraining %s", ratio_metrics)
                ratio_pretraining_complete = True
                phase_metrics["ratio"] = ratio_metrics
                phase_metrics["ratio_pretraining_complete"] = True
                write_json(phase_metrics_path, phase_metrics)
                with timed_phase("checkpoint", timings, monitor, torch):
                    save_training_checkpoint(0, phase="ratio_pretraining")
            elif not trainer_config.use_ratio_correction or not args.ratio_pretrain_updates:
                ratio_pretraining_complete = True

            metrics_path = run_dir / "metrics.jsonl"
            latest_ratio_validation = dict(ratio_metrics)
            committed_epoch_metrics = (
                [
                    json.loads(line)
                    for line in metrics_path.read_text().splitlines()
                    if line.strip()
                ]
                if metrics_path.is_file()
                else []
            )
            if len(committed_epoch_metrics) != start_epoch:
                raise RuntimeError(
                    "committed epoch metrics disagree with the restored epoch"
                )
            actual_simulator_decision_events = int(
                warmup_metrics.get("events", 0)
            ) + sum(
                int(row.get("rollout", {}).get("events", 0))
                for row in committed_epoch_metrics
            )
            maximum_simultaneous_decisions = max(
                [int(warmup_metrics.get("maximum_simultaneous_decisions", 0))]
                + [
                    int(
                        row.get("rollout", {}).get(
                            "maximum_simultaneous_decisions", 0
                        )
                    )
                    for row in committed_epoch_metrics
                ]
            )
            final_blocked_batch_size = (
                int(
                    committed_epoch_metrics[-1]
                    .get("rollout", {})
                    .get("blocked_simultaneous_batch_size", 0)
                )
                if committed_epoch_metrics
                else int(
                    warmup_metrics.get("blocked_simultaneous_batch_size", 0)
                )
            )
            final_hard_cap_reached = (
                bool(
                    committed_epoch_metrics[-1]
                    .get("rollout", {})
                    .get("hard_event_cap_reached", False)
                )
                if committed_epoch_metrics
                else bool(warmup_metrics.get("hard_event_cap_reached", False))
            )
            for epoch in range(start_epoch, effective_epochs):
                assert sampler is not None
                epoch_event_request, cumulative_event_cap = (
                    cumulative_epoch_event_request(
                        warmup_event_cap=args.warmup_events,
                        events_per_epoch=args.events_per_epoch,
                        epoch_number=epoch + 1,
                        actual_events_so_far=actual_simulator_decision_events,
                    )
                )
                final_hard_event_cap = (
                    mdp_version in {"v6", "v7"} and epoch + 1 == effective_epochs
                )
                rollout_before = timings.get("online_rollout", 0.0)
                training_before = timings.get("online_training", 0.0)
                rollout_metrics, train_metrics = run_interleaved_epoch(
                    trainer,
                    sampler,
                    events=epoch_event_request,
                    updates=args.updates_per_epoch,
                    chunk_events=args.interaction_chunk_events,
                    mode=profile.online_update_mode,
                    log_every=args.log_every,
                    logger=logger,
                    timings=timings,
                    monitor=monitor,
                    torch=torch,
                    hard_event_cap=final_hard_event_cap,
                )
                actual_simulator_decision_events += int(
                    rollout_metrics["events"]
                )
                maximum_simultaneous_decisions = max(
                    maximum_simultaneous_decisions,
                    int(
                        rollout_metrics.get(
                            "maximum_simultaneous_decisions", 0
                        )
                    ),
                )
                final_blocked_batch_size = int(
                    rollout_metrics.get("blocked_simultaneous_batch_size", 0)
                )
                final_hard_cap_reached = bool(
                    rollout_metrics.get("hard_event_cap_reached", False)
                )
                if (
                    not final_hard_event_cap
                    and actual_simulator_decision_events < cumulative_event_cap
                ):
                    raise RuntimeError(
                        "online collection failed to meet its cumulative event target"
                    )
                if (
                    final_hard_event_cap
                    and actual_simulator_decision_events > cumulative_event_cap
                ):
                    raise RuntimeError("online collection exceeded its hard event cap")
                if trainer_config.use_ratio_correction:
                    latest_ratio_validation = trainer.evaluate_ratio()
                if cache_stats:
                    cache_stats["online_rows"] = replay.online_size

                epoch_metrics = {
                    "epoch": epoch + 1,
                    "total_updates": trainer.total_updates,
                    "online_buffer_size": replay.online_size,
                    "requested_simulator_decision_events": epoch_event_request,
                    "cumulative_simulator_decision_event_cap": cumulative_event_cap,
                    "actual_cumulative_simulator_decision_events": (
                        actual_simulator_decision_events
                    ),
                    "rollout": rollout_metrics,
                    "train": train_metrics,
                    "ratio_validation": latest_ratio_validation,
                    "timings_cumulative_sec": dict(timings),
                    "device_cache": cache_stats,
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(epoch_metrics, sort_keys=True) + "\n")
                write_json(run_dir / "timings.json", timings)
                logger.info(
                    "epoch=%d/%d reward=%.5f online=%d rollout=%.2fs train=%.2fs",
                    epoch + 1,
                    effective_epochs,
                    rollout_metrics["mean_reward"],
                    replay.online_size,
                    timings.get("online_rollout", 0.0) - rollout_before,
                    timings.get("online_training", 0.0) - training_before,
                )
                if (epoch + 1) % args.checkpoint_every == 0:
                    with timed_phase("checkpoint", timings, monitor, torch):
                        save_training_checkpoint(epoch + 1, phase="online_training")

            simulator_decision_event_cap = (
                args.warmup_events + effective_epochs * args.events_per_epoch
                if uses_online_interaction
                else 0
            )
            simulator_decision_event_underfill = (
                simulator_decision_event_cap - actual_simulator_decision_events
            )
            if mdp_version in {"v6", "v7"} and simulator_decision_event_underfill < 0:
                raise RuntimeError(
                    f"{mdp_version} training exceeded its simulator event cap"
                )
            if mdp_version in {"v6", "v7"} and simulator_decision_event_underfill:
                if (
                    not final_hard_cap_reached
                    or final_blocked_batch_size
                    <= simulator_decision_event_underfill
                ):
                    raise RuntimeError(
                        f"{mdp_version} event-cap underfill lacks a larger "
                        "blocked atomic batch"
                    )

            if resume_checkpoint is None or start_epoch < effective_epochs:
                with timed_phase("checkpoint", timings, monitor, torch):
                    save_training_checkpoint(effective_epochs, phase="complete")
            write_json(run_dir / "timings.json", timings)
            latest_checkpoint = run_dir / "checkpoints" / "latest.pt"
            latest_checkpoint_sha256 = sha256(latest_checkpoint)
            final_policy = None
            if mdp_version == "v7":
                final_policy_path = run_dir / "checkpoints" / "final_actor.pt"
                policy_extra = {
                    "epoch": effective_epochs,
                    "phase": "complete",
                    "run_signature": args.run_signature,
                    "method": args.method,
                    "method_profile": asdict(profile),
                    "training_seed": args.seed,
                    "run_dir": str(run_dir),
                    "dataset_sha256": dataset_sha256,
                    "simulator_runtime_sha256": simulator["runtime_sha256"],
                    "simulator_aggregate_sha256": simulator["aggregate_sha256"],
                    "source_sha256": sources,
                    "contract_config": contract_config_dict,
                    "mdp_version": mdp_version,
                    "trainer_config": trainer_config_dict,
                    "total_updates": trainer.total_updates,
                    "simulator_decision_events": actual_simulator_decision_events,
                    "training_checkpoint_sha256": latest_checkpoint_sha256,
                    "training_admission": training_admission,
                    "offline_initializer_consumed": (
                        offline_initializer_consumed
                    ),
                    "offline_initializer_produced": (
                        offline_initializer_produced
                    ),
                }
                trainer.save_policy_artifact(final_policy_path, extra=policy_extra)
                final_policy = {
                    "path": str(final_policy_path),
                    "sha256": sha256(final_policy_path),
                    "source_training_checkpoint_sha256": (
                        latest_checkpoint_sha256
                    ),
                }
            summary = {
                "schema_version": {
                    "v3": "transit-training-summary-v4",
                    "v4": "transit-training-summary-v5",
                    "v5": "transit-training-summary-v6",
                    "v6": "transit-training-summary-v7",
                    "v7": "transit-training-summary-v8",
                }[mdp_version],
                "run_dir": str(run_dir),
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "method": args.method,
                "run_signature": args.run_signature or None,
                "mdp_version": mdp_version,
                "method_protocol": METHOD_PROTOCOLS[args.method],
                "method_profile": asdict(profile),
                "protocol_resolution": protocol_resolution,
                "seed": args.seed,
                "start_epoch": start_epoch,
                "completed_epochs": effective_epochs,
                "offline_metrics": offline_metrics,
                "offline_initializer_consumed": offline_initializer_consumed,
                "offline_initializer_produced": offline_initializer_produced,
                "warmup_metrics": warmup_metrics,
                "ratio_metrics": ratio_metrics,
                "final_ratio_validation": latest_ratio_validation,
                "simulator_decision_event_cap": simulator_decision_event_cap,
                "simulator_decision_events": actual_simulator_decision_events,
                "simulator_decision_event_underfill": (
                    simulator_decision_event_underfill
                ),
                "maximum_simultaneous_decisions": (
                    maximum_simultaneous_decisions
                ),
                "final_blocked_simultaneous_batch_size": (
                    final_blocked_batch_size
                ),
                "online_buffer_size": replay.online_size,
                "total_updates": trainer.total_updates,
                "checkpoint": {
                    "path": str(latest_checkpoint),
                    "sha256": latest_checkpoint_sha256,
                },
                "final_policy": final_policy,
                "timings_sec": timings,
                "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
                "max_cuda_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / 2**20
                    if device.startswith("cuda")
                    else 0.0
                ),
            }
            write_json(run_dir / "summary.json", summary)
            logger.info("completed %s", summary)
            print(run_dir)
        finally:
            monitor.stop()
    finally:
        lease.close()


if __name__ == "__main__":
    main()
