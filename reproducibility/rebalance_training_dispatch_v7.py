#!/usr/bin/env python3
"""Rebalance a frozen training plan across GPU lanes using observed runtimes.

This tool changes only the execution partition. It preserves the signed plan,
job signatures, and one-process-per-GPU contract. Existing dispatches are never
overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.run_training_dispatch_v6 import load_dispatch


PROFILE_SCHEMA = "transit-training-runtime-profile-v1"
ALGORITHM = "unrelated-machine-greedy-earliest-finish-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_profile(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError(f"runtime profile must implement {PROFILE_SCHEMA}")
    if payload.get("unit") != "seconds":
        raise ValueError("runtime profile unit must be seconds")
    hosts = payload.get("hosts")
    if not isinstance(hosts, dict) or not hosts:
        raise ValueError("runtime profile hosts must be a non-empty object")
    for hostname, host_profile in hosts.items():
        if not isinstance(hostname, str) or not hostname:
            raise ValueError("runtime profile contains an invalid hostname")
        if not isinstance(host_profile, dict):
            raise ValueError(f"runtime profile for {hostname!r} must be an object")
        methods = host_profile.get("methods", {})
        jobs = host_profile.get("jobs", {})
        if not isinstance(methods, dict) or not isinstance(jobs, dict):
            raise ValueError(f"runtime profile for {hostname!r} has invalid estimates")
        if not methods and not jobs:
            raise ValueError(f"runtime profile for {hostname!r} has no estimates")
        for label, estimates in (("methods", methods), ("jobs", jobs)):
            for key, value in estimates.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"runtime profile {hostname!r}.{label} has invalid key")
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(
                        f"runtime profile {hostname!r}.{label}.{key} is not numeric"
                    )
                if not math.isfinite(float(value)) or float(value) <= 0:
                    raise ValueError(
                        f"runtime profile {hostname!r}.{label}.{key} must be positive"
                    )
    payload["path"] = str(path)
    payload["sha256"] = _sha256(path)
    return payload


def _estimate_seconds(
    profile: dict[str, Any], hostname: str, job: dict[str, Any]
) -> float:
    try:
        host_profile = profile["hosts"][hostname]
    except KeyError as exc:
        raise ValueError(f"runtime profile lacks host {hostname!r}") from exc
    job_estimates = host_profile.get("jobs", {})
    method_estimates = host_profile.get("methods", {})
    if job["name"] in job_estimates:
        return float(job_estimates[job["name"]])
    if job["method"] in method_estimates:
        return float(method_estimates[job["method"]])
    raise ValueError(
        f"runtime profile lacks an estimate for {job['name']!r} on {hostname!r}"
    )


def _dispatch_lanes(
    dispatch: dict[str, Any],
) -> tuple[list[tuple[str, str]], dict[str, str], dict[tuple[str, str], list[str]]]:
    lanes: list[tuple[str, str]] = []
    original_host: dict[str, str] = {}
    original_lanes: dict[tuple[str, str], list[str]] = {}
    for host in dispatch["hosts"]:
        hostname = host["hostname"]
        for gpu in host["gpus"]:
            lane = (hostname, gpu["physical_gpu"])
            lanes.append(lane)
            names = [
                name
                for stage in gpu["stages"]
                for worker in stage["workers"]
                for name in worker["jobs"]
            ]
            original_lanes[lane] = names
            for name in names:
                original_host[name] = hostname
    return lanes, original_host, original_lanes


def schedule_jobs(
    *,
    jobs: Iterable[dict[str, Any]],
    lanes: list[tuple[str, str]],
    profile: dict[str, Any],
    original_host: dict[str, str],
    allow_cross_host: bool,
) -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str], float]]:
    """Assign longest jobs first to the lane with the earliest predicted finish."""

    jobs = list(jobs)
    eligible_lanes: dict[str, list[tuple[str, str]]] = {}
    for job in jobs:
        eligible = (
            list(lanes)
            if allow_cross_host
            else [lane for lane in lanes if lane[0] == original_host[job["name"]]]
        )
        if not eligible:
            raise ValueError(f"job {job['name']!r} has no eligible GPU lane")
        for hostname, _ in eligible:
            _estimate_seconds(profile, hostname, job)
        eligible_lanes[job["name"]] = eligible

    ordered = sorted(
        jobs,
        key=lambda job: (
            -max(
                _estimate_seconds(profile, hostname, job)
                for hostname, _ in eligible_lanes[job["name"]]
            ),
            job["name"],
        ),
    )
    assignments = {lane: [] for lane in lanes}
    loads = {lane: 0.0 for lane in lanes}
    lane_order = {lane: index for index, lane in enumerate(lanes)}
    for job in ordered:
        lane = min(
            eligible_lanes[job["name"]],
            key=lambda candidate: (
                loads[candidate]
                + _estimate_seconds(profile, candidate[0], job),
                lane_order[candidate],
            ),
        )
        assignments[lane].append(job["name"])
        loads[lane] += _estimate_seconds(profile, lane[0], job)
    return assignments, loads


def _estimated_original_loads(
    *,
    jobs_by_name: dict[str, dict[str, Any]],
    original_lanes: dict[tuple[str, str], list[str]],
    profile: dict[str, Any],
) -> dict[tuple[str, str], float]:
    return {
        lane: sum(
            _estimate_seconds(profile, lane[0], jobs_by_name[name]) for name in names
        )
        for lane, names in original_lanes.items()
    }


def _serialized_loads(loads: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    return [
        {
            "hostname": hostname,
            "physical_gpu": physical_gpu,
            "estimated_seconds": round(seconds, 6),
        }
        for (hostname, physical_gpu), seconds in loads.items()
    ]


def build_rebalanced_dispatch(
    *,
    dispatch_path: Path,
    profile_path: Path,
    output_path: Path,
    allow_cross_host: bool,
) -> dict[str, Any]:
    dispatch_path = dispatch_path.expanduser().resolve(strict=True)
    output_path = output_path.expanduser().resolve()
    dispatch, jobs, evidence = load_dispatch(dispatch_path)
    profile = _load_profile(profile_path)
    lanes, original_host, original_lanes = _dispatch_lanes(dispatch)
    jobs_by_name = {job["name"]: job for job in jobs}
    assignments, balanced_loads = schedule_jobs(
        jobs=jobs,
        lanes=lanes,
        profile=profile,
        original_host=original_host,
        allow_cross_host=allow_cross_host,
    )
    original_loads = _estimated_original_loads(
        jobs_by_name=jobs_by_name,
        original_lanes=original_lanes,
        profile=profile,
    )

    hosts = []
    source_hosts = {host["hostname"]: host for host in dispatch["hosts"]}
    for hostname, source_host in source_hosts.items():
        gpus = []
        for source_gpu in source_host["gpus"]:
            physical_gpu = source_gpu["physical_gpu"]
            names = assignments[(hostname, physical_gpu)]
            if not names:
                continue
            gpus.append(
                {
                    "physical_gpu": physical_gpu,
                    "stages": [
                        {
                            "name": "runtime-balanced-serial",
                            "workers": [
                                {
                                    "name": (
                                        f"{hostname}-gpu{physical_gpu}-runtime-balanced"
                                    ),
                                    "jobs": names,
                                }
                            ],
                        }
                    ],
                }
            )
        if gpus:
            hosts.append(
                {
                    "hostname": hostname,
                    "hostname_aliases": list(source_host.get("hostname_aliases", [])),
                    "gpus": gpus,
                }
            )

    plan_path = Path(dispatch["plan_path"])
    source_dispatch_reference = os.path.relpath(dispatch_path, output_path.parent)
    profile_reference = os.path.relpath(Path(profile["path"]), output_path.parent)
    plan_reference = os.path.relpath(plan_path, output_path.parent)
    original_makespan = max(original_loads.values())
    balanced_makespan = max(balanced_loads.values())
    payload = {
        "schema_version": dispatch["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "runtime-balanced-before-launch",
        "plan_id": dispatch["plan_id"],
        "plan": {"path": plan_reference, "sha256": evidence["plan_sha256"]},
        "job_set_sha256": dispatch["job_set_sha256"],
        "policy": {
            **dispatch.get("policy", {}),
            "runtime_balancing": (
                "execution-only longest-processing-time sharding; signed plan, "
                "job signatures, and one process per physical GPU are unchanged"
            ),
        },
        "runtime_balancing": {
            "algorithm": ALGORITHM,
            "allow_cross_host": allow_cross_host,
            "source_dispatch": {
                "path": source_dispatch_reference,
                "sha256": dispatch["sha256"],
            },
            "runtime_profile": {
                "path": profile_reference,
                "sha256": profile["sha256"],
            },
            "original_estimated_makespan_seconds": round(original_makespan, 6),
            "balanced_estimated_makespan_seconds": round(balanced_makespan, 6),
            "estimated_reduction_fraction": round(
                1.0 - balanced_makespan / original_makespan, 9
            ),
            "original_lane_loads": _serialized_loads(original_loads),
            "balanced_lane_loads": _serialized_loads(balanced_loads),
        },
        "hosts": hosts,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dispatch", type=Path, required=True)
    parser.add_argument("--runtime-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-cross-host",
        action="store_true",
        help="allow jobs to move between hosts; otherwise only rebalance GPUs per host",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing dispatch {output}")
    payload = build_rebalanced_dispatch(
        dispatch_path=args.dispatch,
        profile_path=args.runtime_profile,
        output_path=output,
        allow_cross_host=args.allow_cross_host,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    load_dispatch(output)
    print(
        json.dumps(
            {
                "dispatch": str(output),
                "sha256": _sha256(output),
                **payload["runtime_balancing"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
