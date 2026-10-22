#!/usr/bin/env python3
"""Freeze two-server dispatches for formal v7 plans."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.build_training_dispatch_v6 import (
    BASELINE_METHOD_ORDER,
    HOSTNAME_ALIASES,
    HOST_GPUS,
    _method_seed_index,
)
from reproducibility.build_training_plan_v7 import (
    ABLATION_METHODS,
    FORMAL_SEEDS,
    FORMAL_SERVER_GPU_CONCURRENCY,
    H2OPLUS_CORE_METHODS,
    H2OPLUS_SENSITIVITY_METHODS,
    PLAN_SCHEMA,
    SERVER_CORE_METHODS,
    SERVER_H2OPLUS_EXECUTION_CLASS,
    SERVER_SENSITIVITY_METHODS,
)
from reproducibility.run_training_dispatch_v6 import (
    DISPATCH_SCHEMA,
    job_set_sha256,
    load_dispatch,
)
from reproducibility.run_training_matrix_v4 import load_plan
from SimpleSAC.train_transit_v3 import sha256


def _capacity_index(paths: Iterable[Path]) -> dict[tuple[str, str], Path]:
    index = {}
    for raw_path in paths:
        path = raw_path.expanduser().resolve(strict=True)
        payload = json.loads(path.read_text())
        key = (str(payload.get("host_label", "")), str(payload.get("physical_gpu", "")))
        if not all(key):
            raise ValueError(f"capacity audit lacks host/GPU identity: {path}")
        if key in index:
            raise ValueError(f"duplicate capacity audit for {key}")
        index[key] = path
    return index


def _lanes() -> list[tuple[str, str]]:
    return [
        (hostname, physical_gpu)
        for hostname, physical_gpus in HOST_GPUS
        for physical_gpu in physical_gpus
    ]


def _build_core_dispatch(
    plan_path: Path,
    plan: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the v7 core layout without inheriting the v6 shared-GPU quota."""

    methods = {job["method"] for job in jobs}
    seeds = {job["seed"] for job in jobs}
    if methods != set(SERVER_CORE_METHODS) or seeds != set(FORMAL_SEEDS):
        raise ValueError("formal core dispatch lacks the registered method/seed set")
    if len(jobs) != len(SERVER_CORE_METHODS) * len(FORMAL_SEEDS):
        raise ValueError("formal core dispatch has an invalid job count")
    if any(job["max_canonical_processes_per_gpu"] != 1 for job in jobs):
        raise ValueError("formal v7 core jobs require one process per physical GPU")

    by_method_seed = _method_seed_index(jobs)
    lanes = _lanes()
    baseline_queues: dict[tuple[str, str], list[str]] = {
        lane: [] for lane in lanes
    }
    for method_index, method in enumerate(BASELINE_METHOD_ORDER):
        for seed_index, seed in enumerate(FORMAL_SEEDS):
            lane_index = (
                method_index % 2
                if seed_index == len(FORMAL_SEEDS) - 1
                else (seed_index + method_index) % len(lanes)
            )
            baseline_queues[lanes[lane_index]].append(
                by_method_seed[(method, seed)]
            )

    rlpd_assignments = {
        lanes[0]: [
            by_method_seed[("rlpd", FORMAL_SEEDS[0])],
            by_method_seed[("rlpd", FORMAL_SEEDS[1])],
        ],
        lanes[1]: [by_method_seed[("rlpd", FORMAL_SEEDS[2])]],
        lanes[2]: [by_method_seed[("rlpd", FORMAL_SEEDS[3])]],
        lanes[3]: [by_method_seed[("rlpd", FORMAL_SEEDS[4])]],
    }
    hosts = []
    for hostname, physical_gpus in HOST_GPUS:
        gpus = []
        for physical_gpu in physical_gpus:
            lane = (hostname, physical_gpu)
            stages = [
                {
                    "name": "rlpd-serial",
                    "workers": [
                        {
                            "name": f"{hostname}-gpu{physical_gpu}-rlpd",
                            "jobs": rlpd_assignments[lane],
                        }
                    ],
                }
            ]
            if baseline_queues[lane]:
                stages.append(
                    {
                        "name": "baseline-serial",
                        "workers": [
                            {
                                "name": f"{hostname}-gpu{physical_gpu}-baseline",
                                "jobs": baseline_queues[lane],
                            }
                        ],
                    }
                )
            gpus.append({"physical_gpu": physical_gpu, "stages": stages})
        hosts.append(
            {
                "hostname": hostname,
                "hostname_aliases": list(HOSTNAME_ALIASES[hostname]),
                "gpus": gpus,
            }
        )
    return {
        "schema_version": DISPATCH_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-formal-outcomes",
        "plan_id": plan.get("plan_id"),
        "plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
        "job_set_sha256": job_set_sha256(jobs),
        "policy": {
            "rlpd": (
                "four seeds execute concurrently with one process on each "
                "physical GPU; the fifth seed follows serially on the first lane"
            ),
            "other_methods": "one serial worker per physical GPU",
            "baseline_assignment": (
                "each method spans all four GPUs; its fifth seed alternates "
                "between the two faster-server GPUs"
            ),
            "runtime_comparison": (
                "wall-clock timings are resource observations and are not used "
                "for cross-method efficiency claims"
            ),
        },
        "hosts": hosts,
    }


def _build_sensitivity_dispatch(
    plan_path: Path,
    plan: dict[str, Any],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    if {job["method"] for job in jobs} != set(SERVER_SENSITIVITY_METHODS):
        raise ValueError("server sensitivity dispatch has an unexpected method set")
    if {job["seed"] for job in jobs} != set(FORMAL_SEEDS):
        raise ValueError("server sensitivity dispatch lacks the formal seed set")
    if any(job["max_canonical_processes_per_gpu"] != 1 for job in jobs):
        raise ValueError("ratio-SAC sensitivity jobs require one process per GPU")
    ordered = sorted(jobs, key=lambda job: job["seed"])
    assignments = {
        lane: [] for lane in _lanes()
    }
    for index, job in enumerate(ordered):
        assignments[_lanes()[index % len(_lanes())]].append(job["name"])
    hosts = []
    for hostname, physical_gpus in HOST_GPUS:
        gpu_payloads = []
        for physical_gpu in physical_gpus:
            names = assignments[(hostname, physical_gpu)]
            gpu_payloads.append(
                {
                    "physical_gpu": physical_gpu,
                    "stages": (
                        [
                            {
                                "name": "ratio-sac-serial",
                                "workers": [
                                    {
                                        "name": (
                                            f"{hostname}-gpu{physical_gpu}-ratio-sac"
                                        ),
                                        "jobs": names,
                                    }
                                ],
                            }
                        ]
                        if names
                        else []
                    ),
                }
            )
        hosts.append(
            {
                "hostname": hostname,
                "hostname_aliases": list(HOSTNAME_ALIASES[hostname]),
                "gpus": gpu_payloads,
            }
        )
    return {
        "schema_version": DISPATCH_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-formal-outcomes",
        "plan_id": plan.get("plan_id"),
        "plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
        "job_set_sha256": job_set_sha256(jobs),
        "policy": {
            "ratio_sac": (
                "five seeds are distributed over all four server GPUs; the "
                "fifth seed follows the first lane serially"
            ),
            "runtime_comparison": (
                "wall-clock timings are resource observations and are not used "
                "for cross-method efficiency claims"
            ),
        },
        "hosts": hosts,
    }


def _build_h2oplus_dispatch(
    plan_path: Path,
    plan: dict[str, Any],
    jobs: list[dict[str, Any]],
    expected_methods: tuple[str, ...],
    role: str,
) -> dict[str, Any]:
    """Distribute a formal H2O+ family plan over four server GPU lanes."""

    methods = {job["method"] for job in jobs}
    if methods != set(expected_methods):
        raise ValueError(f"server H2O+ {role} dispatch has an unexpected method set")
    if plan.get("execution_class") != SERVER_H2OPLUS_EXECUTION_CLASS:
        raise ValueError("H2O+ dispatch requires the server-h2oplus execution class")
    if plan.get("registration", {}).get("status") != "frozen-before-formal-outcomes":
        raise ValueError("H2O+ dispatch requires a frozen formal plan")
    if {job["seed"] for job in jobs} != set(FORMAL_SEEDS):
        raise ValueError(f"server H2O+ {role} dispatch lacks the formal seed set")
    if len(jobs) != len(expected_methods) * len(FORMAL_SEEDS):
        raise ValueError(f"server H2O+ {role} dispatch has an invalid job count")
    if any(job["max_canonical_processes_per_gpu"] != 1 for job in jobs):
        raise ValueError("server H2O+ jobs require one process per physical GPU")

    index = {(job["method"], job["seed"]): job["name"] for job in jobs}
    ordered = [
        index[(method, seed)]
        for method in expected_methods
        for seed in FORMAL_SEEDS
    ]
    assignments = {lane: [] for lane in _lanes()}
    for position, name in enumerate(ordered):
        assignments[_lanes()[position % len(_lanes())]].append(name)

    hosts = []
    for hostname, physical_gpus in HOST_GPUS:
        gpus = []
        for physical_gpu in physical_gpus:
            names = assignments[(hostname, physical_gpu)]
            stages = []
            if names:
                stages.append(
                    {
                        "name": f"h2oplus-{role}-serial",
                        "workers": [
                            {
                                "name": (
                                    f"{hostname}-gpu{physical_gpu}-h2oplus-{role}"
                                ),
                                "jobs": names,
                            }
                        ],
                    }
                )
            gpus.append({"physical_gpu": physical_gpu, "stages": stages})
        hosts.append(
            {
                "hostname": hostname,
                "hostname_aliases": list(HOSTNAME_ALIASES[hostname]),
                "gpus": gpus,
            }
        )
    return {
        "schema_version": DISPATCH_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen-before-formal-outcomes",
        "plan_id": plan.get("plan_id"),
        "plan": {"path": str(plan_path), "sha256": sha256(plan_path)},
        "job_set_sha256": job_set_sha256(jobs),
        "policy": {
            "h2oplus": (
                f"the formal H2O+ {role} jobs are deterministically distributed "
                "over all four server GPUs with one serial worker per GPU"
            ),
            "runtime_comparison": (
                "wall-clock timings are resource observations and are not used "
                "for cross-method efficiency claims"
            ),
        },
        "hosts": hosts,
    }


def build_dispatch_payload(
    plan_path: Path, capacity_audits: Iterable[Path]
) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    if plan.get("schema_version") != PLAN_SCHEMA:
        raise ValueError("v7 dispatch requires a transit-training-plan-v7 plan")
    jobs, _ = load_plan(plan_path)
    jobs_by_name = {job["name"]: job for job in jobs}
    methods = {job["method"] for job in jobs}
    if methods == set(SERVER_CORE_METHODS):
        if any(
            job["max_canonical_processes_per_gpu"]
            != FORMAL_SERVER_GPU_CONCURRENCY.get(job["method"], 1)
            for job in jobs
        ):
            raise ValueError("formal core server GPU quotas drifted")
        payload = _build_core_dispatch(plan_path, plan, jobs)
    elif methods == set(SERVER_SENSITIVITY_METHODS):
        payload = _build_sensitivity_dispatch(plan_path, plan, jobs)
    elif methods == set(H2OPLUS_CORE_METHODS):
        payload = _build_h2oplus_dispatch(
            plan_path, plan, jobs, H2OPLUS_CORE_METHODS, "core"
        )
    elif methods == set(ABLATION_METHODS):
        payload = _build_h2oplus_dispatch(
            plan_path, plan, jobs, ABLATION_METHODS, "ablation"
        )
    elif methods == set(H2OPLUS_SENSITIVITY_METHODS):
        payload = _build_h2oplus_dispatch(
            plan_path,
            plan,
            jobs,
            H2OPLUS_SENSITIVITY_METHODS,
            "sensitivity",
        )
    else:
        raise ValueError("v7 dispatch does not recognize the formal method set")
    capacities = _capacity_index(capacity_audits)
    required = set()
    for host in payload["hosts"]:
        for gpu in host["gpus"]:
            needs_capacity = any(
                len(stage["workers"]) > 1
                and any(
                    jobs_by_name[name]["method"] == "rlpd"
                    and jobs_by_name[name]["max_canonical_processes_per_gpu"] > 1
                    for worker in stage["workers"]
                    for name in worker["jobs"]
                )
                for stage in gpu["stages"]
            )
            if not needs_capacity:
                continue
            key = (host["hostname"], gpu["physical_gpu"])
            required.add(key)
            if key not in capacities:
                raise ValueError(f"missing full-data capacity audit for {key}")
            path = capacities[key]
            gpu["capacity_audit"] = {
                "path": str(path),
                "sha256": sha256(path),
            }
    extras = sorted(set(capacities) - required)
    if extras:
        raise ValueError(f"unused capacity audits do not match RLPD lanes: {extras}")
    payload["policy"]["capacity_admission"] = (
        "every shared RLPD lane is bound to a passed full-v7-data two-process "
        "capacity audit for its exact canonical host and physical GPU"
        if required
        else "no method in this plan shares a physical GPU"
    )
    payload["capacity_audit_lanes"] = [
        {"hostname": hostname, "physical_gpu": physical_gpu}
        for hostname, physical_gpu in sorted(required)
    ]
    return payload


def _relativize_descriptors(payload: dict[str, Any], output: Path) -> None:
    for host in payload["hosts"]:
        for gpu in host["gpus"]:
            descriptor = gpu.get("capacity_audit")
            if descriptor is not None:
                descriptor["path"] = os.path.relpath(
                    Path(descriptor["path"]).resolve(), output.parent
                )


def write_dispatch(
    plan_path: Path, capacity_audits: Iterable[Path], output: Path
) -> Path:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen dispatch {output}")
    payload = build_dispatch_payload(plan_path, capacity_audits)
    plan = plan_path.expanduser().resolve()
    payload["plan"]["path"] = os.path.relpath(plan, output.parent)
    _relativize_descriptors(payload, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    try:
        load_dispatch(output)
    except Exception:
        output.unlink(missing_ok=True)
        raise
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument(
        "--capacity-audit", type=Path, action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_dispatch(args.plan, args.capacity_audit, args.output)
    print(
        json.dumps(
            {"dispatch": str(output), "sha256": sha256(output)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
