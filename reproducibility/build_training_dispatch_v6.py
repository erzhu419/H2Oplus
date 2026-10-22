#!/usr/bin/env python3
"""Freeze the canonical two-server dispatch for the formal v6 baseline plan."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.build_training_plan_v6 import (
    FORMAL_SEEDS,
    SERVER_CORE_METHODS,
)
from reproducibility.run_training_dispatch_v6 import (
    DISPATCH_SCHEMA,
    job_set_sha256,
    load_dispatch,
)
from reproducibility.run_training_matrix_v4 import load_plan
from SimpleSAC.train_transit_v3 import sha256


HOST_GPUS = (
    ("jtl110gpu", ("0", "1")),
    ("jtl110gpu2", ("0", "1")),
)
HOSTNAME_ALIASES = {
    "jtl110gpu": ("huiwei-Super-Server",),
    "jtl110gpu2": ("erzhu419-Super-Server",),
}
BASELINE_METHOD_ORDER = (
    "wsrl",
    "cql",
    "iql",
    "sac-mixed",
    "sac-online",
    "bc",
)


def _method_seed_index(jobs: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    index: dict[tuple[str, int], str] = {}
    for job in jobs:
        key = (job["method"], job["seed"])
        if key in index:
            raise ValueError(f"duplicate method/seed job in formal plan: {key}")
        index[key] = job["name"]
    return index


def build_dispatch_payload(plan_path: Path) -> dict[str, Any]:
    plan_path = plan_path.expanduser().resolve()
    jobs, _ = load_plan(plan_path)
    plan = json.loads(plan_path.read_text())
    methods = {job["method"] for job in jobs}
    seeds = {job["seed"] for job in jobs}
    if set(BASELINE_METHOD_ORDER) != set(SERVER_CORE_METHODS) - {"rlpd"}:
        raise RuntimeError("baseline dispatch order no longer covers server methods")
    if methods != set(SERVER_CORE_METHODS) or seeds != set(FORMAL_SEEDS):
        raise ValueError(
            "canonical server dispatch requires the complete formal core baseline plan"
        )
    expected_count = len(SERVER_CORE_METHODS) * len(FORMAL_SEEDS)
    if len(jobs) != expected_count:
        raise ValueError(
            f"formal server plan has {len(jobs)} jobs, expected {expected_count}"
        )
    for job in jobs:
        expected_quota = 2 if job["method"] == "rlpd" else 1
        if job["max_canonical_processes_per_gpu"] != expected_quota:
            raise ValueError(
                f"{job['name']} has GPU quota "
                f"{job['max_canonical_processes_per_gpu']}, expected {expected_quota}"
            )

    by_method_seed = _method_seed_index(jobs)
    lanes = [
        (hostname, physical_gpu)
        for hostname, physical_gpus in HOST_GPUS
        for physical_gpu in physical_gpus
    ]
    queues: dict[tuple[str, str], list[str]] = {lane: [] for lane in lanes}
    for method_index, method in enumerate(BASELINE_METHOD_ORDER):
        for seed_index, seed in enumerate(FORMAL_SEEDS):
            lane_index = (
                method_index % 2
                if seed_index == len(FORMAL_SEEDS) - 1
                else (seed_index + method_index) % len(lanes)
            )
            queues[lanes[lane_index]].append(by_method_seed[(method, seed)])

    rlpd_stages = {
        lanes[0]: [
            by_method_seed[("rlpd", FORMAL_SEEDS[0])],
            by_method_seed[("rlpd", FORMAL_SEEDS[1])],
        ],
        lanes[1]: [
            by_method_seed[("rlpd", FORMAL_SEEDS[2])],
            by_method_seed[("rlpd", FORMAL_SEEDS[3])],
        ],
        lanes[2]: [by_method_seed[("rlpd", FORMAL_SEEDS[4])]],
    }
    hosts = []
    for hostname, physical_gpus in HOST_GPUS:
        gpu_payloads = []
        for physical_gpu in physical_gpus:
            lane = (hostname, physical_gpu)
            stages = []
            rlpd_jobs = rlpd_stages.get(lane, [])
            if rlpd_jobs:
                stages.append(
                    {
                        "name": "rlpd-shared" if len(rlpd_jobs) == 2 else "rlpd-single",
                        "workers": [
                            {
                                "name": f"{hostname}-gpu{physical_gpu}-rlpd-slot{slot}",
                                "jobs": [job_name],
                            }
                            for slot, job_name in enumerate(rlpd_jobs)
                        ],
                    }
                )
            baseline_jobs = queues[lane]
            if baseline_jobs:
                stages.append(
                    {
                        "name": "baseline-serial",
                        "workers": [
                            {
                                "name": f"{hostname}-gpu{physical_gpu}-baseline",
                                "jobs": baseline_jobs,
                            }
                        ],
                    }
                )
            gpu_payloads.append(
                {"physical_gpu": physical_gpu, "stages": stages}
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
        "plan": {
            "path": os.path.relpath(plan_path, Path.cwd()),
            "sha256": sha256(plan_path),
        },
        "job_set_sha256": job_set_sha256(jobs),
        "policy": {
            "rlpd": (
                "two processes per GPU only on jtl110gpu; the fifth seed uses "
                "one slot on jtl110gpu2 GPU 0"
            ),
            "other_methods": "one serial worker per physical GPU",
            "baseline_assignment": (
                "each method spans all four GPUs; its fifth seed alternates between "
                "the two faster-server GPUs"
            ),
            "runtime_comparison": (
                "RLPD wall-clock timings are resource observations and are not used "
                "for cross-method efficiency claims"
            ),
        },
        "hosts": hosts,
    }


def write_dispatch(plan_path: Path, output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen dispatch {output}")
    payload = build_dispatch_payload(plan_path)
    plan = plan_path.expanduser().resolve()
    payload["plan"]["path"] = os.path.relpath(plan, output.parent)
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
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = write_dispatch(args.plan, args.output)
    print(
        json.dumps(
            {"dispatch": str(output), "sha256": sha256(output)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
