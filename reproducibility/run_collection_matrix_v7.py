#!/usr/bin/env python3
"""Run a resumable, provenance-locked SUMO v7 collection matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "collect_policy" / "collect_transit_v7.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v7 import DEFAULT_CONFIG_V7
from bus_h2o.full_service_trace_v7 import (
    FULL_SERVICE_TRACE_SCHEMA,
    full_service_trace_path,
    load_full_service_trace,
)
from bus_h2o.sumo_provenance import (
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    collection_source_paths,
    hash_files,
    load_sumo_runtime_lock,
    load_validated_scenario,
    sha256,
    sumo_runtime_fingerprint,
)
from reproducibility.build_training_environment_design_v7 import (
    TRAINING_DESIGN_SCHEMA,
)


MATRIX_SCHEMA = "transit-smdp-v7-collection-matrix-v2"
SHARD_SCHEMA = "transit-smdp-v7-shard-v1"
POLICY_MIXTURE_SCHEMA = "transit-smdp-v7-behavior-mixture-v1"
POLICIES = (
    {"policy": "zero", "policy_name": "zero"},
    {
        "policy": "random-hurdle",
        "policy_name": "random-hurdle-p0.5-beta1-3",
        "random_zero_probability": 0.5,
        "random_beta_alpha": 1.0,
        "random_beta_beta": 3.0,
    },
    {
        "policy": "random-hurdle",
        "policy_name": "random-hurdle-p0.5-uniform",
        "random_zero_probability": 0.5,
        "random_beta_alpha": 1.0,
        "random_beta_beta": 1.0,
    },
    {
        "policy": "headway-balance",
        "policy_name": "headway-gain-0.05",
        "headway_gain": 0.05,
    },
    {
        "policy": "headway-balance",
        "policy_name": "headway-gain-0.1",
        "headway_gain": 0.1,
    },
    {
        "policy": "headway-balance",
        "policy_name": "headway-gain-0.25",
        "headway_gain": 0.25,
    },
    {
        "policy": "headway-balance",
        "policy_name": "headway-gain-0.5",
        "headway_gain": 0.5,
    },
    {"policy": "myopic-welfare", "policy_name": "myopic-welfare-v2"},
)


def signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def policy_seed(
    base: int,
    policy_name: str,
    scenario_sha256: str,
    sumo_seed_value: int,
) -> int:
    """Return a selection-order-independent signed-int32 behavior seed."""

    canonical_index = {
        policy["policy_name"]: index for index, policy in enumerate(POLICIES)
    }
    if policy_name not in canonical_index:
        raise ValueError(f"unknown v7 behavior policy {policy_name!r}")
    payload = (
        f"{int(base)}:{canonical_index[policy_name]}:"
        f"{scenario_sha256}:{int(sumo_seed_value)}"
    ).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (
        2**31 - 1
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifests", type=Path, nargs="+")
    parser.add_argument("--sumo-seeds", type=int, nargs="+")
    parser.add_argument(
        "--environment-design",
        type=Path,
        help="formal v7 training design; replaces explicit scenario/seed lists",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--design", choices=("paired", "crossed"), default="paired")
    parser.add_argument("--study-stage", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    parser.add_argument("--sumo-runtime-lock", type=Path, required=True)
    parser.add_argument(
        "--block-ids",
        type=int,
        nargs="+",
        help=(
            "execution-only environment-cell block subset for a pilot or resume; "
            "the registered matrix remains complete"
        ),
    )
    parser.add_argument("--policy-seed-base", type=int, default=20_260_804)
    parser.add_argument(
        "--episode-seconds",
        type=int,
        default=int(DEFAULT_CONFIG_V7.horizon_seconds),
    )
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), default="train"
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=tuple(policy["policy_name"] for policy in POLICIES),
    )
    parser.add_argument(
        "--sumo-dir",
        type=Path,
        default=ROOT.parent / "SUMO_ruiguang" / "online_control",
    )
    parser.add_argument(
        "--edge-xml",
        type=Path,
        default=ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml",
    )
    return parser.parse_args()


def _scenario(path: Path) -> dict:
    resolved = path.expanduser().resolve()
    _, values = load_validated_scenario(resolved)
    return {
        "path": resolved,
        "sha256": values["manifest_sha256"],
        "tag": resolved.parent.name,
        "demand_scale": float(values["demand_scale"]),
        "selection_seed": int(values["selection_seed"]),
        "depart_jitter_seconds": float(values.get("depart_jitter_seconds", 0.0)),
        "runtime_input_sha256": values["runtime_input_sha256"],
    }


def _blocks(scenarios: list[dict], sumo_seeds: list[int], design: str) -> list[dict]:
    if len({scenario["sha256"] for scenario in scenarios}) != len(scenarios):
        raise ValueError("v7 scenario manifests must be unique")
    if design == "paired":
        if len(scenarios) != len(sumo_seeds):
            raise ValueError("paired v7 design requires equal scenario/seed counts")
        pairs = zip(scenarios, sumo_seeds)
    else:
        pairs = (
            (scenario, seed) for scenario in scenarios for seed in sumo_seeds
        )
    blocks = [
        {
            "block_id": index,
            "realization_block_id": index,
            "scenario": scenario,
            "sumo_seed": int(seed),
        }
        for index, (scenario, seed) in enumerate(pairs)
    ]
    if len({(b["scenario"]["sha256"], b["sumo_seed"]) for b in blocks}) != len(
        blocks
    ):
        raise ValueError("v7 environment blocks must be unique")
    return blocks


def load_environment_design(path: Path) -> tuple[list[dict], dict]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != TRAINING_DESIGN_SCHEMA:
        raise ValueError("v7 training environment design has an unsupported schema")
    generator = Path(payload.get("generator", "")).expanduser().resolve()
    if generator != (
        ROOT / "reproducibility" / "build_training_environment_design_v7.py"
    ).resolve() or payload.get("generator_sha256") != sha256(generator):
        raise ValueError("v7 training design generator provenance changed")
    scenario_generator = (
        ROOT / "reproducibility" / "build_passenger_scenarios.py"
    ).resolve()
    if payload.get("scenario_generator_sha256") != sha256(scenario_generator):
        raise ValueError("v7 passenger-scenario generator provenance changed")
    reuse = payload.get("physical_scenario_reuse", {})
    reuse_path = Path(reuse.get("path", "")).expanduser().resolve()
    if reuse.get("sha256") != sha256(reuse_path):
        raise ValueError("v7 physical-scenario reuse evidence changed")
    for descriptor in payload.get("source_sha256", {}).values():
        source = Path(descriptor["path"]).expanduser().resolve()
        if descriptor.get("sha256") != sha256(source):
            raise ValueError(f"v7 training source changed: {source}")
    disjointness = payload.get("evaluation_disjointness", {})
    evaluation_path = Path(disjointness.get("path", "")).expanduser().resolve()
    if (
        not disjointness.get("verified_disjoint")
        or disjointness.get("sha256") != sha256(evaluation_path)
    ):
        raise ValueError("v7 evaluation-disjointness evidence changed")
    declared = payload.get("blocks")
    if not isinstance(declared, list) or not declared:
        raise ValueError("v7 training environment design contains no blocks")
    if len({block["name"] for block in declared}) != len(declared):
        raise ValueError("v7 training realization names must be unique")
    demand_seeds = [int(block["demand_realization_seed"]) for block in declared]
    sumo_seeds = [int(block["sumo_seed"]) for block in declared]
    if len(set(demand_seeds)) != len(demand_seeds):
        raise ValueError("v7 training demand-realization seeds are reused")
    if len(set(sumo_seeds)) != len(sumo_seeds):
        raise ValueError("v7 training SUMO seeds are reused")

    cells = []
    expected_scales = tuple(float(value) for value in payload["demand_scales"])
    seen_scenarios = set()
    for realization_id, block in enumerate(declared):
        scenarios = [_scenario(Path(value)) for value in block["scenario_manifests"]]
        scales = tuple(scenario["demand_scale"] for scenario in scenarios)
        if scales != expected_scales:
            raise ValueError(
                f"v7 realization {block['name']} does not contain the declared scales"
            )
        if any(
            scenario["selection_seed"] != int(block["demand_realization_seed"])
            for scenario in scenarios
        ):
            raise ValueError(
                f"v7 realization {block['name']} has a demand-seed mismatch"
            )
        if any(
            scenario["depart_jitter_seconds"]
            != float(payload["depart_jitter_seconds"])
            for scenario in scenarios
        ):
            raise ValueError(
                f"v7 realization {block['name']} has a departure-jitter mismatch"
            )
        for scenario in scenarios:
            if scenario["sha256"] in seen_scenarios:
                raise ValueError("v7 training design reuses a scenario manifest")
            seen_scenarios.add(scenario["sha256"])
            cells.append(
                {
                    "block_id": len(cells),
                    "realization_block_id": realization_id,
                    "realization_block_name": block["name"],
                    "scenario": scenario,
                    "sumo_seed": int(block["sumo_seed"]),
                }
            )
    if len(cells) != int(payload.get("environment_cells", -1)):
        raise ValueError("v7 training design environment-cell count is inconsistent")
    if len(declared) != int(payload.get("independent_realization_blocks", -1)):
        raise ValueError("v7 training design realization count is inconsistent")
    evaluation_payload = json.loads(evaluation_path.read_text())
    evaluation_demand = {
        int(block["demand_realization_seed"])
        for block in evaluation_payload.get("blocks", [])
    }
    evaluation_sumo = {
        int(block["sumo_seed"])
        for block in evaluation_payload.get("blocks", [])
    }
    if set(demand_seeds) & evaluation_demand or set(sumo_seeds) & evaluation_sumo:
        raise ValueError("v7 training and evaluation designs now overlap")
    evidence = {
        "path": str(resolved),
        "sha256": sha256(resolved),
        "schema_version": payload["schema_version"],
        "independent_realization_blocks": len(declared),
        "environment_cells": len(cells),
        "demand_scales": list(expected_scales),
        "evaluation_disjointness": disjointness,
        "physical_scenario_reuse": reuse,
    }
    return cells, evidence


def _job_command(job: dict) -> list[str]:
    policy = job["policy"]
    command = [
        sys.executable,
        str(COLLECTOR),
        "--policy",
        policy["policy"],
        "--policy-name",
        policy["policy_name"],
        "--policy-seed",
        str(job["policy_seed"]),
        "--sumo-seed",
        str(job["sumo_seed"]),
        "--split",
        job["split"],
        "--scenario-manifest",
        str(job["scenario"]["path"]),
        "--episode-seconds",
        str(job["episode_seconds"]),
        "--max-events",
        str(job["max_events"]),
        "--output",
        str(job["output"]),
        "--job-signature",
        job["signature"],
        "--sumo-backend",
        job["sumo_backend"],
        "--expected-sumo-runtime-version",
        job["sumo_runtime_version"],
        "--expected-sumo-runtime-signature",
        job["sumo_runtime_signature"],
        "--sumo-runtime-lock",
        str(job["sumo_runtime_lock"]),
        "--expected-sumo-runtime-lock-sha256",
        job["sumo_runtime_lock_sha256"],
        "--sumo-dir",
        str(job["sumo_dir"]),
        "--edge-xml",
        str(job["edge_xml"]),
    ]
    if job["diagnostic"]:
        command.append("--diagnostic")
    for name in (
        "headway_gain",
        "random_zero_probability",
        "random_beta_alpha",
        "random_beta_beta",
    ):
        if name in policy:
            command.extend(("--" + name.replace("_", "-"), str(policy[name])))
    return command


def _is_complete(job: dict) -> bool:
    output = job["output"]
    summary_path = output.with_suffix(output.suffix + ".json")
    if not output.is_file() or not summary_path.is_file():
        return False
    try:
        summary = json.loads(summary_path.read_text())
        coverage = summary.get("training_reward", {}).get(
            "chronological_coverage_errors", {}
        )
        base_complete = (
            summary.get("schema_version") == SHARD_SCHEMA
            and summary.get("job_signature") == job["signature"]
            and summary.get("shard_sha256") == sha256(output)
            and summary.get("source_sha256") == job["source_sha256"]
            and summary.get("runtime", {}).get("sumo_version")
            == job["sumo_version"]
            and summary.get("runtime", {}).get("sumo_backend")
            == job["sumo_backend"]
            and summary.get("runtime", {}).get("sumo_runtime_signature")
            == job["sumo_runtime_signature"]
            and summary.get("runtime", {}).get("sumo_runtime_lock", {}).get(
                "sha256"
            )
            == job["sumo_runtime_lock_sha256"]
            and summary.get("runtime", {}).get(
                "sumo_runtime_fingerprint", {}
            ).get("portable_identity")
            == job["sumo_runtime_fingerprint"]["portable_identity"]
            and summary.get("scenario", {}).get("runtime_input_sha256")
            == job["scenario"]["runtime_input_sha256"]
            and summary.get("split") == job["split"]
            and bool(summary.get("diagnostic")) == job["diagnostic"]
            and bool(summary.get("training_eligible")) == (not job["diagnostic"])
            and all(abs(float(value)) <= 1e-9 for value in coverage.values())
            and summary.get("bus_service_semantics_audit", {}).get("passed")
            is True
        )
        if not base_complete:
            return False
        requires_full_service_trace = (
            job.get("policy", {}).get("policy_name") == "zero"
            and not job["diagnostic"]
        )
        if not requires_full_service_trace:
            return summary.get("full_service_trace") is None
        trace_path = full_service_trace_path(output)
        trace_summary, trace_arrays = load_full_service_trace(trace_path)
        descriptor = summary.get("full_service_trace", {})
        return (
            descriptor.get("schema_version") == FULL_SERVICE_TRACE_SCHEMA
            and Path(descriptor.get("path", "")).expanduser().resolve()
            == trace_path
            and descriptor.get("sha256") == sha256(trace_path)
            and descriptor.get("summary_sha256")
            == sha256(trace_path.with_suffix(trace_path.suffix + ".json"))
            and trace_summary.get("source_shard", {}).get("sha256")
            == summary["shard_sha256"]
            and trace_summary.get("source_shard", {}).get("job_signature")
            == job["signature"]
            and trace_summary.get("collection_source_sha256")
            == job["source_sha256"]
            and trace_summary.get("full_service_trace", {}).get(
                "completed_trips"
            )
            == summary.get("service", {}).get("completed_trips")
            and len(trace_arrays["segment_bus_code"])
            == trace_summary["full_service_trace"]["segments"]
        )
    except Exception:
        return False


def _quarantine(path: Path) -> None:
    if path.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        os.replace(path, path.with_name(path.name + f".invalid-{timestamp}"))


def _run_job(job: dict) -> dict:
    if _is_complete(job):
        return {"status": "skipped", "job": job}
    output = job["output"]
    summary_path = output.with_suffix(output.suffix + ".json")
    log_path = output.with_suffix(output.suffix + ".log")
    trace_path = full_service_trace_path(output)
    for path in (
        output,
        summary_path,
        log_path,
        trace_path,
        trace_path.with_suffix(trace_path.suffix + ".json"),
    ):
        _quarantine(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        result = subprocess.run(
            _job_command(job),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode or not _is_complete(job):
        return {
            "status": "failed",
            "returncode": result.returncode,
            "log": str(log_path),
            "job": job,
        }
    return {"status": "completed", "job": job}


def _serializable(job: dict) -> dict:
    return {
        **job,
        "output": str(job["output"]),
        "sumo_dir": str(job["sumo_dir"]),
        "edge_xml": str(job["edge_xml"]),
        "sumo_runtime_lock": str(job["sumo_runtime_lock"]),
        "scenario": {**job["scenario"], "path": str(job["scenario"]["path"])},
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.episode_seconds <= 0 or args.max_events < 0:
        raise ValueError("v7 worker/episode/event limits are invalid")
    diagnostic = (
        args.episode_seconds != int(DEFAULT_CONFIG_V7.horizon_seconds)
        or args.max_events > 0
    )
    if diagnostic != bool(args.diagnostic):
        raise ValueError("--diagnostic must exactly match shortened v7 collection")
    if args.study_stage == "formal" and args.diagnostic:
        raise ValueError("formal v7 collection cannot be diagnostic")
    if args.environment_design is not None:
        if args.scenario_manifests is not None or args.sumo_seeds is not None:
            raise ValueError(
                "--environment-design cannot be combined with explicit scenarios/seeds"
            )
        blocks, design_evidence = load_environment_design(args.environment_design)
    else:
        if args.scenario_manifests is None or args.sumo_seeds is None:
            raise ValueError(
                "provide --environment-design or both --scenario-manifests and --sumo-seeds"
            )
        scenarios = [_scenario(path) for path in args.scenario_manifests]
        blocks = _blocks(scenarios, args.sumo_seeds, args.design)
        design_evidence = {
            "mode": args.design,
            "independent_realization_blocks": len(blocks),
            "environment_cells": len(blocks),
            "demand_scales": sorted(
                {float(block["scenario"]["demand_scale"]) for block in blocks}
            ),
        }

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock = (output_root / ".collection.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"v7 collection already running: {output_root}") from error

    selected = set(
        args.policies or (policy["policy_name"] for policy in POLICIES)
    )
    policies = [p for p in POLICIES if p["policy_name"] in selected]
    sumo_dir = args.sumo_dir.expanduser().resolve()
    edge_xml = args.edge_xml.expanduser().resolve()
    source_sha256 = hash_files(
        collection_source_paths(ROOT, sumo_dir, edge_xml, contract_version="v7")
    )
    collector_hash = source_sha256[str(COLLECTOR)]
    runner_hash = sha256(Path(__file__).resolve())
    sumo_runtime_lock = args.sumo_runtime_lock.expanduser().resolve()
    runtime_lock, runtime_lock_identity = load_sumo_runtime_lock(
        sumo_runtime_lock
    )
    active_sumo_runtime = sumo_runtime_fingerprint(
        FORMAL_V7_SUMO_BACKEND,
        expected_version=FORMAL_V7_SUMO_VERSION,
    )
    active_sumo_version = active_sumo_runtime["runtime_version"]
    active_sumo_runtime_signature = active_sumo_runtime["portable_signature"]
    if (
        runtime_lock_identity["runtime_signature"]
        != active_sumo_runtime_signature
        or runtime_lock["runtime_fingerprint"]["portable_identity"]
        != active_sumo_runtime["portable_identity"]
    ):
        raise RuntimeError("active SUMO runtime differs from the collection lock")
    jobs = []
    for policy in policies:
        for block in blocks:
            scenario = block["scenario"]
            seed = block["sumo_seed"]
            behavior_seed = policy_seed(
                args.policy_seed_base, policy["policy_name"], scenario["sha256"], seed
            )
            output = (
                output_root
                / policy["policy_name"]
                / scenario["tag"]
                / f"sumo_seed{seed}.npz"
            )
            payload = {
                "collector_sha256": collector_hash,
                "runner_sha256": runner_hash,
                "source_sha256": source_sha256,
                "sumo_version": active_sumo_version,
                "sumo_backend": FORMAL_V7_SUMO_BACKEND,
                "sumo_runtime_signature": active_sumo_runtime_signature,
                "sumo_runtime_lock_sha256": runtime_lock_identity["sha256"],
                "study_stage": args.study_stage,
                "design": args.design,
                "block_id": block["block_id"],
                "realization_block_id": block["realization_block_id"],
                "policy": policy,
                "policy_mixture_schema": POLICY_MIXTURE_SCHEMA,
                "policy_seed": behavior_seed,
                "sumo_seed": seed,
                "split": args.split,
                "scenario_manifest_sha256": scenario["sha256"],
                "scenario_runtime_input_sha256": scenario["runtime_input_sha256"],
                "episode_seconds": args.episode_seconds,
                "max_events": args.max_events,
                "diagnostic": args.diagnostic,
                "sumo_dir": str(sumo_dir),
                "edge_xml_sha256": sha256(edge_xml),
            }
            jobs.append(
                {
                    "block_id": block["block_id"],
                    "realization_block_id": block["realization_block_id"],
                    "realization_block_name": block.get("realization_block_name"),
                    "policy": policy,
                    "policy_seed": behavior_seed,
                    "sumo_seed": seed,
                    "split": args.split,
                    "scenario": scenario,
                    "episode_seconds": args.episode_seconds,
                    "max_events": args.max_events,
                    "diagnostic": bool(args.diagnostic),
                    "sumo_dir": sumo_dir,
                    "edge_xml": edge_xml,
                    "output": output,
                    "source_sha256": source_sha256,
                    "sumo_version": active_sumo_version,
                    "sumo_backend": FORMAL_V7_SUMO_BACKEND,
                    "sumo_runtime_version": active_sumo_version,
                    "sumo_runtime_signature": active_sumo_runtime_signature,
                    "sumo_runtime_fingerprint": active_sumo_runtime,
                    "sumo_runtime_lock": sumo_runtime_lock,
                    "sumo_runtime_lock_sha256": runtime_lock_identity["sha256"],
                    "signature": signature(payload),
                }
            )

    matrix = {
        "schema_version": MATRIX_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": args.study_stage,
        "environment_design": design_evidence,
        "independent_realization_blocks": design_evidence[
            "independent_realization_blocks"
        ],
        "environment_cells": len(blocks),
        "paired_policy_comparison": True,
        "behavior_mixture": {
            "schema_version": POLICY_MIXTURE_SCHEMA,
            "policies": list(POLICIES),
            "selected_policy_names": [
                policy["policy_name"] for policy in policies
            ],
            "coverage_rationale": (
                "zero and low-hold operational policies are complemented by "
                "a zero-inflated uniform positive-hold policy so the full "
                "admissible holding range has logged support"
            ),
        },
        "collector": str(COLLECTOR),
        "collector_sha256": collector_hash,
        "runner_sha256": runner_hash,
        "source_sha256": source_sha256,
        "sumo_version": active_sumo_version,
        "sumo_backend": FORMAL_V7_SUMO_BACKEND,
        "sumo_runtime_signature": active_sumo_runtime_signature,
        "sumo_runtime_fingerprint": active_sumo_runtime,
        "sumo_runtime_lock": runtime_lock_identity,
        "workers": min(args.workers, len(jobs)),
        "jobs": [_serializable(job) for job in jobs],
    }
    manifest_path = output_root / "matrix_manifest.json"
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)

    execution_jobs = jobs
    if args.block_ids is not None:
        requested = set(args.block_ids)
        registered = {int(job["block_id"]) for job in jobs}
        if not requested or not requested <= registered:
            raise ValueError(
                "requested realization blocks are not registered: "
                f"{sorted(requested - registered)}"
            )
        execution_jobs = [
            job
            for job in jobs
            if int(job["block_id"]) in requested
        ]

    failures = []
    completed = 0
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, len(execution_jobs))
        ) as executor:
            futures = [executor.submit(_run_job, job) for job in execution_jobs]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                completed += 1
                job = result["job"]
                print(
                    f"[{completed}/{len(execution_jobs)}] {result['status']} "
                    f"block={job['block_id']} {job['policy']['policy_name']} "
                    f"{job['scenario']['tag']} sumo_seed={job['sumo_seed']}",
                    flush=True,
                )
                if result["status"] == "failed":
                    failures.append(result)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    if failures:
        details = "\n".join(
            f"{failure['job']['output']}: {failure['log']}"
            for failure in failures
        )
        raise RuntimeError(f"{len(failures)} v7 collection jobs failed:\n{details}")


if __name__ == "__main__":
    main()
