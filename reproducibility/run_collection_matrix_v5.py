#!/usr/bin/env python3
"""Run a resumable, provenance-locked SUMO v5 collection matrix."""

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
COLLECTOR = ROOT / "collect_policy" / "collect_transit_v5.py"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.mdp_contract_v5 import DEFAULT_CONFIG_V5
from bus_h2o.sumo_provenance import (
    collection_source_paths,
    hash_files,
    load_validated_scenario,
    sha256,
    sumo_version,
)


POLICIES = (
    {"policy": "zero", "policy_name": "zero"},
    {
        "policy": "random-hurdle",
        "policy_name": "random-hurdle-p0.5-beta1-3",
        "random_zero_probability": 0.5,
        "random_beta_alpha": 1.0,
        "random_beta_beta": 3.0,
    },
    {"policy": "headway-balance", "policy_name": "headway-gain-0.05", "headway_gain": 0.05},
    {"policy": "headway-balance", "policy_name": "headway-gain-0.1", "headway_gain": 0.1},
    {"policy": "headway-balance", "policy_name": "headway-gain-0.25", "headway_gain": 0.25},
    {"policy": "headway-balance", "policy_name": "headway-gain-0.5", "headway_gain": 0.5},
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
    """Return a stable seed independent of policy or block selection order."""

    canonical_index = {
        policy["policy_name"]: index for index, policy in enumerate(POLICIES)
    }
    if policy_name not in canonical_index:
        raise ValueError(f"unknown behavior policy {policy_name!r}")
    payload = (
        f"{int(base)}:{canonical_index[policy_name]}:"
        f"{scenario_sha256}:{int(sumo_seed_value)}"
    ).encode()
    # Keep the seed representable in the dataset's signed int32 metadata.
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (
        2**31 - 1
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--sumo-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--design",
        choices=("paired", "crossed"),
        default="paired",
        help="pair manifests and SUMO seeds by position, or use their Cartesian product",
    )
    parser.add_argument("--study-stage", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    parser.add_argument("--policy-seed-base", type=int, default=20_260_804)
    parser.add_argument(
        "--episode-seconds",
        type=int,
        default=int(DEFAULT_CONFIG_V5.episode_seconds),
    )
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="allow a shortened horizon/event cap; generated shards are not training eligible",
    )
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
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


def _manifest(path: Path) -> dict:
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
        raise ValueError("scenario manifests must be unique")
    if design == "paired":
        if len(scenarios) != len(sumo_seeds):
            raise ValueError(
                "paired design requires the same number of scenario manifests and SUMO seeds"
            )
        pairs = zip(scenarios, sumo_seeds)
    else:
        pairs = (
            (scenario, sumo_seed)
            for scenario in scenarios
            for sumo_seed in sumo_seeds
        )
    blocks = [
        {"block_id": index, "scenario": scenario, "sumo_seed": int(sumo_seed)}
        for index, (scenario, sumo_seed) in enumerate(pairs)
    ]
    logical = {(block["scenario"]["sha256"], block["sumo_seed"]) for block in blocks}
    if len(logical) != len(blocks):
        raise ValueError("environment blocks must be unique")
    return blocks


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
        return (
            summary.get("schema_version") == "transit-smdp-v5-shard-v2"
            and summary.get("job_signature") == job["signature"]
            and summary.get("shard_sha256") == sha256(output)
            and summary.get("source_sha256") == job["source_sha256"]
            and summary.get("runtime", {}).get("sumo_version") == job["sumo_version"]
            and summary.get("scenario", {}).get("runtime_input_sha256")
            == job["scenario"]["runtime_input_sha256"]
            and summary.get("split") == job["split"]
            and bool(summary.get("diagnostic")) == job["diagnostic"]
            and bool(summary.get("training_eligible")) == (not job["diagnostic"])
        )
    except Exception:
        return False


def _quarantine(path: Path) -> None:
    if not path.exists():
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    os.replace(path, path.with_name(path.name + f".invalid-{timestamp}"))


def _run_job(job: dict) -> dict:
    if _is_complete(job):
        return {"status": "skipped", "job": job}
    output = job["output"]
    summary_path = output.with_suffix(output.suffix + ".json")
    _quarantine(output)
    _quarantine(summary_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(output.suffix + ".log")
    _quarantine(log_path)
    with log_path.open("w") as log:
        result = subprocess.run(
            _job_command(job),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0 or not _is_complete(job):
        return {
            "status": "failed",
            "returncode": result.returncode,
            "log": str(log_path),
            "job": job,
        }
    return {"status": "completed", "job": job}


def _serializable_job(job: dict) -> dict:
    return {
        **job,
        "output": str(job["output"]),
        "sumo_dir": str(job["sumo_dir"]),
        "edge_xml": str(job["edge_xml"]),
        "scenario": {**job["scenario"], "path": str(job["scenario"]["path"])},
    }


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.episode_seconds <= 0 or args.max_events < 0:
        raise ValueError("episode/max-event limits are invalid")
    diagnostic_conditions = (
        args.episode_seconds != int(DEFAULT_CONFIG_V5.episode_seconds)
        or args.max_events > 0
    )
    if diagnostic_conditions != bool(args.diagnostic):
        raise ValueError(
            "--diagnostic must be present exactly when using a shortened horizon or event cap"
        )
    if args.study_stage == "formal" and args.diagnostic:
        raise ValueError("formal collection cannot be diagnostic")

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".collection.lock"
    lock = lock_path.open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"collection matrix already running under {output_root}") from error

    scenarios = [_manifest(path) for path in args.scenario_manifests]
    blocks = _blocks(scenarios, args.sumo_seeds, args.design)
    selected_names = set(
        args.policies or (policy["policy_name"] for policy in POLICIES)
    )
    policies = [
        policy for policy in POLICIES if policy["policy_name"] in selected_names
    ]
    sumo_dir = args.sumo_dir.expanduser().resolve()
    edge_xml = args.edge_xml.expanduser().resolve()
    source_sha256 = hash_files(
        collection_source_paths(ROOT, sumo_dir, edge_xml, contract_version="v5")
    )
    collector_hash = source_sha256[str(COLLECTOR)]
    runner_hash = sha256(Path(__file__).resolve())
    active_sumo_version = sumo_version()
    jobs = []
    for policy in policies:
        for block in blocks:
            scenario = block["scenario"]
            sumo_seed_value = block["sumo_seed"]
            behavior_seed = policy_seed(
                args.policy_seed_base,
                policy["policy_name"],
                scenario["sha256"],
                sumo_seed_value,
            )
            output = (
                output_root
                / policy["policy_name"]
                / scenario["tag"]
                / f"sumo_seed{sumo_seed_value}.npz"
            )
            payload = {
                "collector_sha256": collector_hash,
                "runner_sha256": runner_hash,
                "source_sha256": source_sha256,
                "sumo_version": active_sumo_version,
                "study_stage": args.study_stage,
                "design": args.design,
                "block_id": block["block_id"],
                "policy": policy,
                "policy_seed": behavior_seed,
                "sumo_seed": sumo_seed_value,
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
                    "policy": policy,
                    "policy_seed": behavior_seed,
                    "sumo_seed": sumo_seed_value,
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
                    "signature": signature(payload),
                }
            )

    matrix_manifest = {
        "schema_version": "transit-smdp-v5-collection-matrix-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_stage": args.study_stage,
        "environment_design": args.design,
        "independent_environment_blocks": len(blocks),
        "paired_policy_comparison": True,
        "collector": str(COLLECTOR),
        "collector_sha256": collector_hash,
        "runner_sha256": runner_hash,
        "source_sha256": source_sha256,
        "sumo_version": active_sumo_version,
        "workers": min(args.workers, len(jobs)),
        "jobs": [_serializable_job(job) for job in jobs],
    }
    manifest_path = output_root / "matrix_manifest.json"
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(matrix_manifest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, manifest_path)

    failures = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(args.workers, len(jobs))
    ) as executor:
        futures = [executor.submit(_run_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed += 1
            job = result["job"]
            print(
                f"[{completed}/{len(jobs)}] {result['status']} "
                f"block={job['block_id']} {job['policy']['policy_name']} "
                f"{job['scenario']['tag']} sumo_seed={job['sumo_seed']}",
                flush=True,
            )
            if result["status"] == "failed":
                failures.append(result)
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    if failures:
        details = "\n".join(
            f"{failure['job']['output']}: {failure['log']}" for failure in failures
        )
        raise RuntimeError(f"{len(failures)} collection jobs failed:\n{details}")


if __name__ == "__main__":
    main()
