#!/usr/bin/env python3
"""Run a resumable, provenance-locked v4/v5/v6/v7 SUMO evaluation matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "SimpleSAC" / "evaluate_transit_v4.py"
EVALUATORS = {
    "v4": EVALUATOR,
    "v5": ROOT / "SimpleSAC" / "evaluate_transit_v5.py",
    "v6": ROOT / "SimpleSAC" / "evaluate_transit_v6.py",
    "v7": ROOT / "SimpleSAC" / "evaluate_transit_v7.py",
}
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import (
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    SUMO_BACKENDS,
    hash_files,
    load_sumo_runtime_lock,
    load_validated_scenario,
    sha256,
    sumo_runtime_fingerprint,
    sumo_version,
)
from SimpleSAC.evaluate_transit_v4 import (
    evaluation_source_paths as evaluation_source_paths_v4,
)
from SimpleSAC.evaluate_transit_v5 import (
    evaluation_source_paths as evaluation_source_paths_v5,
)
from SimpleSAC.evaluate_transit_v6 import (
    evaluation_source_paths as evaluation_source_paths_v6,
)
from SimpleSAC.evaluate_transit_v7 import (
    evaluation_source_paths as evaluation_source_paths_v7,
)


MATRIX_SCHEMA = "transit-evaluation-matrix-v4"
METHODS_SCHEMA = "transit-evaluation-methods-v4"
MATRIX_SCHEMAS = {
    "v4": MATRIX_SCHEMA,
    "v5": "transit-evaluation-matrix-v5",
    "v6": "transit-evaluation-matrix-v6",
    "v7": "transit-evaluation-matrix-v7",
}
METHODS_SCHEMAS = {
    "v4": METHODS_SCHEMA,
    "v5": "transit-evaluation-methods-v5",
    "v6": "transit-evaluation-methods-v6",
    "v7": "transit-evaluation-methods-v7",
}
RESULT_SCHEMAS = {
    "v4": "transit-evaluation-v4",
    "v5": "transit-evaluation-v5",
    "v6": "transit-evaluation-v6",
    "v7": "transit-evaluation-v7",
}
ENVIRONMENT_DESIGN_SCHEMA = "transit-evaluation-environment-design-v1"
MATRIX_PARTITION_SCHEMA = "transit-evaluation-matrix-partition-v7-v1"
MATRIX_PARTITION_ASSIGNMENT = "sorted-portable-job-key-round-robin-v1"
WORKER_MEMORY_BYTES = 1536 * 1024**2
CPU_WORKER_FRACTION_NUMERATOR = 2
CPU_WORKER_FRACTION_DENOMINATOR = 3
V7_POLICY_SELECTION_RULE = (
    "completed final actor from every registered training seed; no held-out "
    "evaluation outcome used for checkpoint or hyperparameter selection"
)
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
POLICY_TYPES = {"checkpoint", "zero", "headway-balance", "myopic-welfare"}


def signature(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _physical_core_count() -> int:
    try:
        records = Path("/proc/cpuinfo").read_text().strip().split("\n\n")
        cores = set()
        for record in records:
            fields = {
                key.strip(): value.strip()
                for line in record.splitlines()
                if ":" in line
                for key, value in (line.split(":", 1),)
            }
            cores.add(
                (
                    fields.get("physical id", "0"),
                    fields.get("core id", fields.get("processor", "0")),
                )
            )
        if cores:
            return len(cores)
    except OSError:
        pass
    return max(1, (os.cpu_count() or 1) // 2)


def _available_memory_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 2 * 1024**3


def auto_worker_count(
    job_count: int,
    *,
    physical_cores: int | None = None,
    available_memory_bytes: int | None = None,
) -> tuple[int, dict]:
    """Select isolated SUMO workers with CPU and memory contention headroom."""

    if job_count <= 0:
        raise ValueError("cannot size workers for an empty matrix partition")
    cores = int(physical_cores or _physical_core_count())
    memory = int(available_memory_bytes or _available_memory_bytes())
    if cores <= 0 or memory <= 0:
        raise ValueError("worker capacity inputs must be positive")
    cpu_limit = max(
        1,
        cores * CPU_WORKER_FRACTION_NUMERATOR // CPU_WORKER_FRACTION_DENOMINATOR,
    )
    memory_limit = max(1, memory // WORKER_MEMORY_BYTES)
    selected = min(job_count, cpu_limit, memory_limit)
    return selected, {
        "mode": "automatic-physical-core-and-memory",
        "physical_cores": cores,
        "available_memory_bytes": memory,
        "cpu_worker_fraction": (
            f"{CPU_WORKER_FRACTION_NUMERATOR}/"
            f"{CPU_WORKER_FRACTION_DENOMINATOR}"
        ),
        "reserved_physical_cores": cores - cpu_limit,
        "worker_memory_budget_bytes": WORKER_MEMORY_BYTES,
        "cpu_worker_limit": cpu_limit,
        "memory_worker_limit": memory_limit,
        "selected_workers": selected,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mdp-version", choices=("v4", "v5", "v6", "v7"), default="v4"
    )
    parser.add_argument("--methods-manifest", type=Path, required=True)
    parser.add_argument("--scenario-manifests", type=Path, nargs="+")
    parser.add_argument("--sumo-seeds", type=int, nargs="+")
    parser.add_argument(
        "--environment-design",
        type=Path,
        help=(
            "explicit paired environment blocks; mutually exclusive with "
            "--scenario-manifests/--sumo-seeds"
        ),
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "concurrent isolated SUMO processes; 0 selects a physical-core and "
            "available-memory-aware value"
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="number of deterministic matrix partitions",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="zero-based deterministic partition to execute",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--sumo-backend",
        choices=tuple(sorted(SUMO_BACKENDS)),
        default=FORMAL_V7_SUMO_BACKEND,
    )
    parser.add_argument(
        "--expected-sumo-runtime-version",
        default=FORMAL_V7_SUMO_VERSION,
    )
    parser.add_argument(
        "--sumo-runtime-lock",
        type=Path,
        help="required hash-locked runtime evidence for formal v7 evaluation",
    )
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-sim-time-seconds", type=int, default=0)
    parser.add_argument("--save-event-logs", action="store_true")
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


def load_methods(path: Path, methods_schema: str = METHODS_SCHEMA) -> list[dict]:
    manifest_path = path.expanduser().resolve()
    payload = json.loads(manifest_path.read_text())
    if payload.get("schema_version") != methods_schema:
        raise ValueError(f"methods manifest must use {methods_schema}")
    if methods_schema == METHODS_SCHEMAS["v7"] and not (
        payload.get("study_stage") == "formal"
        and payload.get("selection_rule") == V7_POLICY_SELECTION_RULE
    ):
        raise ValueError("v7 methods manifest lacks the formal selection freeze")
    raw_methods = payload.get("methods")
    if not isinstance(raw_methods, list) or not raw_methods:
        raise ValueError("methods manifest must contain a non-empty methods list")
    methods = []
    names = set()
    for raw in raw_methods:
        if not isinstance(raw, dict):
            raise ValueError("each method must be an object")
        name = raw.get("name")
        algorithm = raw.get("algorithm", name)
        policy_type = raw.get("type")
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
            raise ValueError(f"unsafe or invalid method name {name!r}")
        if not isinstance(algorithm, str) or not SAFE_NAME.fullmatch(algorithm):
            raise ValueError(f"unsafe or invalid algorithm name {algorithm!r}")
        if name in names:
            raise ValueError(f"duplicate method name {name!r}")
        names.add(name)
        if policy_type not in POLICY_TYPES:
            raise ValueError(f"unsupported policy type {policy_type!r}")
        training_seed = raw.get("training_seed")
        if training_seed is not None and (
            isinstance(training_seed, bool) or not isinstance(training_seed, int)
        ):
            raise ValueError(f"training_seed for {name!r} must be an integer or null")
        method = {
            "name": name,
            "algorithm": algorithm,
            "training_seed": training_seed,
            "type": policy_type,
        }
        if policy_type == "checkpoint":
            if methods_schema == METHODS_SCHEMAS["v7"] and "algorithm" not in raw:
                raise ValueError(
                    f"v7 checkpoint method {name!r} needs an explicit algorithm"
                )
            if methods_schema == METHODS_SCHEMAS["v7"] and training_seed is None:
                raise ValueError(
                    f"v7 checkpoint method {name!r} needs a training seed"
                )
            if methods_schema == METHODS_SCHEMAS["v7"]:
                provenance = {
                    key: raw.get(key)
                    for key in (
                        "training_run_signature",
                        "training_admission_sha256",
                        "training_checkpoint_sha256",
                        "training_plan_sha256",
                    )
                }
                malformed = [
                    key
                    for key, value in provenance.items()
                    if not isinstance(value, str) or not SHA256.fullmatch(value)
                ]
                if malformed:
                    raise ValueError(
                        f"v7 checkpoint method {name!r} has malformed formal "
                        f"provenance: {malformed}"
                    )
                method.update(provenance)
            checkpoint_value = raw.get("checkpoint")
            if not isinstance(checkpoint_value, str):
                raise ValueError(f"checkpoint method {name!r} lacks a path")
            checkpoint = Path(checkpoint_value).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = manifest_path.parent / checkpoint
            checkpoint = checkpoint.resolve()
            method.update(
                {"checkpoint": checkpoint, "checkpoint_sha256": sha256(checkpoint)}
            )
        elif policy_type == "headway-balance":
            gain = float(raw.get("gain", -1.0))
            if not 0.0 <= gain < float("inf"):
                raise ValueError(f"invalid headway gain for {name!r}")
            method["gain"] = gain
        methods.append(method)
    return methods


def load_scenarios(paths: list[Path]) -> list[dict]:
    scenarios = []
    tags = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        _, evidence = load_validated_scenario(resolved)
        tag = resolved.parent.name
        if tag in tags:
            raise ValueError(f"duplicate scenario tag {tag!r}")
        tags.add(tag)
        scenarios.append(
            {
                "path": resolved,
                "tag": tag,
                "manifest_sha256": evidence["manifest_sha256"],
                "runtime_input_sha256": evidence["runtime_input_sha256"],
                "demand_scale": float(evidence["demand_scale"]),
                "selection_seed": evidence.get("selection_seed"),
                "depart_jitter_seconds": float(
                    evidence.get("depart_jitter_seconds", 0.0)
                ),
            }
        )
    return scenarios


def load_environment_design(
    path: Path, *, expected_mdp_version: str | None = None
) -> tuple[list[dict], dict]:
    """Load explicit independent environment blocks for paired evaluation."""

    design_path = path.expanduser().resolve()
    payload = json.loads(design_path.read_text())
    if payload.get("schema_version") != ENVIRONMENT_DESIGN_SCHEMA:
        raise ValueError(
            f"environment design must use {ENVIRONMENT_DESIGN_SCHEMA}"
        )
    if expected_mdp_version == "v7":
        disjointness = payload.get("training_disjointness", {})
        generator = Path(payload.get("generator", "")).expanduser()
        if not generator.is_absolute():
            generator = design_path.parent / generator
        generator = generator.resolve()
        if (
            payload.get("mdp_version") != "v7"
            or payload.get("study_stage") != "formal"
            or disjointness.get("verified_disjoint") is not True
            or disjointness.get("scenario_manifest_hash_overlap") != []
            or disjointness.get("scenario_runtime_hash_overlap") != []
            or not generator.is_file()
            or payload.get("generator_sha256") != sha256(generator)
        ):
            raise ValueError(
                "formal v7 evaluation requires the current hash-disjoint v7 "
                "environment design"
            )
    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError("environment design must contain non-empty blocks")

    cells = []
    block_names = set()
    sumo_seeds = set()
    realization_seeds = set()
    expected_scales = None
    serializable_blocks = []
    for raw_block in raw_blocks:
        if not isinstance(raw_block, dict):
            raise ValueError("each environment block must be an object")
        name = raw_block.get("name")
        if not isinstance(name, str) or not SAFE_NAME.fullmatch(name):
            raise ValueError(f"unsafe or invalid environment block name {name!r}")
        if name in block_names:
            raise ValueError(f"duplicate environment block name {name!r}")
        block_names.add(name)
        sumo_seed_value = raw_block.get("sumo_seed")
        if isinstance(sumo_seed_value, bool) or not isinstance(sumo_seed_value, int):
            raise ValueError(f"SUMO seed for block {name!r} must be an integer")
        if sumo_seed_value in sumo_seeds:
            raise ValueError(
                f"SUMO seed {sumo_seed_value} is reused across environment blocks"
            )
        sumo_seeds.add(sumo_seed_value)
        raw_manifests = raw_block.get("scenario_manifests")
        if not isinstance(raw_manifests, list) or not raw_manifests or not all(
            isinstance(value, str) for value in raw_manifests
        ):
            raise ValueError(
                f"block {name!r} must contain scenario manifest paths"
            )
        manifest_paths = []
        for value in raw_manifests:
            manifest_path = Path(value).expanduser()
            if not manifest_path.is_absolute():
                manifest_path = design_path.parent / manifest_path
            manifest_paths.append(manifest_path.resolve())
        scenarios = load_scenarios(manifest_paths)
        scales = tuple(sorted(scenario["demand_scale"] for scenario in scenarios))
        if len(set(scales)) != len(scales):
            raise ValueError(
                f"block {name!r} contains duplicate demand-scale conditions"
            )
        if expected_scales is None:
            expected_scales = scales
        elif scales != expected_scales:
            raise ValueError(
                f"block {name!r} has demand scales {scales}; expected {expected_scales}"
            )
        selection_seeds = {scenario["selection_seed"] for scenario in scenarios}
        if len(selection_seeds) != 1 or None in selection_seeds:
            raise ValueError(
                f"block {name!r} must use one declared demand-realization seed"
            )
        realization_seed = next(iter(selection_seeds))
        if realization_seed in realization_seeds:
            raise ValueError(
                f"demand-realization seed {realization_seed} is reused across blocks"
            )
        realization_seeds.add(realization_seed)
        jitter_values = {
            scenario["depart_jitter_seconds"] for scenario in scenarios
        }
        if len(jitter_values) != 1:
            raise ValueError(
                f"block {name!r} mixes departure-jitter specifications"
            )
        for scenario in scenarios:
            cells.append(
                {
                    "environment_block": name,
                    "sumo_seed": sumo_seed_value,
                    "scenario": scenario,
                }
            )
        serializable_blocks.append(
            {
                "name": name,
                "sumo_seed": sumo_seed_value,
                "demand_realization_seed": realization_seed,
                "depart_jitter_seconds": next(iter(jitter_values)),
                "demand_scales": list(scales),
                "scenario_manifests": [str(path) for path in manifest_paths],
            }
        )
    evidence = {
        "mode": "explicit-blocks",
        "path": str(design_path),
        "sha256": sha256(design_path),
        "schema_version": ENVIRONMENT_DESIGN_SCHEMA,
        "blocks": serializable_blocks,
    }
    if expected_mdp_version == "v7":
        if (
            payload.get("environment_cells") != len(cells)
            or payload.get("independent_realization_blocks") != len(raw_blocks)
        ):
            raise ValueError("v7 environment-design coverage declaration drifted")
        evidence.update(
            {
                "mdp_version": "v7",
                "study_stage": "formal",
                "training_disjointness": payload["training_disjointness"],
                "generator_sha256": payload["generator_sha256"],
            }
        )
    return cells, evidence


def load_environment_cells(args: argparse.Namespace) -> tuple[list[dict], dict]:
    if args.environment_design is not None:
        if args.scenario_manifests is not None or args.sumo_seeds is not None:
            raise ValueError(
                "--environment-design is mutually exclusive with the Cartesian "
                "scenario/seed arguments"
            )
        return load_environment_design(
            args.environment_design,
            expected_mdp_version=getattr(args, "mdp_version", "v4"),
        )
    if not args.scenario_manifests or not args.sumo_seeds:
        raise ValueError(
            "provide --environment-design or both --scenario-manifests and --sumo-seeds"
        )
    if len(set(args.sumo_seeds)) != len(args.sumo_seeds):
        raise ValueError("Cartesian SUMO seeds must be unique")
    scenarios = load_scenarios(args.scenario_manifests)
    cells = [
        {
            "environment_block": f"sumo-seed-{sumo_seed_value}",
            "sumo_seed": int(sumo_seed_value),
            "scenario": scenario,
        }
        for sumo_seed_value in args.sumo_seeds
        for scenario in scenarios
    ]
    return cells, {
        "mode": "cartesian",
        "scenario_manifests": [str(scenario["path"]) for scenario in scenarios],
        "sumo_seeds": [int(seed) for seed in args.sumo_seeds],
    }


def _method_arguments(method: dict) -> list[str]:
    if method["type"] == "checkpoint":
        return ["--checkpoint", str(method["checkpoint"])]
    if method["type"] == "zero":
        return ["--zero-hold"]
    if method["type"] == "headway-balance":
        return ["--headway-balance-gain", str(method["gain"])]
    if method["type"] == "myopic-welfare":
        return ["--myopic-welfare"]
    raise ValueError(f"unsupported method type {method['type']!r}")


def _job_command(job: dict) -> list[str]:
    command = [
        sys.executable,
        str(job.get("evaluator", EVALUATOR)),
        *_method_arguments(job["method"]),
        "--method-name",
        job["method"]["name"],
        "--sumo-seed",
        str(job["sumo_seed"]),
        "--scenario-manifest",
        str(job["scenario"]["path"]),
        "--dataset",
        str(job["dataset"]),
        "--output",
        str(job["output"]),
        "--job-signature",
        job["signature"],
        "--device",
        job["device"],
        "--max-events",
        str(job["max_events"]),
        "--max-sim-time-seconds",
        str(job["max_sim_time_seconds"]),
        "--sumo-dir",
        str(job["sumo_dir"]),
        "--edge-xml",
        str(job["edge_xml"]),
        "--quiet",
    ]
    if job.get("mdp_version") == "v7":
        command.extend(
            (
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
                "--verified-dataset-sha256",
                job["dataset_sha256"],
                "--verified-dataset-size-bytes",
                str(job["dataset_size_bytes"]),
                "--verified-dataset-mtime-ns",
                str(job["dataset_mtime_ns"]),
            )
        )
        if job["method"]["type"] == "checkpoint":
            command.extend(
                (
                    "--expected-algorithm",
                    job["method"]["algorithm"],
                    "--expected-training-seed",
                    str(job["method"]["training_seed"]),
                    "--expected-training-run-signature",
                    job["method"]["training_run_signature"],
                    "--expected-training-admission-sha256",
                    job["method"]["training_admission_sha256"],
                    "--expected-training-checkpoint-sha256",
                    job["method"]["training_checkpoint_sha256"],
                )
            )
    if job["event_log"] is not None:
        command.extend(("--event-log", str(job["event_log"])))
    return command


def _is_complete(job: dict) -> bool:
    output = job["output"]
    if not output.is_file():
        return False
    try:
        payload = json.loads(output.read_text())
        if not (
            payload.get("schema_version")
            == job.get("result_schema", "transit-evaluation-v4")
            and payload.get("evaluation_job", {}).get("signature")
            == job["signature"]
            and payload.get("dataset", {}).get("sha256")
            == job["dataset_sha256"]
            and payload.get("source_sha256") == job["source_sha256"]
            and payload.get("runtime", {}).get("sumo_version")
            == job["sumo_version"]
            and payload.get("scenario", {}).get("runtime_input_sha256")
            == job["scenario"]["runtime_input_sha256"]
        ):
            return False
        if job.get("mdp_version") in {"v6", "v7"} and not (
            job["max_events"] or job["max_sim_time_seconds"]
        ):
            coverage = payload.get("passenger_welfare", {}).get(
                "chronological_coverage_errors", {}
            )
            count_audit = payload.get("ledger_audit", {}).get(
                "count_source_audit", {}
            )
            service = payload.get("service", {})
            episode = payload.get("episode", {})
            service_audit = payload.get("bus_service_semantics_audit", {})
            if not (
                episode.get("observed_end_time_seconds") == 24_000
                and episode.get("terminal_transitions") == 1
                and episode.get("control_decisions") == episode.get("transitions")
                and coverage
                and all(abs(float(value)) <= 1e-9 for value in coverage.values())
                and count_audit.get("max_waiting_count_error") == 0
                and count_audit.get("max_in_vehicle_count_error") == 0
                and count_audit.get("subscription_fallback_queries") == 0
                and service.get("completed_trips") == service.get("departed_trips")
                and service.get("active_trips_at_horizon") == 0
                and (
                    job.get("mdp_version") != "v7"
                    or service_audit.get("passed") is True
                )
            ):
                return False
        if job.get("mdp_version") == "v7":
            dataset = payload.get("dataset", {})
            runtime = payload.get("runtime", {})
            if not (
                dataset.get("size_bytes") == job["dataset_size_bytes"]
                and dataset.get("mtime_ns") == job["dataset_mtime_ns"]
                and dataset.get("verification")
                == "parent-matrix-sha256-plus-child-stat-lock"
                and runtime.get("sumo_backend") == job["sumo_backend"]
                and runtime.get("sumo_runtime_signature")
                == job["sumo_runtime_signature"]
                and runtime.get("sumo_runtime_lock", {}).get("sha256")
                == job["sumo_runtime_lock_sha256"]
                and runtime.get("sumo_runtime_fingerprint", {}).get(
                    "portable_identity"
                )
                == job["sumo_runtime_fingerprint"]["portable_identity"]
            ):
                return False
            method = job["method"]
            if method["type"] == "checkpoint":
                policy = payload.get("policy", {})
                training = policy.get("training_metadata", {})
                if not (
                    policy.get("sha256") == method.get("checkpoint_sha256")
                    and training.get("run_signature")
                    == method.get("training_run_signature")
                    and training.get("training_checkpoint_sha256")
                    == method.get("training_checkpoint_sha256")
                    and training.get("training_admission", {}).get("sha256")
                    == method.get("training_admission_sha256")
                ):
                    return False
        event_log = job["event_log"]
        if event_log is None:
            return payload.get("event_log") is None
        return (
            event_log.is_file()
            and payload.get("event_log", {}).get("sha256") == sha256(event_log)
        )
    except Exception:
        return False


def _quarantine(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    os.replace(path, path.with_name(path.name + f".invalid-{timestamp}"))


def _run_job(job: dict) -> dict:
    if _is_complete(job):
        return {"status": "skipped", "job": job}
    _quarantine(job["output"])
    _quarantine(job["event_log"])
    job["output"].parent.mkdir(parents=True, exist_ok=True)
    log_path = job["output"].with_suffix(".log")
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
    method = dict(job["method"])
    if "checkpoint" in method:
        method["checkpoint"] = str(method["checkpoint"])
    return {
        **job,
        "method": method,
        "scenario": {**job["scenario"], "path": str(job["scenario"]["path"])},
        "dataset": str(job["dataset"]),
        "sumo_dir": str(job["sumo_dir"]),
        "edge_xml": str(job["edge_xml"]),
        "output": str(job["output"]),
        "event_log": str(job["event_log"]) if job["event_log"] else None,
        "evaluator": str(job.get("evaluator", EVALUATOR)),
        "sumo_runtime_lock": (
            str(job["sumo_runtime_lock"])
            if job.get("sumo_runtime_lock") is not None
            else None
        ),
    }


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _archive_existing_manifest(path: Path) -> Path | None:
    """Move an earlier matrix attempt into immutable timestamped history."""

    if not path.is_file():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    digest = sha256(path)
    archive_root = path.parent / "matrix_manifest_history"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / f"matrix_manifest.{timestamp}.{digest[:16]}.json"
    os.replace(path, archive)
    return archive


def build_jobs(args: argparse.Namespace) -> tuple[list[dict], dict]:
    mdp_version = getattr(args, "mdp_version", "v4")
    evaluator = EVALUATORS[mdp_version]
    result_schema = RESULT_SCHEMAS[mdp_version]
    source_paths = {
        "v4": evaluation_source_paths_v4,
        "v5": evaluation_source_paths_v5,
        "v6": evaluation_source_paths_v6,
        "v7": evaluation_source_paths_v7,
    }[mdp_version]
    methods_manifest = args.methods_manifest.expanduser().resolve()
    methods = load_methods(methods_manifest, METHODS_SCHEMAS[mdp_version])
    environment_cells, environment_design = load_environment_cells(args)
    dataset = args.dataset.expanduser().resolve()
    dataset_stat_before = dataset.stat()
    dataset_hash = sha256(dataset)
    dataset_stat_after = dataset.stat()
    if (
        dataset_stat_before.st_size,
        dataset_stat_before.st_mtime_ns,
    ) != (
        dataset_stat_after.st_size,
        dataset_stat_after.st_mtime_ns,
    ):
        raise RuntimeError("evaluation dataset changed while it was hashed")
    dataset_size_bytes = int(dataset_stat_after.st_size)
    dataset_mtime_ns = int(dataset_stat_after.st_mtime_ns)
    output_root = args.output_root.expanduser().resolve()
    sumo_dir = args.sumo_dir.expanduser().resolve()
    edge_xml = args.edge_xml.expanduser().resolve()
    source_sha256 = hash_files(source_paths(sumo_dir, edge_xml))
    if mdp_version == "v7":
        sumo_backend = getattr(args, "sumo_backend", FORMAL_V7_SUMO_BACKEND)
        expected_sumo_runtime_version = getattr(
            args, "expected_sumo_runtime_version", FORMAL_V7_SUMO_VERSION
        )
        if sumo_backend != FORMAL_V7_SUMO_BACKEND:
            raise ValueError(
                f"formal v7 requires {FORMAL_V7_SUMO_BACKEND}, not {sumo_backend}"
            )
        if expected_sumo_runtime_version != FORMAL_V7_SUMO_VERSION:
            raise ValueError(
                f"formal v7 requires SUMO {FORMAL_V7_SUMO_VERSION}"
            )
        raw_runtime_lock = getattr(args, "sumo_runtime_lock", None)
        if raw_runtime_lock is None:
            raise ValueError("formal v7 requires --sumo-runtime-lock")
        sumo_runtime_lock = raw_runtime_lock.expanduser().resolve()
        runtime_lock, runtime_lock_identity = load_sumo_runtime_lock(
            sumo_runtime_lock
        )
        active_sumo_runtime = sumo_runtime_fingerprint(
            sumo_backend,
            expected_version=expected_sumo_runtime_version,
        )
        active_sumo_version = active_sumo_runtime["runtime_version"]
        active_sumo_runtime_signature = active_sumo_runtime["portable_signature"]
        if (
            runtime_lock_identity["runtime_signature"]
            != active_sumo_runtime_signature
            or runtime_lock["runtime_fingerprint"]["portable_identity"]
            != active_sumo_runtime["portable_identity"]
        ):
            raise RuntimeError("active SUMO runtime differs from its formal lock")
    else:
        sumo_backend = None
        sumo_runtime_lock = None
        runtime_lock_identity = None
        active_sumo_runtime = None
        active_sumo_runtime_signature = None
        active_sumo_version = sumo_version()
    runner_hash = sha256(Path(__file__).resolve())
    jobs = []
    for method in methods:
        for cell in environment_cells:
            scenario = cell["scenario"]
            sumo_seed_value = cell["sumo_seed"]
            output_dir = output_root / method["name"] / scenario["tag"]
            output = output_dir / f"sumo_seed{sumo_seed_value}.json"
            event_log = (
                output_dir / f"sumo_seed{sumo_seed_value}.npz"
                if args.save_event_logs
                else None
            )
            payload = {
                "mdp_version": mdp_version,
                "evaluator_sha256": sha256(evaluator),
                "runner_sha256": runner_hash,
                "source_sha256": source_sha256,
                "sumo_version": active_sumo_version,
                "sumo_backend": sumo_backend,
                "sumo_runtime_fingerprint": (
                    active_sumo_runtime["portable_identity"]
                    if active_sumo_runtime is not None
                    else None
                ),
                "sumo_runtime_signature": active_sumo_runtime_signature,
                "sumo_runtime_lock_sha256": (
                    runtime_lock_identity["sha256"]
                    if runtime_lock_identity is not None
                    else None
                ),
                "methods_manifest_sha256": sha256(methods_manifest),
                "method": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in method.items()
                },
                "scenario_manifest_sha256": scenario["manifest_sha256"],
                "scenario_runtime_input_sha256": scenario[
                    "runtime_input_sha256"
                ],
                "environment_block": cell["environment_block"],
                "sumo_seed": int(sumo_seed_value),
                "dataset_sha256": dataset_hash,
                "dataset_size_bytes": dataset_size_bytes,
                "dataset_mtime_ns": dataset_mtime_ns,
                "device": args.device,
                "max_events": args.max_events,
                "max_sim_time_seconds": args.max_sim_time_seconds,
                "save_event_logs": args.save_event_logs,
                "sumo_dir": str(sumo_dir),
                "edge_xml_sha256": sha256(edge_xml),
            }
            jobs.append(
                {
                    "method": method,
                    "mdp_version": mdp_version,
                    "evaluator": evaluator,
                    "result_schema": result_schema,
                    "scenario": scenario,
                    "environment_block": cell["environment_block"],
                    "sumo_seed": int(sumo_seed_value),
                    "dataset": dataset,
                    "dataset_sha256": dataset_hash,
                    "dataset_size_bytes": dataset_size_bytes,
                    "dataset_mtime_ns": dataset_mtime_ns,
                    "device": args.device,
                    "max_events": args.max_events,
                    "max_sim_time_seconds": args.max_sim_time_seconds,
                    "sumo_dir": sumo_dir,
                    "edge_xml": edge_xml,
                    "output": output,
                    "event_log": event_log,
                    "source_sha256": source_sha256,
                    "sumo_version": active_sumo_version,
                    "sumo_backend": sumo_backend,
                    "sumo_runtime_version": active_sumo_version,
                    "sumo_runtime_signature": active_sumo_runtime_signature,
                    "sumo_runtime_fingerprint": active_sumo_runtime,
                    "sumo_runtime_lock": sumo_runtime_lock,
                    "sumo_runtime_lock_sha256": (
                        runtime_lock_identity["sha256"]
                        if runtime_lock_identity is not None
                        else None
                    ),
                    "signature": signature(payload),
                }
            )
    metadata = {
        "schema_version": MATRIX_SCHEMAS[mdp_version],
        "mdp_version": mdp_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "methods_manifest": str(methods_manifest),
        "methods_manifest_sha256": sha256(methods_manifest),
        "environment_design": environment_design,
        "dataset": str(dataset),
        "dataset_sha256": dataset_hash,
        "dataset_size_bytes": dataset_size_bytes,
        "dataset_mtime_ns": dataset_mtime_ns,
        "evaluator": str(evaluator),
        "evaluator_sha256": sha256(evaluator),
        "runner_sha256": runner_hash,
        "source_sha256": source_sha256,
        "sumo_version": active_sumo_version,
        "sumo_backend": sumo_backend,
        "sumo_runtime_signature": active_sumo_runtime_signature,
        "sumo_runtime_fingerprint": active_sumo_runtime,
        "sumo_runtime_lock": runtime_lock_identity,
        "jobs": [_serializable_job(job) for job in jobs],
    }
    return jobs, metadata


def portable_job_identity(job: dict) -> dict:
    method = job["method"]
    return {
        "method": {
            key: method.get(key)
            for key in (
                "name",
                "algorithm",
                "training_seed",
                "type",
                "checkpoint_sha256",
                "training_run_signature",
                "training_admission_sha256",
                "training_checkpoint_sha256",
                "training_plan_sha256",
                "gain",
            )
            if key in method
        },
        "environment_block": job["environment_block"],
        "sumo_seed": int(job["sumo_seed"]),
        "scenario_manifest_sha256": job["scenario"]["manifest_sha256"],
        "scenario_runtime_content_sha256": signature(
            {
                "digests": sorted(
                    job["scenario"]["runtime_input_sha256"].values()
                )
            }
        ),
    }


def portable_job_key(job: dict) -> str:
    return signature(portable_job_identity(job))


def full_matrix_partition_signature(jobs: list[dict], metadata: dict) -> str:
    keys = sorted(portable_job_key(job) for job in jobs)
    return signature(
        {
            "schema_version": MATRIX_PARTITION_SCHEMA,
            "assignment": MATRIX_PARTITION_ASSIGNMENT,
            "mdp_version": metadata["mdp_version"],
            "dataset_sha256": metadata["dataset_sha256"],
            "methods_manifest_sha256": metadata["methods_manifest_sha256"],
            "environment_design_sha256": metadata["environment_design"].get(
                "sha256"
            ),
            "evaluator_sha256": metadata["evaluator_sha256"],
            "source_content_sha256": signature(
                {"digests": sorted(metadata["source_sha256"].values())}
            ),
            "sumo_version": metadata["sumo_version"],
            "sumo_backend": metadata.get("sumo_backend"),
            "sumo_runtime_signature": metadata.get("sumo_runtime_signature"),
            "sumo_runtime_lock_sha256": (
                metadata.get("sumo_runtime_lock", {}).get("sha256")
                if metadata.get("sumo_runtime_lock")
                else None
            ),
            "portable_job_keys": keys,
        }
    )


def partition_jobs(
    jobs: list[dict],
    metadata: dict,
    *,
    shard_count: int,
    shard_index: int,
) -> tuple[list[dict], dict]:
    """Create a host-independent, balanced partition of the complete matrix."""

    if shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if shard_count > len(jobs):
        raise ValueError("shard-count cannot exceed the number of matrix jobs")

    keyed = sorted((portable_job_key(job), job) for job in jobs)
    keys = [key for key, _ in keyed]
    if len(keys) != len(set(keys)):
        raise ValueError("portable matrix job identities are not unique")
    selected = [
        job
        for ordinal, (_, job) in enumerate(keyed)
        if ordinal % shard_count == shard_index
    ]
    selected_keys = [portable_job_key(job) for job in selected]
    full_matrix_signature = full_matrix_partition_signature(jobs, metadata)
    return selected, {
        "schema_version": MATRIX_PARTITION_SCHEMA,
        "assignment": MATRIX_PARTITION_ASSIGNMENT,
        "full_matrix_signature": full_matrix_signature,
        "shard_count": shard_count,
        "shard_index": shard_index,
        "full_job_count": len(jobs),
        "selected_job_count": len(selected),
        "selected_job_keys_sha256": signature(
            {"portable_job_keys": sorted(selected_keys)}
        ),
    }


def main() -> None:
    args = parse_args()
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid matrix shard specification")
    if args.mdp_version != "v7" and (
        args.shard_count != 1 or args.shard_index != 0
    ):
        raise ValueError("deterministic matrix partitioning is supported only for v7")
    if args.max_events < 0 or args.max_sim_time_seconds < 0:
        raise ValueError("diagnostic limits cannot be negative")
    if args.max_events and args.max_sim_time_seconds:
        raise ValueError("diagnostic limits are mutually exclusive")
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shard_suffix = (
        f".shard-{args.shard_index:04d}-of-{args.shard_count:04d}"
        if args.shard_count > 1
        else ""
    )
    lock = (output_root / f".evaluation{shard_suffix}.lock").open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(
            f"evaluation matrix already running under {output_root}"
        ) from error

    jobs, manifest = build_jobs(args)
    if args.mdp_version == "v7":
        jobs, partition = partition_jobs(
            jobs,
            manifest,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
        )
        manifest["matrix_partition"] = partition
        manifest["jobs"] = [_serializable_job(job) for job in jobs]
    if args.workers == 0:
        workers, worker_selection = auto_worker_count(len(jobs))
    else:
        workers = min(args.workers, len(jobs))
        worker_selection = {
            "mode": "explicit",
            "requested_workers": args.workers,
            "selected_workers": workers,
        }
    manifest["workers"] = workers
    manifest["worker_selection"] = worker_selection
    manifest_path = output_root / (
        f"matrix_manifest{shard_suffix}.json"
        if args.shard_count > 1
        else "matrix_manifest.json"
    )
    previous_manifest = _archive_existing_manifest(manifest_path)
    manifest["previous_manifest"] = (
        str(previous_manifest) if previous_manifest is not None else None
    )
    _write_json(manifest_path, manifest)

    failures = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_run_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            completed += 1
            job = result["job"]
            print(
                f"[{completed}/{len(jobs)}] {result['status']} "
                f"{job['method']['name']} {job['scenario']['tag']} "
                f"seed={job['sumo_seed']}",
                flush=True,
            )
            if result["status"] == "failed":
                failures.append(result)
    fcntl.flock(lock, fcntl.LOCK_UN)
    lock.close()
    if failures:
        details = "\n".join(
            f"{failure['job']['output']}: {failure['log']}"
            for failure in failures
        )
        raise RuntimeError(f"{len(failures)} evaluation jobs failed:\n{details}")


if __name__ == "__main__":
    main()
