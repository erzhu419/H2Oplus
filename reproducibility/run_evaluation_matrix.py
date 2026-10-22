#!/usr/bin/env python3
"""Run a provenance-locked SUMO evaluation matrix in parallel processes."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "SimpleSAC" / "evaluate_transit_v3.py"
DEFAULT_DATASET = ROOT / "bus_h2o" / "datasets_v3" / "transit_smdp_v3.h5"
DEFAULT_SUMO_DIR = ROOT.parent / "SUMO_ruiguang" / "online_control"
DEFAULT_EDGE_XML = ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml"
MATRIX_SCHEMA = "transit-evaluation-matrix-v2"
JOB_SCHEMA = "transit-evaluation-job-v1"
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.Lock()


@dataclass(frozen=True)
class PolicySpec:
    name: str
    arguments: tuple[str, ...]
    identity: dict


@dataclass(frozen=True)
class EvaluationJob:
    policy: PolicySpec
    sumo_seed: int
    scenario_manifest: Path
    output: Path
    event_log: Path | None
    signature: str
    identity: dict


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_hash(file_hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, file_hash in sorted(file_hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def json_hash(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assignment(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name cannot be empty")
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"checkpoint does not exist: {path}")
    return name, path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=_assignment, default=[])
    parser.add_argument("--include-zero-hold", action="store_true")
    parser.add_argument("--constant-hold", action="append", type=float, default=[])
    parser.add_argument("--headway-gain", action="append", type=float, default=[])
    parser.add_argument("--sumo-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--scenario-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--sumo-dir", type=Path, default=DEFAULT_SUMO_DIR)
    parser.add_argument("--edge-xml", type=Path, default=DEFAULT_EDGE_XML)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Concurrent SUMO processes; 0 selects a CPU/RAM-aware value (up to 8)",
    )
    diagnostic_limit = parser.add_mutually_exclusive_group()
    diagnostic_limit.add_argument("--max-events", type=int, default=0)
    diagnostic_limit.add_argument("--max-sim-time-seconds", type=int, default=0)
    parser.add_argument("--save-event-logs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


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


def auto_worker_count() -> int:
    # A measured canonical evaluator peaks near 0.9 GiB.  Budget 1.5 GiB per
    # process and cap at eight to retain memory and desktop responsiveness.
    memory_limit = max(1, _available_memory_bytes() // (1536 * 1024**2))
    return max(1, min(8, _physical_core_count(), memory_limit))


def _hash_paths(paths: set[Path]) -> dict[str, str]:
    resolved = sorted({path.expanduser().resolve() for path in paths})
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"evaluation provenance input is missing: {missing[0]}")
    return {str(path): sha256(path) for path in resolved}


def implementation_hashes(sumo_dir: Path) -> dict[str, str]:
    paths = {
        EVALUATOR,
        ROOT / "bus_h2o" / "mdp_contract.py",
        ROOT / "bus_h2o" / "common" / "data_utils.py",
        ROOT / "bus_h2o" / "envs" / "all_lines_sumo_env.py",
        ROOT / "bus_h2o" / "envs" / "sumo_gym_env.py",
        ROOT / "bus_h2o" / "sumo_env" / "rl_bridge.py",
        ROOT / "bus_h2o" / "sumo_env" / "rl_env.py",
        ROOT / "bus_h2o" / "sumo_env" / "case" / "d_8_compute_running_time.py",
        ROOT / "bus_h2o" / "sumo_env" / "case" / "e_8_gurobi_test_considerbusnum_V3.py",
        ROOT / "bus_h2o" / "sumo_env" / "case" / "f_8_create_obj.py",
        ROOT / "SimpleSAC" / "transit_v3" / "evaluation.py",
        ROOT / "SimpleSAC" / "transit_v3" / "networks.py",
        ROOT / "SimpleSAC" / "transit_v3" / "trainer.py",
    }
    paths.update((sumo_dir / "sim_obj").glob("*.py"))
    return _hash_paths(paths)


def _sumo_config_inputs(config_path: Path) -> set[Path]:
    config_path = config_path.resolve()
    root = ET.parse(config_path).getroot()
    input_section = root.find("input")
    paths = {config_path}
    if input_section is None:
        return paths
    for option in input_section:
        for raw_path in option.get("value", "").split(","):
            raw_path = raw_path.strip()
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = config_path.parent / path
            paths.add(path.resolve())
    return paths


def scenario_input_hashes(
    manifest_path: Path,
    sumo_dir: Path,
    observation_edge_xml: Path,
) -> dict[str, str]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "transit-passenger-demand-v1":
        raise ValueError(f"unsupported scenario manifest: {manifest_path}")
    config_path = Path(manifest["sumo_config"]).resolve()
    paths = _sumo_config_inputs(config_path)
    paths.update(
        {
            manifest_path,
            Path(manifest["passenger_objects"]).resolve(),
            observation_edge_xml.resolve(),
            sumo_dir / "intersection_delay" / "a_sorted_busline_edge.xml",
            sumo_dir / "initialize_obj" / "save_obj_lane.add.xml",
            sumo_dir / "initialize_obj" / "save_obj_stop.add.xml",
            sumo_dir / "initialize_obj" / "save_obj_signal.add.xml",
            sumo_dir / "initialize_obj" / "save_obj_line.add.xml",
            sumo_dir / "initialize_obj" / "save_obj_bus.add.xml",
        }
    )
    hashes = _hash_paths(paths)
    for raw_path, expected in manifest.get("scenario_sha256", {}).items():
        actual = hashes.get(str(Path(raw_path).resolve()))
        if actual != expected:
            raise ValueError(
                f"scenario hash mismatch for {raw_path}: expected={expected}, actual={actual}"
            )
    return hashes


def _sumo_version() -> str:
    completed = subprocess.run(
        ["sumo", "--version"], capture_output=True, text=True, check=True
    )
    return completed.stdout.splitlines()[0]


def _valid_output(
    job: EvaluationJob,
    max_events: int,
    max_sim_time_seconds: int,
) -> tuple[dict | None, list[str]]:
    reasons = []
    try:
        payload = json.loads(job.output.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"unreadable output: {error}"]
    if payload.get("schema_version") != "transit-evaluation-v3":
        reasons.append("wrong evaluation schema")
    if payload.get("evaluation_job", {}).get("signature") != job.signature:
        reasons.append("job signature mismatch")
    if payload.get("evaluation_job", {}).get("max_events") != max_events:
        reasons.append("max-events mismatch")
    if (
        payload.get("evaluation_job", {}).get("max_sim_time_seconds")
        != max_sim_time_seconds
    ):
        reasons.append("max-sim-time-seconds mismatch")
    if payload.get("method") != job.policy.name:
        reasons.append("policy name mismatch")
    if payload.get("sumo_seed") != job.sumo_seed:
        reasons.append("SUMO seed mismatch")
    if max_events == 0 and max_sim_time_seconds == 0:
        episode = payload.get("episode", {})
        if episode.get("stopped_by_event_limit", True) or episode.get(
            "stopped_by_time_limit", True
        ):
            reasons.append("formal episode is partial")
    if job.event_log is not None and not zipfile.is_zipfile(job.event_log):
        reasons.append("event log is missing or invalid")
    return (payload if not reasons else None), reasons


def _command(job: EvaluationJob, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(EVALUATOR),
        *job.policy.arguments,
        "--method-name",
        job.policy.name,
        "--sumo-seed",
        str(job.sumo_seed),
        "--device",
        "cpu",
        "--scenario-manifest",
        str(job.scenario_manifest),
        "--dataset",
        str(args.dataset),
        "--sumo-dir",
        str(args.sumo_dir),
        "--edge-xml",
        str(args.edge_xml),
        "--job-signature",
        job.signature,
        "--quiet",
        "--output",
        str(job.output),
    ]
    if args.max_events:
        command.extend(("--max-events", str(args.max_events)))
    if args.max_sim_time_seconds:
        command.extend(("--max-sim-time-seconds", str(args.max_sim_time_seconds)))
    if job.event_log is not None:
        command.extend(("--event-log", str(job.event_log)))
    return command


def _run_job(job: EvaluationJob, args: argparse.Namespace) -> dict:
    existing, invalid_reasons = _valid_output(
        job, args.max_events, args.max_sim_time_seconds
    )
    if existing is not None and not args.overwrite:
        return {
            "status": "skipped",
            "output": str(job.output),
            "policy": job.policy.name,
            "sumo_seed": job.sumo_seed,
            "scenario_manifest": str(job.scenario_manifest),
            "job_signature": job.signature,
        }

    job.output.parent.mkdir(parents=True, exist_ok=True)
    command = _command(job, args)
    log_path = job.output.with_suffix(".log")
    child_environment = os.environ.copy()
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        child_environment[variable] = "1"
    child_environment["CUDA_VISIBLE_DEVICES"] = ""
    started = time.perf_counter()
    process = None
    with log_path.open("w") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=child_environment,
                start_new_session=True,
            )
            with _ACTIVE_LOCK:
                _ACTIVE_PROCESSES.add(process)
            returncode = process.wait()
        finally:
            if process is not None:
                with _ACTIVE_LOCK:
                    _ACTIVE_PROCESSES.discard(process)
    elapsed = time.perf_counter() - started
    _, validation_reasons = _valid_output(
        job, args.max_events, args.max_sim_time_seconds
    )
    status = "completed" if returncode == 0 and not validation_reasons else "failed"
    return {
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed,
        "output": str(job.output),
        "log": str(log_path),
        "policy": job.policy.name,
        "sumo_seed": job.sumo_seed,
        "scenario_manifest": str(job.scenario_manifest),
        "job_signature": job.signature,
        "replaced_output_reasons": invalid_reasons if job.output.exists() else [],
        "validation_errors": validation_reasons,
        "command": command,
    }


def _terminate_active_processes() -> None:
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE_PROCESSES)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 5.0
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


@contextmanager
def _matrix_lock(output_root: Path):
    lock_path = output_root / ".matrix.lock"
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another matrix runner holds {lock_path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _policies(args: argparse.Namespace) -> list[PolicySpec]:
    policies = [
        PolicySpec(
            name,
            ("--checkpoint", str(path)),
            {"type": "checkpoint", "path": str(path), "sha256": sha256(path)},
        )
        for name, path in args.checkpoint
    ]
    if args.include_zero_hold:
        policies.append(PolicySpec("zero-hold", ("--zero-hold",), {"type": "zero-hold"}))
    policies.extend(
        PolicySpec(
            f"constant-hold-{hold:g}s",
            ("--constant-hold-seconds", str(hold)),
            {"type": "constant-hold", "hold_seconds": hold},
        )
        for hold in args.constant_hold
    )
    policies.extend(
        PolicySpec(
            f"headway-balance-gain-{gain:g}",
            ("--headway-balance-gain", str(gain)),
            {"type": "two-sided-headway-balance", "gain": gain},
        )
        for gain in args.headway_gain
    )
    if not policies:
        raise ValueError("at least one policy must be selected")
    names = [_safe_name(policy.name) for policy in policies]
    if len(names) != len(set(names)):
        raise ValueError("policy names are not unique after filesystem sanitization")
    return policies


def main() -> None:
    args = parse_args()
    if args.max_events < 0:
        raise ValueError("max-events cannot be negative")
    if args.max_sim_time_seconds < 0:
        raise ValueError("max-sim-time-seconds cannot be negative")
    args.dataset = args.dataset.expanduser().resolve()
    args.sumo_dir = args.sumo_dir.expanduser().resolve()
    args.edge_xml = args.edge_xml.expanduser().resolve()
    for required in (args.dataset, args.edge_xml):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not args.sumo_dir.is_dir():
        raise FileNotFoundError(args.sumo_dir)
    if len(args.sumo_seeds) != len(set(args.sumo_seeds)):
        raise ValueError("SUMO seeds must be unique")

    workers = auto_worker_count() if args.workers == 0 else args.workers
    if workers <= 0:
        raise ValueError("workers must be positive")
    policies = _policies(args)
    manifests = [path.expanduser().resolve() for path in args.scenario_manifests]
    if len(manifests) != len(set(manifests)):
        raise ValueError("scenario manifests must be unique")

    source_hashes = implementation_hashes(args.sumo_dir)
    scenario_hashes = {
        str(path): scenario_input_hashes(path, args.sumo_dir, args.edge_xml)
        for path in manifests
    }
    dataset_hash = sha256(args.dataset)
    sumo_version = _sumo_version()
    source_aggregate = aggregate_hash(source_hashes)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = []
    seen_outputs = set()
    for policy in policies:
        for scenario_manifest in manifests:
            scenario_name = _safe_name(scenario_manifest.parent.name)
            scenario_files = scenario_hashes[str(scenario_manifest)]
            for sumo_seed in args.sumo_seeds:
                identity = {
                    "schema_version": JOB_SCHEMA,
                    "method": policy.name,
                    "policy": policy.identity,
                    "sumo_seed": int(sumo_seed),
                    "scenario_manifest": str(scenario_manifest),
                    "scenario_input_aggregate_sha256": aggregate_hash(scenario_files),
                    "dataset": {"path": str(args.dataset), "sha256": dataset_hash},
                    "sumo_dir": str(args.sumo_dir),
                    "edge_xml": str(args.edge_xml),
                    "implementation_aggregate_sha256": source_aggregate,
                    "sumo_version": sumo_version,
                    "max_events": int(args.max_events),
                    "max_sim_time_seconds": int(args.max_sim_time_seconds),
                }
                signature = json_hash(identity)
                directory = output_root / _safe_name(policy.name) / scenario_name
                stem = f"sumo_seed{sumo_seed}"
                output = directory / f"{stem}.json"
                if output in seen_outputs:
                    raise ValueError(f"duplicate evaluation output path: {output}")
                seen_outputs.add(output)
                jobs.append(
                    EvaluationJob(
                        policy=policy,
                        sumo_seed=sumo_seed,
                        scenario_manifest=scenario_manifest,
                        output=output,
                        event_log=(directory / f"{stem}.npz" if args.save_event_logs else None),
                        signature=signature,
                        identity=identity,
                    )
                )

    report_path = output_root / "matrix_manifest.json"
    started_utc = datetime.now(timezone.utc).isoformat()
    status_by_output = {
        str(job.output): {
            "status": "pending",
            "output": str(job.output),
            "policy": job.policy.name,
            "sumo_seed": job.sumo_seed,
            "scenario_manifest": str(job.scenario_manifest),
            "job_signature": job.signature,
        }
        for job in jobs
    }
    report = {
        "schema_version": MATRIX_SCHEMA,
        "started_utc": started_utc,
        "completed_utc": None,
        "workers": workers,
        "max_events": args.max_events,
        "max_sim_time_seconds": args.max_sim_time_seconds,
        "sumo_version": sumo_version,
        "dataset": {"path": str(args.dataset), "sha256": dataset_hash},
        "implementation_sha256": source_hashes,
        "implementation_aggregate_sha256": source_aggregate,
        "scenario_input_sha256": scenario_hashes,
        "jobs": list(status_by_output.values()),
    }

    with _matrix_lock(output_root):
        _write_report(report_path, report)
        executor = ThreadPoolExecutor(max_workers=workers)
        futures = {
            executor.submit(_run_job, job, args): job
            for job in jobs
        }
        try:
            for completed_count, future in enumerate(as_completed(futures), start=1):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {
                        "status": "failed",
                        "output": str(job.output),
                        "policy": job.policy.name,
                        "sumo_seed": job.sumo_seed,
                        "scenario_manifest": str(job.scenario_manifest),
                        "job_signature": job.signature,
                        "exception": repr(error),
                    }
                status_by_output[str(job.output)] = result
                report["jobs"] = sorted(
                    status_by_output.values(),
                    key=lambda row: (row["policy"], row["scenario_manifest"], row["sumo_seed"]),
                )
                _write_report(report_path, report)
                print(
                    f"[{completed_count}/{len(jobs)}] {result['status']:>9} "
                    f"{result['policy']} seed={result['sumo_seed']} "
                    f"{Path(result['scenario_manifest']).parent.name}",
                    flush=True,
                )
        except KeyboardInterrupt:
            _terminate_active_processes()
            for future in futures:
                future.cancel()
            report["interrupted_utc"] = datetime.now(timezone.utc).isoformat()
            _write_report(report_path, report)
            raise SystemExit(130)
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        _write_report(report_path, report)

    failures = [
        row
        for row in status_by_output.values()
        if row["status"] not in {"completed", "skipped"}
    ]
    if failures:
        raise SystemExit(f"{len(failures)} evaluation jobs failed; see {report_path}")


if __name__ == "__main__":
    main()
