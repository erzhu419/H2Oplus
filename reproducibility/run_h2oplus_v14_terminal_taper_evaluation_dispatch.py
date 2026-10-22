#!/usr/bin/env python3
"""Run one fail-closed remote CPU shard of the v14 development evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import (
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    load_sumo_runtime_lock,
    load_validated_scenario,
    sha256,
    sumo_runtime_fingerprint,
)
from reproducibility.run_evaluation_matrix_v14_terminal_taper import (
    METHODS_SCHEMA,
    load_development_methods,
)


EXECUTION_SCHEMA = "transit-h2oplus-v14-terminal-taper-evaluation-host-execution-v1"
SUMMARY_SCHEMA = "transit-h2oplus-v14-terminal-taper-evaluation-host-summary-v1"
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
EVALUATION_SOURCES = (
    "SimpleSAC/evaluate_transit_v11_actor_hold.py",
    "SimpleSAC/evaluate_transit_v12_staged_release.py",
    "SimpleSAC/evaluate_transit_v13_proximal_release.py",
    "SimpleSAC/evaluate_transit_v14_terminal_taper.py",
    "reproducibility/run_evaluation_matrix_v11_actor_hold.py",
    "reproducibility/run_evaluation_matrix_v12_staged_release.py",
    "reproducibility/run_evaluation_matrix_v13_proximal_release.py",
    "reproducibility/run_evaluation_matrix_v14_terminal_taper.py",
    "reproducibility/run_h2oplus_v14_terminal_taper_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v14_terminal_taper.py",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def read_bound(path: Path, expected: str, schema: str | None = None) -> tuple[Path, dict]:
    resolved = path.expanduser().resolve(strict=True)
    actual = sha256(resolved)
    if actual != expected:
        raise ValueError(f"identity drift for {resolved}: {actual}")
    payload = json.loads(resolved.read_text())
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError(f"schema drift for {resolved}")
    return resolved, payload


def resolve_relative(owner: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = owner.parent / path
    return path.resolve(strict=True)


def verify_design(path: Path, payload: dict) -> list[dict]:
    disjointness = payload.get("disjointness", {})
    if (
        payload.get("study_stage") != "development"
        or payload.get("formal_held_out_outcomes_read") is not False
        or disjointness.get("verified_disjoint") is not True
        or any(value for key, value in disjointness.items() if key.endswith("_overlap"))
        or payload.get("environment_cells") != 9
    ):
        raise ValueError("frozen development environment scope drifted")
    blocks = payload.get("blocks", [])
    if len(blocks) != 3:
        raise ValueError("v14 evaluation requires exactly three development blocks")
    result, names, sumo_seeds, demand_seeds = [], set(), set(), set()
    for block in blocks:
        name = block.get("name")
        sumo_seed = block.get("sumo_seed")
        demand_seed = block.get("demand_realization_seed")
        raw_manifests = block.get("scenario_manifests", [])
        if (
            not isinstance(name, str)
            or not SAFE_NAME.fullmatch(name)
            or name in names
            or not isinstance(sumo_seed, int)
            or isinstance(sumo_seed, bool)
            or sumo_seed in sumo_seeds
            or not isinstance(demand_seed, int)
            or isinstance(demand_seed, bool)
            or demand_seed in demand_seeds
            or len(raw_manifests) != 3
        ):
            raise ValueError(f"malformed development block: {block!r}")
        names.add(name)
        sumo_seeds.add(sumo_seed)
        demand_seeds.add(demand_seed)
        manifests, scales = [], []
        for raw in raw_manifests:
            manifest = resolve_relative(path, raw)
            _, evidence = load_validated_scenario(manifest)
            if evidence.get("selection_seed") != demand_seed:
                raise ValueError(f"development demand seed drifted in {manifest}")
            manifests.append(str(manifest))
            scales.append(float(evidence["demand_scale"]))
        if sorted(scales) != [0.85, 1.0, 1.15]:
            raise ValueError(f"development demand scales drifted in {name}")
        result.append({"name": name, "sumo_seed": sumo_seed, "manifests": manifests})
    expected_names = {f"v12-staged-release-development-0{index}" for index in range(1, 4)}
    if names != expected_names or sumo_seeds != {5101, 5102, 5103}:
        raise ValueError("development block identity drifted")
    return sorted(result, key=lambda item: item["name"])


def source_freeze_map(payload: dict) -> dict[str, str]:
    result = {}
    for row in payload.get("sources", []):
        path, digest = row.get("path"), row.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str) or path in result:
            raise ValueError("malformed v14 source freeze")
        result[path] = digest
    return result


def verify_inputs(args: argparse.Namespace) -> dict:
    controls = (
        ("registration", "transit-h2oplus-v14-terminal-taper-registration-v1"),
        ("authorization", "transit-h2oplus-v14-terminal-taper-authorization-v1"),
        ("freeze", "transit-h2oplus-v14-terminal-taper-freeze-v1"),
        ("environment_design", "transit-h2oplus-v12-staged-release-environment-v2"),
        ("protocol", "transit-h2oplus-v14-terminal-taper-protocol-v1"),
        ("training_closeout", "transit-h2oplus-v14-terminal-taper-branch-closeout-v1"),
        (
            "training_verification",
            "transit-h2oplus-v14-terminal-taper-branch-closeout-verification-v1",
        ),
        ("training_audit", "transit-h2oplus-v14-terminal-taper-branch-audit-v1"),
        (
            "training_file_manifest",
            "transit-h2oplus-v14-terminal-taper-branch-file-manifest-v1",
        ),
        (
            "source_freeze",
            "transit-h2oplus-v14-terminal-taper-evaluation-source-freeze-v1",
        ),
    )
    loaded = {}
    for name, schema in controls:
        loaded[name] = read_bound(
            getattr(args, name), getattr(args, f"expected_{name}_sha256"), schema
        )
    methods_path = args.methods_manifest.expanduser().resolve(strict=True)
    if sha256(methods_path) != args.expected_methods_sha256:
        raise ValueError("v14 methods identity drifted")
    methods = load_development_methods(methods_path, METHODS_SCHEMA)
    dataset = args.dataset.expanduser().resolve(strict=True)
    if sha256(dataset) != args.expected_dataset_sha256:
        raise ValueError("dataset identity drifted")
    deployment_path = args.deployment_manifest.expanduser().resolve(strict=True)
    if sha256(deployment_path) != args.expected_deployment_manifest_sha256:
        raise ValueError("deployment identity drifted")
    deployment = json.loads(deployment_path.read_text())
    if (
        deployment.get("schema_version")
        != "transit-h2oplus-v14-terminal-taper-evaluation-deployment-manifest-v1"
    ):
        raise ValueError("deployment schema drifted")
    runtime_lock, runtime_identity = load_sumo_runtime_lock(
        args.sumo_runtime_lock,
        expected_sha256=args.expected_runtime_lock_sha256,
    )
    active_runtime = sumo_runtime_fingerprint(
        FORMAL_V7_SUMO_BACKEND,
        expected_version=FORMAL_V7_SUMO_VERSION,
    )
    source_freeze = loaded["source_freeze"][1]
    sources = source_freeze_map(source_freeze)
    if set(sources) != set(EVALUATION_SOURCES):
        raise ValueError("v14 evaluation source set drifted")
    for relative in EVALUATION_SOURCES:
        if sources.get(relative) != sha256(ROOT / relative):
            raise ValueError(f"v14 evaluation source drifted: {relative}")
    registration = loaded["registration"][1]
    authorization = loaded["authorization"][1]
    freeze = loaded["freeze"][1]
    environment = loaded["environment_design"][1]
    protocol = loaded["protocol"][1]
    closeout = loaded["training_closeout"][1]
    verification = loaded["training_verification"][1]
    audit = loaded["training_audit"][1]
    file_manifest = loaded["training_file_manifest"][1]
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or registration.get("development_evaluation_submitted") is not False
        or registration.get("new_training_runs") != 15
        or registration.get("predesignated_candidate") != "terminal-quarter"
        or authorization.get("formal_held_out_access_authorized") is not False
        or authorization.get("local_training_or_sumo_authorized") is not False
        or authorization.get(
            "remote_development_evaluation_after_training_closeout_authorized"
        )
        is not True
        or authorization.get("formal_confirmation_stage_authorized") is not False
        or freeze.get("formal_held_out_outcomes_read") is not False
        or freeze.get("development_evaluation_submitted") is not False
        or environment.get("formal_held_out_outcomes_read") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("development_evaluation_submitted") is not False
        or closeout.get("status") != "passed"
        or closeout.get("artifact_integrity_passed") is not True
        or closeout.get("scientific_stability_passed_for_all_15_branches") is not True
        or closeout.get("exact_seed_matched_v12_10k_handoffs_passed") is not True
        or closeout.get("development_evaluation_gate_open") is not True
        or verification.get("passed") is not True
        or verification.get("all_15_scientific_stability_passed") is not True
        or verification.get("exact_seed_matched_v12_10k_handoffs_passed") is not True
        or verification.get("development_evaluation_gate_open") is not True
        or audit.get("passed") is not True
        or audit.get("artifact_integrity_passed") is not True
        or audit.get("all_15_stability_passed") is not True
        or audit.get("all_seed_matched_v12_10k_handoffs_exact") is not True
        or file_manifest.get("file_count") != 255
        or source_freeze.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("evaluation_outcomes_read") is not False
        or source_freeze.get("physical_actor_count") != 70
        or source_freeze.get("logical_alias_count") != 180
        or source_freeze.get("expected_remote_sumo_episodes") != 630
        or source_freeze.get("registered_maximum_remote_sumo_episodes") != 1620
        or source_freeze.get("source_file_count") != 10
        or len(methods) != 70
        or deployment.get("physical_actor_count") != 70
        or deployment.get("logical_alias_count") != 180
        or deployment.get("expected_remote_sumo_episodes") != 630
        or deployment.get("methods_manifest", {}).get("sha256")
        != args.expected_methods_sha256
        or deployment.get("source_freeze", {}).get("sha256")
        != args.expected_source_freeze_sha256
        or deployment.get("environment_design", {}).get("sha256")
        != args.expected_environment_design_sha256
        or deployment.get("training_closeout", {}).get("sha256")
        != args.expected_training_closeout_sha256
        or deployment.get("training_verification", {}).get("sha256")
        != args.expected_training_verification_sha256
        or deployment.get("sumo_runtime_signature") != runtime_identity["runtime_signature"]
        or deployment.get("evaluation_source_sha256") != sources
        or active_runtime["portable_signature"] != runtime_identity["runtime_signature"]
        or active_runtime["portable_identity"]
        != runtime_lock["runtime_fingerprint"]["portable_identity"]
    ):
        raise ValueError("v14 evaluation scope or immutable evidence drifted")
    evidence = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, (path, _) in loaded.items()
    }
    evidence.update(
        {
            "methods": {
                "path": str(methods_path),
                "sha256": sha256(methods_path),
                "count": 70,
            },
            "dataset": {"path": str(dataset), "sha256": sha256(dataset)},
            "deployment_manifest": {
                "path": str(deployment_path),
                "sha256": sha256(deployment_path),
                "aggregate_sha256": deployment.get("aggregate_sha256"),
            },
            "runtime_lock": runtime_identity,
            "runtime_fingerprint": active_runtime,
            "blocks": verify_design(loaded["environment_design"][0], environment),
            "training_scientific_stability_passed": True,
            "exact_seed_matched_v12_10k_handoffs_passed": True,
            "stability_failure_runs": [],
        }
    )
    return evidence


def block_command(args: argparse.Namespace, block: dict, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "reproducibility/run_evaluation_matrix_v14_terminal_taper.py"),
        "--mdp-version",
        "v7",
        "--methods-manifest",
        str(args.methods_manifest.expanduser().resolve()),
        "--scenario-manifests",
        *(str(path) for path in block["manifests"]),
        "--sumo-seeds",
        str(block["sumo_seed"]),
        "--dataset",
        str(args.dataset.expanduser().resolve()),
        "--output-root",
        str(output),
        "--workers",
        str(args.workers),
        "--shard-count",
        str(args.shard_count),
        "--shard-index",
        str(args.shard_index),
        "--device",
        "cpu",
        "--sumo-backend",
        FORMAL_V7_SUMO_BACKEND,
        "--expected-sumo-runtime-version",
        FORMAL_V7_SUMO_VERSION,
        "--sumo-runtime-lock",
        str(args.sumo_runtime_lock.expanduser().resolve()),
        "--sumo-dir",
        str(args.sumo_dir.expanduser().resolve()),
        "--edge-xml",
        str(args.edge_xml.expanduser().resolve()),
    ]


def verify_block(path: Path, args: argparse.Namespace) -> dict:
    payload = json.loads(path.read_text())
    partition = payload.get("matrix_partition", {})
    jobs = payload.get("jobs", [])
    expected_selected = 105
    if (
        payload.get("schema_version") != "transit-evaluation-matrix-v7"
        or payload.get("methods_manifest_sha256") != args.expected_methods_sha256
        or payload.get("dataset_sha256") != args.expected_dataset_sha256
        or payload.get("sumo_runtime_lock", {}).get("sha256")
        != args.expected_runtime_lock_sha256
        or partition.get("shard_count") != 2
        or partition.get("shard_index") != args.shard_index
        or partition.get("full_job_count") != 210
        or partition.get("selected_job_count") != expected_selected
        or len(jobs) != expected_selected
    ):
        raise ValueError(f"completed v14 block manifest drifted: {path}")
    signatures = set()
    source_hashes = payload.get("source_sha256")
    if not isinstance(source_hashes, dict) or len(source_hashes) != 47:
        raise ValueError(f"v14 source contract drifted: {path}")
    for job in jobs:
        output = Path(job["output"])
        if not output.is_file():
            raise FileNotFoundError(output)
        result = json.loads(output.read_text())
        signature = job["signature"]
        if (
            result.get("schema_version") != "transit-evaluation-v7"
            or result.get("evaluation_job", {}).get("signature") != signature
            or result.get("source_sha256") != source_hashes
            or signature in signatures
        ):
            raise ValueError(f"v14 result binding drifted: {output}")
        signatures.add(signature)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "full_matrix_signature": partition["full_matrix_signature"],
        "selected_job_keys_sha256": partition["selected_job_keys_sha256"],
        "selected_job_count": expected_selected,
        "source_file_count": len(source_hashes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = (
        "registration",
        "authorization",
        "freeze",
        "environment_design",
        "protocol",
        "training_closeout",
        "training_verification",
        "training_audit",
        "training_file_manifest",
        "source_freeze",
        "methods_manifest",
        "dataset",
        "sumo_runtime_lock",
        "deployment_manifest",
        "output_root",
        "execution_json",
        "sumo_dir",
        "edge_xml",
    )
    for name in paths:
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    expected = (
        "registration",
        "authorization",
        "freeze",
        "environment_design",
        "protocol",
        "training_closeout",
        "training_verification",
        "training_audit",
        "training_file_manifest",
        "source_freeze",
        "methods",
        "dataset",
        "runtime_lock",
        "deployment_manifest",
    )
    for name in expected:
        parser.add_argument("--expected-" + name.replace("_", "-") + "-sha256", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.workers != 12 or args.shard_count != 2 or args.shard_index not in (0, 1):
        raise ValueError("v14 evaluation is frozen to 12 workers and two shards")
    output_root = args.output_root.expanduser().resolve()
    execution_path = args.execution_json.expanduser().resolve()
    execution = {
        "schema_version": EXECUTION_SCHEMA,
        "status": "starting",
        "started_utc": utc_now(),
        "hostname": os.uname().nodename,
        "pid": os.getpid(),
        "python_executable": sys.executable,
        "workers": 12,
        "shard_count": 2,
        "shard_index": args.shard_index,
        "training": False,
        "formal_held_out_outcomes_read": False,
        "completed_blocks": [],
    }
    atomic_json(execution_path, execution)
    try:
        evidence = verify_inputs(args)
        execution.update({"status": "running", "evidence": evidence})
        atomic_json(execution_path, execution)
        output_root.mkdir(parents=True, exist_ok=True)
        completed = []
        suffix = f".shard-{args.shard_index:04d}-of-0002"
        for block in evidence["blocks"]:
            block_output = output_root / block["name"]
            block_output.mkdir(parents=True, exist_ok=True)
            command = block_command(args, block, block_output)
            log_path = output_root / f"{block['name']}.shard-{args.shard_index}.log"
            execution.update(
                {
                    "active_block": block["name"],
                    "active_command": command,
                    "active_log": str(log_path),
                }
            )
            atomic_json(execution_path, execution)
            with log_path.open("a") as log:
                log.write(f"[{utc_now()}] START {' '.join(command)}\n")
                log.flush()
                result = subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False
                )
                log.write(f"[{utc_now()}] RETURN_CODE {result.returncode}\n")
            if result.returncode != 0:
                raise RuntimeError(f"block {block['name']} returned {result.returncode}")
            manifest = block_output / f"matrix_manifest{suffix}.json"
            completed.append(verify_block(manifest, args))
            execution["completed_blocks"] = completed
            atomic_json(execution_path, execution)
        expected_jobs = 315
        if sum(item["selected_job_count"] for item in completed) != expected_jobs:
            raise ValueError(f"v14 shard did not complete exactly {expected_jobs} jobs")
        summary = {
            "schema_version": SUMMARY_SCHEMA,
            "completed_utc": utc_now(),
            "hostname": os.uname().nodename,
            "workers": 12,
            "shard_count": 2,
            "shard_index": args.shard_index,
            "completed_blocks": completed,
            "completed_jobs": expected_jobs,
            "training": False,
            "formal_held_out_outcomes_read": False,
            "evidence": evidence,
        }
        summary_path = output_root / "host_summary.json"
        atomic_json(summary_path, summary)
        execution.update(
            {
                "status": "completed",
                "completed_utc": utc_now(),
                "returncode": 0,
                "active_block": None,
                "active_command": None,
                "summary": {"path": str(summary_path), "sha256": sha256(summary_path)},
            }
        )
        atomic_json(execution_path, execution)
        print(
            f"DONE H2O+ v14 terminal-taper shard {args.shard_index}: "
            f"{expected_jobs}/{expected_jobs} jobs",
            flush=True,
        )
        return 0
    except Exception as error:
        execution.update(
            {
                "status": "failed",
                "failed_utc": utc_now(),
                "returncode": 1,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_json(execution_path, execution)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
