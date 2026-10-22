#!/usr/bin/env python3
"""Run the frozen v19 completion-safety paired development screen."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import (  # noqa: E402
    FORMAL_V7_SUMO_BACKEND,
    FORMAL_V7_SUMO_VERSION,
    load_sumo_runtime_lock,
    load_validated_scenario,
    sha256,
    sumo_runtime_fingerprint,
)
from reproducibility import (  # noqa: E402
    run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch as base,
)
from reproducibility.run_evaluation_matrix_v19_completion_safety import (  # noqa: E402
    METHODS_SCHEMA,
    load_development_methods,
)


EXECUTION_SCHEMA = (
    "transit-h2oplus-v19-completion-safety-evaluation-host-execution-v1"
)
SUMMARY_SCHEMA = (
    "transit-h2oplus-v19-completion-safety-evaluation-host-summary-v1"
)
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
MATRIX_SOURCES = tuple(
    [
        f"SimpleSAC/evaluate_transit_v{version}_{name}.py"
        for version, name in (
            (11, "actor_hold"),
            (12, "staged_release"),
            (13, "proximal_release"),
            (14, "terminal_taper"),
            (15, "hold_calibration"),
            (16, "headway_reserve"),
            (17, "tail_service_reserve"),
            (19, "completion_safety_reserve"),
        )
    ]
    + [
        f"reproducibility/run_evaluation_matrix_v{version}_{name}.py"
        for version, name in (
            (11, "actor_hold"),
            (12, "staged_release"),
            (13, "proximal_release"),
            (14, "terminal_taper"),
            (15, "hold_calibration"),
            (16, "headway_reserve"),
            (17, "tail_service_reserve"),
            (19, "completion_safety"),
        )
    ]
)
OPERATIONAL_SOURCES = (
    "reproducibility/analyze_h2oplus_v18_empty_tail_reserve.py",
    "reproducibility/run_h2oplus_v18_empty_tail_reserve_evaluation_dispatch.py",
    "reproducibility/analyze_h2oplus_v19_completion_safety.py",
    "reproducibility/run_h2oplus_v19_completion_safety_evaluation_dispatch.py",
)
REQUIRED_ENVIRONMENT = dict(base.REQUIRED_ENVIRONMENT)
REQUIRED_ENVIRONMENT["H2OPLUS_V19_COMPLETION_SAFETY_PROTOCOL_SHA256"] = (
    "fa5f629588d35b1435e28bd39931f49d9851811ea3e308d62672f9eacf1f7184"
)


def verify_design(path: Path, payload: dict) -> list[dict]:
    disjointness = payload.get("disjointness", {})
    if (
        payload.get("study_stage") != "development"
        or payload.get("formal_held_out_outcomes_read") is not False
        or payload.get("formal_held_out_design_metadata_read") is not True
        or disjointness.get("verified_disjoint") is not True
        or any(value for key, value in disjointness.items() if key.endswith("_overlap"))
        or payload.get("environment_cells") != 9
    ):
        raise ValueError("frozen v19 development environment scope drifted")
    blocks = payload.get("blocks", [])
    result, names, sumo_seeds, demand_seeds = [], set(), set(), set()
    for block in blocks:
        name = block.get("name")
        sumo_seed = block.get("sumo_seed")
        demand_seed = block.get("demand_realization_seed")
        manifests = [
            base.resolve_relative(path, raw)
            for raw in block.get("scenario_manifests", [])
        ]
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
            or len(manifests) != 3
        ):
            raise ValueError(f"malformed v19 development block: {block!r}")
        scales = []
        for manifest in manifests:
            _, evidence = load_validated_scenario(manifest)
            if evidence.get("selection_seed") != demand_seed:
                raise ValueError(f"v19 demand seed drifted in {manifest}")
            scales.append(float(evidence["demand_scale"]))
        if sorted(scales) != [0.85, 1.0, 1.15]:
            raise ValueError(f"v19 demand scales drifted in {name}")
        names.add(name)
        sumo_seeds.add(sumo_seed)
        demand_seeds.add(demand_seed)
        result.append(
            {
                "name": name,
                "sumo_seed": sumo_seed,
                "manifests": [str(manifest) for manifest in manifests],
            }
        )
    expected_names = {
        f"v19-completion-safety-development-{index:02d}"
        for index in range(1, 4)
    }
    if (
        names != expected_names
        or sumo_seeds != {7301, 7302, 7303}
        or demand_seeds != {7401, 7402, 7403}
    ):
        raise ValueError("v19 development block identity drifted")
    return sorted(result, key=lambda row: row["name"])


def verify_inputs(args: argparse.Namespace) -> dict:
    mismatches = {
        key: {"observed": os.environ.get(key), "expected": value}
        for key, value in REQUIRED_ENVIRONMENT.items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise ValueError(f"frozen v19 environment drifted: {mismatches}")
    controls = (
        ("registration", "transit-h2oplus-v19-completion-safety-registration-v1"),
        ("authorization", "transit-h2oplus-v19-completion-safety-authorization-v1"),
        ("rationale", "transit-h2oplus-v19-completion-safety-rationale-v1"),
        ("environment_design", "transit-h2oplus-v19-completion-safety-environment-v1"),
        ("protocol", "transit-h2oplus-v19-completion-safety-protocol-v1"),
        (
            "correction",
            "transit-h2oplus-v19-completion-safety-environment-binding-correction-v2",
        ),
        ("parent_v18_audit", "transit-h2oplus-v18-empty-tail-reserve-evaluation-audit-v3"),
        ("parent_v18_closeout", "transit-h2oplus-v18-empty-tail-reserve-evaluation-closeout-v3"),
        (
            "parent_v18_verification",
            "transit-h2oplus-v18-empty-tail-reserve-evaluation-closeout-verification-v3",
        ),
        ("design_audit", "transit-h2oplus-v19-completion-safety-design-audit-v2"),
        (
            "design_verification",
            "transit-h2oplus-v19-completion-safety-design-verification-v2",
        ),
        ("data_environment_audit", "transit-h2oplus-v17-data-environment-convergence-audit-v1"),
        (
            "data_environment_verification",
            "transit-h2oplus-v17-data-environment-convergence-audit-verification-v1",
        ),
        ("source_freeze", "transit-h2oplus-v19-completion-safety-source-freeze-v2"),
    )
    loaded = {
        name: base.read_bound(
            getattr(args, name),
            getattr(args, f"expected_{name}_sha256"),
            schema,
        )
        for name, schema in controls
    }
    methods_path = args.methods_manifest.expanduser().resolve(strict=True)
    if sha256(methods_path) != args.expected_methods_sha256:
        raise ValueError("v19 methods identity drifted")
    methods = load_development_methods(methods_path, METHODS_SCHEMA)
    dataset = args.dataset.expanduser().resolve(strict=True)
    if sha256(dataset) != args.expected_dataset_sha256:
        raise ValueError("v19 dataset identity drifted")
    deployment_path = args.deployment_manifest.expanduser().resolve(strict=True)
    if sha256(deployment_path) != args.expected_deployment_manifest_sha256:
        raise ValueError("v19 deployment identity drifted")
    deployment = json.loads(deployment_path.read_text())
    runtime_lock, runtime_identity = load_sumo_runtime_lock(
        args.sumo_runtime_lock,
        expected_sha256=args.expected_runtime_lock_sha256,
    )
    active_runtime = sumo_runtime_fingerprint(
        FORMAL_V7_SUMO_BACKEND,
        expected_version=FORMAL_V7_SUMO_VERSION,
    )
    registration = loaded["registration"][1]
    authorization = loaded["authorization"][1]
    rationale = loaded["rationale"][1]
    environment = loaded["environment_design"][1]
    protocol = loaded["protocol"][1]
    correction = loaded["correction"][1]
    parent_audit = loaded["parent_v18_audit"][1]
    parent_closeout = loaded["parent_v18_closeout"][1]
    parent_verification = loaded["parent_v18_verification"][1]
    design_audit = loaded["design_audit"][1]
    design_verification = loaded["design_verification"][1]
    data_audit = loaded["data_environment_audit"][1]
    data_verification = loaded["data_environment_verification"][1]
    source_freeze = loaded["source_freeze"][1]
    matrix_sources = base._freeze_map(
        source_freeze.get("matrix_sources", []), "matrix"
    )
    operational = base._freeze_map(
        source_freeze.get("operational_sources", []), "operational"
    )
    if set(matrix_sources) != set(MATRIX_SOURCES) or set(operational) != set(
        OPERATIONAL_SOURCES
    ):
        raise ValueError("v19 scientific source set drifted")
    for relative, digest in {**matrix_sources, **operational}.items():
        if sha256(ROOT / relative) != digest:
            raise ValueError(f"v19 scientific source drifted: {relative}")
    scope = authorization.get("authorization_scope", {})
    candidate = registration.get("candidate", {})
    if (
        registration.get("formal_held_out_outcomes_read") is not False
        or registration.get("v19_development_evaluation_outcomes_read") is not False
        or registration.get("new_training_runs") != 0
        or candidate.get("algorithm")
        != "h2oplus-v19-completion-safety-quarter-load-sixty"
        or candidate.get("reserve_floor_seconds") != 60.0
        or candidate.get("max_onboard_capacity_fraction") != 0.25
        or scope.get("remote_development_evaluation_authorized") is not True
        or scope.get("remote_sumo_episode_cap") != 135
        or scope.get("unique_task_count_cap") != 1
        or scope.get("remote_host") != "jtl110gpu"
        or scope.get("cpu_cores") != 12
        or scope.get("ram_mb") != 32768
        or scope.get("gpu_count") != 0
        or any(
            scope.get(key) is not False
            for key in (
                "duplicate_submission_authorized",
                "force_dispatch_authorized",
                "formal_confirmation_stage_authorized",
                "formal_held_out_access_authorized",
                "local_training_or_sumo_authorized",
                "move_host_authorized",
                "new_training_authorized",
                "checkpoint_selection_by_behavior_authorized",
            )
        )
        or rationale.get("formal_held_out_outcomes_read") is not False
        or rationale.get("v19_development_evaluation_outcomes_read") is not False
        or protocol.get("formal_held_out_outcomes_read") is not False
        or protocol.get("v19_development_evaluation_outcomes_read") is not False
        or protocol.get("screen_type") != "paired-mechanism-screen"
        or protocol.get("evaluation", {}).get("new_remote_sumo_episodes") != 135
        or correction.get("status") != "frozen-predeployment-zero-result-correction"
        or correction.get("correction_scope", {}).get(
            "add_exact_protocol_environment_binding"
        )
        is not True
        or correction.get("correction_scope", {}).get("change_policy_behavior")
        is not False
        or correction.get("execution_boundary", {}).get(
            "completed_scientific_result_count"
        )
        != 0
        or correction.get("execution_boundary", {}).get(
            "scientific_numeric_outcomes_read"
        )
        is not False
        or parent_audit.get("passed") is not True
        or parent_audit.get("artifact_integrity", {}).get("passed") is not True
        or parent_audit.get("parent_training_stability", {}).get("passed") is not True
        or parent_closeout.get("passed") is not True
        or parent_verification.get("passed") is not True
        or any(
            payload.get("formal_held_out_outcomes_read") is not False
            for payload in (parent_audit, parent_closeout, parent_verification)
        )
        or design_audit.get("claim_boundary", {}).get(
            "development_result_available"
        )
        is not False
        or design_verification.get("passed") is not True
        or data_audit.get("formal_held_out_outcomes_read") is not False
        or data_audit.get("conclusions", {}).get(
            "material_data_or_environment_factors_present"
        )
        is not True
        or data_verification.get("passed") is not True
        or source_freeze.get("formal_held_out_outcomes_read") is not False
        or source_freeze.get("v19_development_evaluation_outcomes_read") is not False
        or source_freeze.get("physical_behavior_count") != 15
        or source_freeze.get("expected_new_remote_sumo_episodes") != 135
        or source_freeze.get("expected_matrix_source_file_count") != 55
        or source_freeze.get("matrix_source_count") != 16
        or source_freeze.get("operational_source_count") != 4
        or len(methods) != 15
        or deployment.get("schema_version")
        != "transit-h2oplus-v19-completion-safety-evaluation-deployment-manifest-v2"
        or deployment.get("physical_behavior_count") != 15
        or deployment.get("expected_new_remote_sumo_episodes") != 135
        or deployment.get("methods_manifest", {}).get("sha256")
        != args.expected_methods_sha256
        or deployment.get("environment_design", {}).get("sha256")
        != args.expected_environment_design_sha256
        or deployment.get("source_freeze", {}).get("sha256")
        != args.expected_source_freeze_sha256
        or deployment.get("sumo_runtime_signature")
        != runtime_identity["runtime_signature"]
        or active_runtime["portable_signature"]
        != runtime_identity["runtime_signature"]
        or active_runtime["portable_identity"]
        != runtime_lock["runtime_fingerprint"]["portable_identity"]
    ):
        raise ValueError("v19 evaluation scope or immutable evidence drifted")
    evidence = {
        name: {"path": str(path), "sha256": getattr(args, f"expected_{name}_sha256")}
        for name, (path, _) in loaded.items()
    }
    evidence.update(
        {
            "methods": {
                "path": str(methods_path),
                "sha256": args.expected_methods_sha256,
                "count": 15,
            },
            "dataset": {"path": str(dataset), "sha256": args.expected_dataset_sha256},
            "deployment_manifest": {
                "path": str(deployment_path),
                "sha256": args.expected_deployment_manifest_sha256,
            },
            "runtime_lock": runtime_identity,
            "runtime_fingerprint": active_runtime,
            "required_environment": dict(REQUIRED_ENVIRONMENT),
            "blocks": verify_design(loaded["environment_design"][0], environment),
            "parent_v18_artifact_integrity_passed": True,
            "parent_training_stability_passed": True,
            "data_environment_material_factors_present": True,
        }
    )
    return evidence


def block_command(args: argparse.Namespace, block: dict, output: Path) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "reproducibility/run_evaluation_matrix_v19_completion_safety.py"),
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
        "1",
        "--shard-index",
        "0",
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
    source_hashes = payload.get("source_sha256")
    if (
        payload.get("schema_version") != "transit-evaluation-matrix-v7"
        or payload.get("methods_manifest_sha256") != args.expected_methods_sha256
        or payload.get("dataset_sha256") != args.expected_dataset_sha256
        or payload.get("sumo_runtime_lock", {}).get("sha256")
        != args.expected_runtime_lock_sha256
        or partition.get("shard_count") != 1
        or partition.get("shard_index") != 0
        or partition.get("full_job_count") != 45
        or partition.get("selected_job_count") != 45
        or len(jobs) != 45
        or not isinstance(source_hashes, dict)
        or len(source_hashes) != args.expected_matrix_source_file_count
    ):
        raise ValueError(f"completed v19 block manifest drifted: {path}")
    source_freeze = json.loads(
        args.source_freeze.expanduser().resolve(strict=True).read_text()
    )
    for source in source_freeze.get("matrix_sources", []):
        if not base._source_is_bound(
            source_hashes, str(source.get("path", "")), source.get("sha256")
        ):
            raise ValueError(f"v19 matrix source binding missing: {source}")
    signatures = set()
    for job in jobs:
        output = Path(job["output"])
        if not output.is_file():
            raise FileNotFoundError(output)
        result = json.loads(output.read_text())
        signature = job["signature"]
        method = job.get("method", {})
        policy = result.get("policy", {})
        reserve = policy.get("completion_safety_reserve", {})
        algorithm = str(method.get("algorithm", ""))
        prefix = "h2oplus-v19-completion-safety-"
        variant = algorithm[len(prefix) :] if algorithm.startswith(prefix) else ""
        expected_candidate = variant == "quarter-load-sixty"
        if (
            result.get("schema_version") != "transit-evaluation-v7"
            or result.get("formal_held_out_outcomes_read") not in (None, False)
            or result.get("evaluation_job", {}).get("signature") != signature
            or result.get("source_sha256") != source_hashes
            or result.get("method") != method.get("name")
            or result.get("runtime", {}).get("python_version") != "3.8.20"
            or result.get("runtime", {}).get("sumo_backend") != "libsumo"
            or policy.get("parent_method") != method.get("v17_parent_method")
            or reserve.get("schema_version")
            != "transit-h2oplus-v19-completion-safety-v1"
            or reserve.get("variant") != variant
            or reserve.get("parent_tail_service_floor_seconds")
            != method.get("tail_service_floor_seconds")
            or reserve.get("reserve_enabled") is not method.get("v19_reserve_enabled")
            or reserve.get("reserve_floor_seconds")
            != method.get("v19_reserve_floor_seconds")
            or reserve.get("activation_start_fraction")
            != method.get("v19_activation_start_fraction")
            or reserve.get("activation_end_fraction")
            != method.get("v19_activation_end_fraction")
            or reserve.get("max_onboard_capacity_fraction")
            != method.get("v19_max_onboard_fraction")
            or reserve.get("requires_forward_bus_present") is not True
            or reserve.get("requires_backward_bus_absent") is not True
            or reserve.get("requires_positive_eligible_arrival_rate") is not True
            or reserve.get("v19_protocol_sha256") != args.expected_protocol_sha256
            or reserve.get("parent_action_preserved_outside_guard") is not True
            or reserve.get("guarded_hold_may_exceed_parent") is not True
            or reserve.get("physical_max_hold_preserved") is not True
            or reserve.get("predesignated_candidate") is not expected_candidate
            or reserve.get("checkpoint_selection_by_behavior") is not False
            or reserve.get("formal_held_out_outcomes_read") is not False
            or not base._ledger_and_service_pass(result)
            or signature in signatures
        ):
            raise ValueError(f"v19 result binding drifted: {output}")
        signatures.add(signature)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "full_matrix_signature": partition["full_matrix_signature"],
        "selected_job_keys_sha256": partition["selected_job_keys_sha256"],
        "selected_job_count": 45,
        "source_file_count": len(source_hashes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = (
        "registration",
        "authorization",
        "rationale",
        "environment_design",
        "protocol",
        "correction",
        "parent_v18_audit",
        "parent_v18_closeout",
        "parent_v18_verification",
        "design_audit",
        "design_verification",
        "data_environment_audit",
        "data_environment_verification",
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
        "rationale",
        "environment_design",
        "protocol",
        "correction",
        "parent_v18_audit",
        "parent_v18_closeout",
        "parent_v18_verification",
        "design_audit",
        "design_verification",
        "data_environment_audit",
        "data_environment_verification",
        "source_freeze",
        "methods",
        "dataset",
        "runtime_lock",
        "deployment_manifest",
    )
    for name in expected:
        parser.add_argument(
            "--expected-" + name.replace("_", "-") + "-sha256", required=True
        )
    parser.add_argument("--expected-matrix-source-file-count", type=int, required=True)
    parser.add_argument("--workers", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    execution_path = args.execution_json.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    execution = {
        "schema_version": EXECUTION_SCHEMA,
        "status": "running",
        "started_utc": base.utc_now(),
        "study_stage": "development",
        "screen_type": "paired-mechanism-screen",
        "training": False,
        "formal_held_out_outcomes_read": False,
        "workers": args.workers,
        "expected_results": 135,
        "completed_results": 0,
        "completed_blocks": [],
    }
    base.atomic_json(execution_path, execution)
    base.emit("v19_dispatch_started", workers=args.workers, expected_results=135)
    try:
        if args.workers != 12:
            raise ValueError("v19 dispatch requires exactly 12 workers")
        evidence = verify_inputs(args)
        base.emit(
            "v19_inputs_verified",
            blocks=len(evidence["blocks"]),
            methods=evidence["methods"]["count"],
        )
        output_root.mkdir(parents=True, exist_ok=True)
        completed = []
        for block in evidence["blocks"]:
            block_root = output_root / block["name"]
            log_path = output_root / f"{block['name']}.log"
            execution["active_block"] = block["name"]
            execution["immutable_evidence"] = evidence
            base.atomic_json(execution_path, execution)
            command = block_command(args, block, block_root)
            base.emit(
                "v19_block_started",
                block=block["name"],
                completed_results=len(completed) * 45,
            )
            with log_path.open("a") as log:
                log.write(
                    json.dumps(
                        {
                            "event": "v19_matrix_started",
                            "utc": base.utc_now(),
                            "block": block["name"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                log.flush()
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if result.returncode:
                raise RuntimeError(
                    f"v19 matrix block failed: {block['name']}: {result.returncode}"
                )
            manifest = verify_block(block_root / "matrix_manifest.json", args)
            completed.append(
                {
                    "block": block["name"],
                    "manifest": manifest,
                    "log": str(log_path),
                }
            )
            execution["completed_blocks"] = completed
            execution["completed_results"] = len(completed) * 45
            execution.pop("active_block", None)
            base.atomic_json(execution_path, execution)
            base.emit(
                "v19_block_completed",
                block=block["name"],
                completed_results=execution["completed_results"],
            )
        execution.update(
            {
                "status": "completed",
                "completed_utc": base.utc_now(),
                "completed_results": 135,
                "formal_held_out_outcomes_read": False,
            }
        )
        base.atomic_json(execution_path, execution)
        base.atomic_json(
            output_root / "host_summary.json",
            {
                "schema_version": SUMMARY_SCHEMA,
                "status": "completed",
                "created_utc": base.utc_now(),
                "physical_evaluation_results": 135,
                "matrix_manifests": completed,
                "formal_held_out_outcomes_read": False,
                "training": False,
            },
        )
        base.emit("v19_dispatch_completed", completed_results=135)
        return 0
    except Exception as exc:
        execution.update(
            {
                "status": "failed",
                "failed_utc": base.utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "formal_held_out_outcomes_read": False,
            }
        )
        base.atomic_json(execution_path, execution)
        base.emit("v19_dispatch_failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
