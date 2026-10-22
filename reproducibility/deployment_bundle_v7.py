#!/usr/bin/env python3
"""Build and verify a minimal, admission-locked transit-smdp-v7 workspace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reproducibility.build_training_admission_v7 import (
    SCHEMA_VERSION as ADMISSION_SCHEMA,
)
from reproducibility.build_training_plan_v7 import (
    FORMAL_SERVER_GPU_CONCURRENCY,
    H2OPLUS_METHODS,
    PLAN_SCHEMA,
    SERVER_EXECUTION_CLASS,
    SERVER_H2OPLUS_EXECUTION_CLASS,
    SERVER_METHODS,
    load_admission,
)
from reproducibility.deployment_bundle_v6 import (
    DEFAULT_COMPILER,
    MANIFEST_NAME,
    TOOLCHAIN_INCLUDE,
    _actual_paths,
    _aggregate,
    _copy_calibrated_environment,
    _copy_file,
    _manifest_runtime_environment,
    _resolve_manifest_runtime_environment,
    _runtime_environment,
    _runtime_fingerprint,
    _walk_entries,
    sha256,
)
from reproducibility.run_training_dispatch_v6 import load_dispatch
from reproducibility.run_training_matrix_v4 import load_plan
from SimpleSAC.train_transit_v3 import simulator_input_hashes, source_hashes


SCHEMA_VERSION = "transit-v7-deployment-bundle-v2"
AUDIT_SCHEMA = "transit-v7-deployment-audit-v2"
LEGACY_DATA_PARTS = {
    "datasets_v2",
    "datasets_v3",
    "datasets_v4",
    "datasets_v5",
    "datasets_v6",
    "snapshots",
    "snapshot_store",
}
FORBIDDEN_PAYLOAD_SUFFIXES = {".pkl", ".pt", ".pth", ".ckpt"}


def _inside_root(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"{label} must be inside the H2Oplus workspace: {resolved}")
    return resolved


def _safe_relative(base: Path, raw: Any, *, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is missing")
    relative = Path(raw).expanduser()
    if relative.is_absolute():
        raise ValueError(f"{label} must be a safe relative path")
    resolved = (base / relative).resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes the H2Oplus workspace")
    return resolved


def _python_sources() -> Iterable[Path]:
    for source_root in (
        ROOT / "SimpleSAC",
        ROOT / "bus_h2o",
        ROOT / "collect_policy",
    ):
        for path in source_root.rglob("*.py"):
            if (
                "__pycache__" not in path.parts
                and not any(part in LEGACY_DATA_PARTS for part in path.parts)
            ):
                yield path
    for path in (ROOT / "reproducibility").iterdir():
        if path.is_file() and path.suffix == ".py":
            yield path
    for path in (ROOT / "tests").glob("*.py"):
        yield path
    toolchain = ROOT / "reproducibility" / "toolchain"
    if toolchain.is_dir():
        for path in toolchain.rglob("*"):
            if path.is_file():
                yield path
    for path in (
        ROOT / "requirements.txt",
        ROOT / "pytest.ini",
        ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml",
    ):
        yield path


def _descriptor(path: Path) -> dict[str, Any]:
    path = _inside_root(path, label="bundle artifact")
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def _admission_artifacts(admission_path: Path) -> tuple[dict[str, Any], set[Path]]:
    admission_path = _inside_root(admission_path, label="training admission")
    payload = json.loads(admission_path.read_text())
    if payload.get("schema_version") != ADMISSION_SCHEMA:
        raise ValueError("deployment admission schema is invalid")
    artifacts = {admission_path}
    for name in ("dataset", "pilot_dataset"):
        descriptor = payload.get(name, {})
        artifact = _safe_relative(
            admission_path.parent,
            descriptor.get("path"),
            label=f"admission {name}",
        )
        artifacts.add(_inside_root(artifact, label=f"admission {name}"))
    evidence = payload.get("evidence", {})
    if not isinstance(evidence, dict):
        raise ValueError("deployment admission evidence is invalid")
    for name, descriptor in evidence.items():
        artifact = _safe_relative(
            admission_path.parent,
            descriptor.get("path"),
            label=f"admission evidence {name}",
        )
        artifacts.add(
            _inside_root(artifact, label=f"admission evidence {name}")
        )
    for relative in payload.get("training_source_sha256", {}):
        source = Path(relative)
        if source.is_absolute() or ".." in source.parts:
            raise ValueError("admission training source path is unsafe")
        artifacts.add(_inside_root(ROOT / source, label="training source"))
    artifacts.add(ROOT / "reproducibility" / "build_training_admission_v7.py")
    return payload, artifacts


def _offline_initializer_source_plans(
    plan_path: Path, payload: dict[str, Any]
) -> set[Path]:
    """Resolve and authenticate source plans declared by branch jobs."""

    dependencies: set[Path] = set()
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError(f"deployment plan jobs are invalid: {plan_path}")
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise ValueError(f"deployment plan job {index} is invalid: {plan_path}")
        dependency = job.get("offline_initializer")
        if dependency is None:
            continue
        if not isinstance(dependency, dict):
            raise ValueError(
                f"deployment initializer dependency is invalid: {plan_path}"
            )
        raw_source = dependency.get("source_plan")
        if not isinstance(raw_source, str) or not raw_source:
            raise ValueError(
                f"deployment initializer source plan is missing: {plan_path}"
            )
        source = Path(raw_source).expanduser()
        if not source.is_absolute():
            source = plan_path.parent / source
        source = _inside_root(source, label="offline initializer source plan")
        if dependency.get("source_plan_sha256") != sha256(source):
            raise ValueError(
                f"offline initializer source plan is stale: {source}"
            )
        dependencies.add(source)
    return dependencies


def _verified_artifact_descriptor(
    base: Path, descriptor: dict[str, Any], *, label: str
) -> Path:
    path = _safe_relative(base, descriptor.get("path"), label=label)
    path = _inside_root(path, label=label)
    if (
        descriptor.get("sha256") != sha256(path)
        or descriptor.get("bytes") != path.stat().st_size
    ):
        raise ValueError(f"{label} is stale: {path}")
    return path


def _host_concurrency_artifacts(
    plan_path: Path, payload: dict[str, Any]
) -> set[Path]:
    """Close the plan -> audit -> raw benchmark evidence chain."""

    host_policy = (
        payload.get("registration", {})
        .get("compute_policy", {})
        .get("host_gpu_concurrency", {})
    )
    descriptors = host_policy.get("evidence", [])
    if not descriptors:
        return set()
    if not isinstance(descriptors, list):
        raise ValueError("host concurrency evidence is invalid")
    artifacts: set[Path] = set()
    for index, descriptor in enumerate(descriptors):
        if not isinstance(descriptor, dict):
            raise ValueError("host concurrency evidence descriptor is invalid")
        audit = _verified_artifact_descriptor(
            plan_path.parent,
            descriptor,
            label=f"host concurrency audit {index}",
        )
        artifacts.add(audit)
        audit_payload = json.loads(audit.read_text())
        raw_descriptors = [audit_payload.get("single_process_reference")]
        raw_descriptors.extend(audit_payload.get("workers") or [])
        if any(not isinstance(item, dict) for item in raw_descriptors):
            raise ValueError(f"host concurrency audit evidence is invalid: {audit}")
        for report_index, report in enumerate(raw_descriptors):
            artifacts.add(
                _verified_artifact_descriptor(
                    audit.parent,
                    report,
                    label=(
                        f"host concurrency raw report {index}:{report_index}"
                    ),
                )
            )
    return artifacts


def _validate_plan_concurrency(
    payload: dict[str, Any],
    jobs: list[dict[str, Any]],
    methods: set[str],
) -> None:
    registration = payload.get("registration", {})
    status = registration.get("status")
    if status == "frozen-before-formal-outcomes":
        expected = {
            method: FORMAL_SERVER_GPU_CONCURRENCY.get(method, 1)
            for method in methods
        }
    elif status == "execution-gate":
        expected = {method: 1 for method in methods}
    else:
        raise ValueError("deployment plan registration status is invalid")

    declared = registration.get("compute_policy", {}).get(
        "canonical_jobs_per_physical_gpu_by_method"
    )
    if declared != expected:
        raise ValueError("deployment plan concurrency registration drifted")
    for job in jobs:
        if job["max_canonical_processes_per_gpu"] != expected[job["method"]]:
            raise ValueError(f"GPU quota drifted for {job['name']}")


def _plan_artifacts(
    plan_paths: Iterable[Path], admission_path: Path
) -> tuple[list[dict[str, Any]], set[Path]]:
    requested = {
        _inside_root(path, label="training plan") for path in plan_paths
    }
    ordered: list[tuple[Path, dict[str, Any], int, list[str]]] = []
    artifacts: set[Path] = set()
    admission_path = admission_path.resolve()
    visited: set[Path] = set()
    active: set[Path] = set()

    def visit(plan_path: Path) -> None:
        if plan_path in visited:
            return
        if plan_path in active:
            raise ValueError(
                f"cyclic deployment training-plan dependency: {plan_path}"
            )
        active.add(plan_path)
        payload = json.loads(plan_path.read_text())
        if payload.get("schema_version") != PLAN_SCHEMA:
            raise ValueError(f"deployment plan is not v7: {plan_path}")
        configured_admission = _safe_relative(
            plan_path.parent,
            payload.get("training_admission"),
            label="plan training admission",
        )
        if configured_admission != admission_path:
            raise ValueError(f"deployment plan uses another admission: {plan_path}")
        jobs, _ = load_plan(plan_path)
        methods = {job["method"] for job in jobs}
        _validate_plan_concurrency(payload, jobs, methods)
        execution_class = payload.get("execution_class")
        if execution_class == SERVER_H2OPLUS_EXECUTION_CLASS:
            if not methods <= set(H2OPLUS_METHODS):
                raise ValueError("server H2O+ plan contains a non-H2O+ method")
        elif execution_class == SERVER_EXECUTION_CLASS:
            if not methods <= set(SERVER_METHODS):
                raise ValueError("server baseline plan contains an H2O+ method")
        elif execution_class != "cpu-validation":
            raise ValueError(f"unsupported v7 execution class: {execution_class}")

        dependencies = _offline_initializer_source_plans(plan_path, payload)
        for dependency in sorted(dependencies):
            visit(dependency)

        artifacts.add(plan_path)
        artifacts.update(_host_concurrency_artifacts(plan_path, payload))
        registration = payload.get("registration", {})
        for name, descriptor in registration.get("evidence", {}).items():
            evidence_path = _safe_relative(
                plan_path.parent,
                descriptor.get("path"),
                label=f"plan evidence {name}",
            )
            artifacts.add(_inside_root(evidence_path, label=f"plan evidence {name}"))
        ordered.append(
            (plan_path, payload, len(jobs), sorted(methods))
        )
        active.remove(plan_path)
        visited.add(plan_path)

    for plan_path in sorted(requested):
        visit(plan_path)

    descriptors = [
        {
            **_descriptor(plan_path),
            "plan_id": payload.get("plan_id"),
            "execution_class": payload.get("execution_class"),
            "jobs": jobs,
            "methods": methods,
            "inclusion": (
                "requested"
                if plan_path in requested
                else "offline-initializer-source"
            ),
        }
        for plan_path, payload, jobs, methods in ordered
    ]
    return descriptors, artifacts


def _dispatch_artifacts(
    dispatch_paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], set[Path]]:
    descriptors = []
    artifacts: set[Path] = set()
    for raw_path in dispatch_paths:
        dispatch_path = _inside_root(raw_path, label="training dispatch")
        dispatch, _, _ = load_dispatch(dispatch_path)
        artifacts.add(dispatch_path)
        artifacts.add(
            _inside_root(
                Path(dispatch["plan_path"]), label="dispatch training plan"
            )
        )
        for host in dispatch["hosts"]:
            for gpu in host["gpus"]:
                capacity = gpu.get("capacity_audit")
                if capacity is not None:
                    artifacts.add(
                        _inside_root(
                            Path(capacity["path"]),
                            label="dispatch GPU capacity audit",
                        )
                    )
        descriptors.append(_descriptor(dispatch_path))
    return descriptors, artifacts


def _copy_artifacts(
    paths: Iterable[Path], destination: Path, *, link_large_files: bool
) -> None:
    for source in sorted({path.resolve() for path in paths}):
        _copy_file(
            source,
            destination / source.relative_to(ROOT),
            link_large_files=link_large_files,
        )


def _exclude_tree(paths: Iterable[Path], tree: Path) -> list[Path]:
    tree = tree.resolve()
    return [
        path
        for path in paths
        if path.resolve() != tree and tree not in path.resolve().parents
    ]


def _registered_calibrated_npz(simulator: dict[str, Any]) -> set[str]:
    runtime_files = simulator.get("runtime_files_sha256")
    if not isinstance(runtime_files, dict):
        raise ValueError("simulator runtime file inventory is invalid")
    registered = set()
    for raw_path in runtime_files:
        relative = Path(raw_path)
        if relative.suffix.lower() != ".npz":
            continue
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("registered simulator NPZ path is unsafe")
        registered.add(
            (Path("bus_h2o") / "calibrated_env" / relative).as_posix()
        )
    return registered


def _payload_policy(
    entries: list[dict[str, Any]],
    *,
    allowed_calibrated_npz: Iterable[str] = (),
) -> dict[str, Any]:
    allowed_npz = set(allowed_calibrated_npz)
    legacy = []
    forbidden = []
    v7_hdf5 = []
    registered_npz = []
    for entry in entries:
        if entry.get("type") != "file":
            continue
        path = Path(entry["path"])
        if any(part in LEGACY_DATA_PARTS for part in path.parts):
            legacy.append(entry["path"])
        if path.suffix.lower() in FORBIDDEN_PAYLOAD_SUFFIXES:
            forbidden.append(entry["path"])
        if path.suffix.lower() == ".npz":
            if entry["path"] in allowed_npz:
                registered_npz.append(entry["path"])
            else:
                forbidden.append(entry["path"])
        if path.suffix.lower() in {".h5", ".hdf5"}:
            if "datasets_v7" not in path.parts:
                legacy.append(entry["path"])
            else:
                v7_hdf5.append(entry["path"])
    missing_npz = sorted(allowed_npz - set(registered_npz))
    if legacy or forbidden or missing_npz:
        raise ValueError(
            "deployment contains legacy/snapshot/checkpoint payloads: "
            f"legacy={legacy[:10]}, forbidden={forbidden[:10]}, "
            f"missing_registered_npz={missing_npz[:10]}"
        )
    return {
        "legacy_dataset_files": 0,
        "snapshot_or_checkpoint_payload_files": 0,
        "registered_calibrated_npz_files": sorted(registered_npz),
        "v7_hdf5_files": sorted(v7_hdf5),
        "statement": (
            "the bundle contains v7 HDF5 data and admission-registered calibrated "
            "simulator NPZ files only; no legacy snapshot, checkpoint, or replay "
            "pickle payload is deployed"
        ),
    }


def build(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    compiler = args.compiler.expanduser().resolve(strict=True)
    admission_path = _inside_root(
        args.training_admission, label="training admission"
    )
    if output.exists():
        raise FileExistsError(f"refusing to replace deployment workspace {output}")

    admission = load_admission(admission_path)
    admission_payload, admission_files = _admission_artifacts(admission_path)
    plan_descriptors, plan_files = _plan_artifacts(args.plan, admission_path)
    additional = {
        _inside_root(path, label="additional deployment artifact")
        for path in args.artifact
    }
    dispatch_descriptors, dispatch_files = _dispatch_artifacts(args.dispatch)
    additional.update(dispatch_files)
    if not plan_descriptors:
        raise ValueError("at least one v7 training plan is required")
    canonical_simulator = ROOT / "bus_h2o" / "calibrated_env"

    output.mkdir(parents=True)
    try:
        _copy_artifacts(
            _exclude_tree(
                list(_python_sources())
                + list(admission_files)
                + list(plan_files)
                + list(additional),
                canonical_simulator,
            ),
            output,
            link_large_files=args.link_large_files,
        )
        if admission["simulator_root"] != canonical_simulator.resolve():
            raise ValueError("v7 deployment requires the canonical calibrated simulator")
        _copy_calibrated_environment(output)

        entries = _walk_entries(output)
        payload_policy = _payload_policy(
            entries,
            allowed_calibrated_npz=_registered_calibrated_npz(
                admission["simulator"]
            ),
        )
        runtime_environment = _runtime_environment(output, compiler)
        runtime_lock = _runtime_fingerprint(runtime_environment)["required"]
        if runtime_lock != admission["training_runtime_lock"]:
            raise ValueError(
                "deployment build runtime differs from the admitted precision runtime"
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(ROOT),
            "bundle_label": args.label,
            "entry_count": len(entries),
            "total_file_bytes": sum(
                int(item.get("bytes", 0)) for item in entries
            ),
            "aggregate_sha256": _aggregate(entries),
            "runtime_environment": _manifest_runtime_environment(compiler),
            "runtime_lock": runtime_lock,
            "training_admission": _descriptor(admission_path),
            "training_source_sha256": admission_payload[
                "training_source_sha256"
            ],
            "plans": plan_descriptors,
            "dispatches": dispatch_descriptors,
            "data_policy": payload_policy,
            "entries": entries,
        }
        manifest_path = output / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "workspace": str(output),
                    "manifest": str(manifest_path),
                    "aggregate_sha256": manifest["aggregate_sha256"],
                    "entry_count": manifest["entry_count"],
                    "total_file_bytes": manifest["total_file_bytes"],
                    "plans": len(plan_descriptors),
                },
                sort_keys=True,
            )
        )
    except Exception:
        shutil.rmtree(str(output), ignore_errors=True)
        raise
    return 0


def _verify_descriptor(root: Path, descriptor: dict[str, Any], label: str) -> Path:
    path = _safe_relative(root, descriptor.get("path"), label=label)
    if (
        not path.is_file()
        or descriptor.get("sha256") != sha256(path)
        or descriptor.get("bytes") != path.stat().st_size
    ):
        raise ValueError(f"{label} is missing or stale")
    return path


def _verify_admission_tree(
    root: Path, descriptor: dict[str, Any]
) -> tuple[Path, dict[str, Any]]:
    admission_path = _verify_descriptor(root, descriptor, "training admission")
    payload = json.loads(admission_path.read_text())
    if (
        payload.get("schema_version") != ADMISSION_SCHEMA
        or payload.get("study_stage") != "formal"
        or payload.get("passed") is not True
    ):
        raise ValueError("training admission is not passed formal evidence")
    for name in ("dataset", "pilot_dataset"):
        _verify_descriptor(admission_path.parent, payload[name], f"admission {name}")
    for name, evidence in payload.get("evidence", {}).items():
        path = _verify_descriptor(
            admission_path.parent, evidence, f"admission evidence {name}"
        )
        evidence_payload = json.loads(path.read_text())
        if evidence_payload.get("schema_version") != evidence.get("schema_version"):
            raise ValueError(f"admission evidence {name} schema drifted")
    for relative, expected in payload.get("training_source_sha256", {}).items():
        source = _safe_relative(root, relative, label="training source")
        if not source.is_file() or sha256(source) != expected:
            raise ValueError(f"training source is missing or stale: {relative}")
    simulator_root = _safe_relative(
        admission_path.parent,
        payload.get("simulator", {}).get("path"),
        label="admission simulator",
    )
    simulator = simulator_input_hashes(
        simulator_root,
        network_path=root
        / "bus_h2o"
        / "network_data"
        / "a_sorted_busline_edge.xml",
        runtime_calibration_version="v7_v4",
    )
    if (
        simulator["aggregate_sha256"]
        != payload["simulator"].get("aggregate_sha256")
        or simulator["runtime_sha256"]
        != payload["simulator"].get("runtime_sha256")
    ):
        raise ValueError("deployment simulator no longer matches admission")
    return admission_path, simulator


def verify(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else root / MANIFEST_NAME
    )
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("deployment manifest schema mismatch")
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("deployment manifest entries must be a list")
    if _aggregate(entries) != manifest.get("aggregate_sha256"):
        errors.append("deployment manifest aggregate hash mismatch")
    expected_paths = {str(item.get("path")) for item in entries}
    for item in entries:
        relative = str(item.get("path"))
        path = root / relative
        if item.get("type") == "symlink":
            if not path.is_symlink():
                errors.append(f"missing symlink: {relative}")
            elif os.readlink(str(path)) != item.get("target"):
                errors.append(f"symlink target mismatch: {relative}")
        elif item.get("type") == "file":
            if not path.is_file() or path.is_symlink():
                errors.append(f"missing file: {relative}")
            elif path.stat().st_size != int(item.get("bytes", -1)):
                errors.append(f"file size mismatch: {relative}")
            elif sha256(path) != item.get("sha256"):
                errors.append(f"file hash mismatch: {relative}")
        else:
            errors.append(f"unsupported manifest entry type: {relative}")
    unexpected = sorted(_actual_paths(root) - expected_paths)
    if unexpected:
        errors.append("unexpected workspace files: " + ", ".join(unexpected[:20]))

    runtime_environment: dict[str, str] | None = None
    runtime_fingerprint: dict[str, Any] | None = None
    admitted_runtime_lock: dict[str, Any] | None = None
    try:
        admission_path, simulator = _verify_admission_tree(
            root, manifest.get("training_admission", {})
        )
        verified_policy = _payload_policy(
            entries,
            allowed_calibrated_npz=_registered_calibrated_npz(simulator),
        )
        if verified_policy != manifest.get("data_policy"):
            raise ValueError("deployment data policy drifted")
        admitted_runtime_lock = json.loads(admission_path.read_text()).get(
            "training_runtime_lock"
        )
        for descriptor in manifest.get("plans", []):
            plan_path = _verify_descriptor(root, descriptor, "training plan")
            plan_payload = json.loads(plan_path.read_text())
            _host_concurrency_artifacts(plan_path, plan_payload)
            jobs, _ = load_plan(plan_path)
            if len(jobs) != descriptor.get("jobs"):
                raise ValueError("deployment training-plan job count drifted")
        for descriptor in manifest.get("dispatches", []):
            dispatch_path = _verify_descriptor(
                root, descriptor, "training dispatch"
            )
            load_dispatch(dispatch_path)
        runtime_environment = _resolve_manifest_runtime_environment(root, manifest)
        runtime_fingerprint = _runtime_fingerprint(runtime_environment)
        if runtime_fingerprint["required"] != manifest.get("runtime_lock"):
            raise ValueError("deployment runtime lock mismatch")
        if runtime_fingerprint["required"] != admitted_runtime_lock:
            raise ValueError(
                "deployment runtime differs from the admitted precision runtime"
            )
    except Exception as error:
        errors.append(str(error))

    report = {
        "schema_version": AUDIT_SCHEMA,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "manifest": str(manifest_path),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "runtime": (
            runtime_fingerprint["required"]
            if runtime_fingerprint is not None
            else None
        ),
        "host_observation": (
            runtime_fingerprint["host"]
            if runtime_fingerprint is not None
            else None
        ),
        "runtime_environment": runtime_environment,
        "data_policy": manifest.get("data_policy"),
        "passed": not errors,
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--label", required=True)
    build_parser.add_argument("--training-admission", type=Path, required=True)
    build_parser.add_argument("--plan", type=Path, action="append", default=[])
    build_parser.add_argument("--dispatch", type=Path, action="append", default=[])
    build_parser.add_argument("--artifact", type=Path, action="append", default=[])
    build_parser.add_argument("--link-large-files", action="store_true")
    build_parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    build_parser.set_defaults(function=build)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path)
    verify_parser.set_defaults(function=verify)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
