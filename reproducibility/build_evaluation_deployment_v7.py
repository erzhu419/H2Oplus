#!/usr/bin/env python3
"""Build or verify a snapshot-free formal v7 CPU evaluation workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
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
from reproducibility.run_evaluation_matrix_v4 import (
    METHODS_SCHEMAS,
    evaluation_source_paths_v7,
    load_environment_design,
    load_methods,
)


SCHEMA_VERSION = "transit-v7-cpu-evaluation-deployment-v1"
AUDIT_SCHEMA = "transit-v7-cpu-evaluation-deployment-audit-v1"
MANIFEST_NAME = "deployment_manifest.json"
LEGACY_PARTS = {
    "datasets_v2",
    "datasets_v3",
    "datasets_v4",
    "datasets_v5",
    "datasets_v6",
    "snapshots",
    "snapshot_store",
}


def _aggregate(entries: list[dict]) -> str:
    encoded = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _descriptor(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256(path),
    }


def _source_destination(output: Path, source: Path) -> Path:
    source = source.resolve()
    if source == ROOT or ROOT in source.parents:
        return output / "H2Oplus" / source.relative_to(ROOT)
    if source == PROJECT_ROOT or PROJECT_ROOT in source.parents:
        return output / source.relative_to(PROJECT_ROOT)
    raise ValueError(f"deployment source is outside the project: {source}")


def _python_sources() -> set[Path]:
    paths = set()
    for source_root in (
        ROOT / "SimpleSAC",
        ROOT / "bus_h2o",
        ROOT / "collect_policy",
    ):
        paths.update(
            path.resolve()
            for path in source_root.rglob("*.py")
            if "__pycache__" not in path.parts
            and not any(part in LEGACY_PARTS for part in path.parts)
        )
    paths.update(
        path.resolve()
        for path in (ROOT / "reproducibility").glob("*.py")
    )
    for path in (ROOT / "requirements.txt", ROOT / "pytest.ini"):
        if path.is_file():
            paths.add(path.resolve())
    return paths


def _copy(source: Path, destination: Path, *, link_large_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    if link_large_files and source.stat().st_size >= 100 * 1024**2:
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def _collect_sources(args: argparse.Namespace) -> tuple[set[Path], dict]:
    design = args.environment_design.expanduser().resolve()
    cells, design_evidence = load_environment_design(
        design, expected_mdp_version="v7"
    )
    dataset = args.dataset.expanduser().resolve()
    runtime_lock = args.sumo_runtime_lock.expanduser().resolve()
    runtime_payload, runtime_identity = load_sumo_runtime_lock(runtime_lock)
    sumo_dir = args.sumo_dir.expanduser().resolve()
    edge_xml = args.edge_xml.expanduser().resolve()

    sources = _python_sources()
    sources.update({design, dataset, runtime_lock})
    generator = Path(design_evidence["path"]).parent / json.loads(
        design.read_text()
    )["generator"]
    sources.add(generator.resolve())
    scenario_descriptors = []
    for cell in cells:
        manifest = Path(cell["scenario"]["path"]).resolve()
        runtime, evidence = load_validated_scenario(manifest)
        scenario_paths = {
            manifest,
            Path(runtime["sumo_cfg"]).resolve(),
            Path(runtime["passenger_obj_xml"]).resolve(),
            *(Path(path).resolve() for path in evidence["runtime_input_sha256"]),
        }
        sources.update(scenario_paths)
        scenario_descriptors.append(
            {
                "manifest": str(manifest),
                "manifest_sha256": evidence["manifest_sha256"],
                "runtime_content_sha256": sorted(
                    evidence["runtime_input_sha256"].values()
                ),
            }
        )
    sources.update(evaluation_source_paths_v7(sumo_dir, edge_xml))

    method_descriptors = []
    for manifest in args.methods_manifest:
        resolved = manifest.expanduser().resolve()
        methods = load_methods(resolved, METHODS_SCHEMAS["v7"])
        sources.add(resolved)
        for method in methods:
            if method.get("checkpoint") is not None:
                sources.add(Path(method["checkpoint"]).resolve())
        method_descriptors.append(
            {
                "path": str(resolved),
                "sha256": sha256(resolved),
                "methods": [method["name"] for method in methods],
            }
        )

    missing = sorted(str(path) for path in sources if not path.is_file())
    if missing:
        raise FileNotFoundError(f"deployment sources are missing: {missing[:20]}")
    metadata = {
        "environment_design": {
            "path": str(design),
            "sha256": sha256(design),
            "cells": len(cells),
        },
        "dataset": {
            "path": str(dataset),
            "sha256": sha256(dataset),
            "size_bytes": int(dataset.stat().st_size),
        },
        "sumo_runtime_lock": runtime_identity,
        "sumo_runtime_signature": runtime_payload["runtime_signature"],
        "scenario_descriptors": scenario_descriptors,
        "methods_manifests": method_descriptors,
    }
    return sources, metadata


def _data_policy(entries: list[dict], expected_dataset_sha256: str) -> dict:
    legacy = []
    hdf5 = []
    forbidden_models = []
    for entry in entries:
        path = Path(entry["path"])
        if any(part in LEGACY_PARTS for part in path.parts):
            legacy.append(entry["path"])
        if path.suffix.lower() in {".h5", ".hdf5"}:
            hdf5.append(entry)
        if path.suffix.lower() in {".ckpt", ".pth"}:
            forbidden_models.append(entry["path"])
    if legacy or forbidden_models or len(hdf5) != 1:
        raise ValueError(
            "evaluation deployment data policy failed: "
            f"legacy={legacy}, models={forbidden_models}, hdf5={len(hdf5)}"
        )
    if hdf5[0]["sha256"] != expected_dataset_sha256:
        raise ValueError("deployment HDF5 is not the canonical v7 dataset")
    return {
        "legacy_or_snapshot_files": 0,
        "hdf5_files": 1,
        "canonical_dataset_sha256": expected_dataset_sha256,
        "statement": (
            "one compact transit-smdp-v7 HDF5 is present; no legacy snapshot "
            "dataset or training checkpoint is deployed"
        ),
    }


def build(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace deployment {output}")
    sources, metadata = _collect_sources(args)
    output.mkdir(parents=True)
    try:
        for source in sorted(sources):
            _copy(
                source,
                _source_destination(output, source),
                link_large_files=args.link_large_files,
            )
        files = sorted(
            path for path in output.rglob("*") if path.is_file() and not path.is_symlink()
        )
        entries = [_descriptor(path, output) for path in files]
        data_policy = _data_policy(entries, metadata["dataset"]["sha256"])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(ROOT),
            "entry_count": len(entries),
            "total_file_bytes": sum(entry["size_bytes"] for entry in entries),
            "aggregate_sha256": _aggregate(entries),
            "data_policy": data_policy,
            **metadata,
            "entries": entries,
        }
        destination = output / MANIFEST_NAME
        destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "output": str(output),
                    "manifest_sha256": sha256(destination),
                    "aggregate_sha256": manifest["aggregate_sha256"],
                    "entry_count": len(entries),
                    "total_file_bytes": manifest["total_file_bytes"],
                },
                sort_keys=True,
            )
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return 0


def verify(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    errors = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("manifest schema mismatch")
    entries = manifest.get("entries", [])
    if _aggregate(entries) != manifest.get("aggregate_sha256"):
        errors.append("manifest aggregate mismatch")
    for entry in entries:
        path = root / entry["path"]
        if (
            not path.is_file()
            or path.stat().st_size != entry["size_bytes"]
            or sha256(path) != entry["sha256"]
        ):
            errors.append(f"missing or stale file: {entry['path']}")
    try:
        _data_policy(entries, manifest["dataset"]["sha256"])
        lock_path = root / "H2Oplus" / "reproducibility" / Path(
            manifest["sumo_runtime_lock"]["path"]
        ).name
        lock, identity = load_sumo_runtime_lock(
            lock_path,
            expected_sha256=manifest["sumo_runtime_lock"]["sha256"],
        )
        if identity["runtime_signature"] != manifest["sumo_runtime_signature"]:
            raise ValueError("deployment runtime lock signature drifted")
        if args.verify_runtime:
            active = sumo_runtime_fingerprint(
                FORMAL_V7_SUMO_BACKEND,
                expected_version=FORMAL_V7_SUMO_VERSION,
            )
            if (
                active["portable_signature"] != identity["runtime_signature"]
                or active["portable_identity"]
                != lock["runtime_fingerprint"]["portable_identity"]
            ):
                raise RuntimeError("active server runtime differs from deployment lock")
    except Exception as error:
        errors.append(str(error))
    report = {
        "schema_version": AUDIT_SCHEMA,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "manifest_sha256": sha256(manifest_path),
        "runtime_verified": bool(args.verify_runtime and not errors),
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
    build_parser.add_argument(
        "--environment-design",
        type=Path,
        default=ROOT
        / "reproducibility"
        / "evaluation_environment_design_v7_formal_v2.json",
    )
    build_parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT
        / "bus_h2o"
        / "datasets_v7"
        / "formal_v2_full"
        / "transit_smdp_v7_formal_v2.h5",
    )
    build_parser.add_argument(
        "--sumo-runtime-lock",
        type=Path,
        default=ROOT / "reproducibility" / "sumo_runtime_lock_v7_cpu_server.json",
    )
    build_parser.add_argument(
        "--methods-manifest", type=Path, action="append", default=[]
    )
    build_parser.add_argument("--link-large-files", action="store_true")
    build_parser.add_argument(
        "--sumo-dir", type=Path, default=PROJECT_ROOT / "SUMO_ruiguang" / "online_control"
    )
    build_parser.add_argument(
        "--edge-xml",
        type=Path,
        default=ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml",
    )
    build_parser.set_defaults(function=build)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--verify-runtime", action="store_true")
    verify_parser.set_defaults(function=verify)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
