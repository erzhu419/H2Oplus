#!/usr/bin/env python3
"""Build, verify, and run the immutable v7 CPU fidelity evidence bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility.deployment_bundle_v6 import (
    MANIFEST_NAME,
    _aggregate,
    _walk_entries,
)
from SimpleSAC.train_transit_v3 import simulator_input_hashes


SCHEMA_VERSION = "transit-v7-cpu-fidelity-bundle-v1"
AUDIT_SCHEMA = "transit-v7-cpu-fidelity-bundle-audit-v1"
CPU_NODES = tuple(f"node{index:03d}" for index in range(1, 7))
DEFAULT_SERVER_PYTHON = Path(
    "/home/zhengliang01/scheduleurm_work/conda_envs/scomp-py310/bin/python"
)
ORIGINAL_SOURCE_ROOT = Path(
    "/home/erzhu419/mine_code/sumo-rl/H2Oplus"
)
CALIBRATED_ENV = Path("bus_h2o/calibrated_env")
ZERO_EVIDENCE = Path("bus_h2o/datasets_v7/formal_v2_zero")
BEHAVIOR_EVIDENCE = Path("bus_h2o/datasets_v7/formal_v2_behavior")
COMBINED_MANIFEST = Path(
    "bus_h2o/datasets_v7/formal_v2_full/combined_matrix_manifest.json"
)
NETWORK_XML = Path("bus_h2o/network_data/a_sorted_busline_edge.xml")
OUTPUT_PATHS = {
    "zero": Path("results/fidelity_audit_v7_formal_v4.json"),
    "intervention": Path(
        "results/intervention_fidelity_audit_v7_formal_v3.json"
    ),
}
LEGACY_DATA_PARTS = {
    "datasets_v2",
    "datasets_v3",
    "datasets_v4",
    "datasets_v5",
    "datasets_v6",
    "snapshots",
    "snapshot_store",
}
FORBIDDEN_MODEL_SUFFIXES = {".ckpt", ".h5", ".hdf5", ".pkl", ".pt", ".pth"}
REQUIRED_LINE_FILES = {
    "passenger_OD.xlsx",
    "passenger_OD_sumo.xlsx",
    "route_news.xlsx",
    "stop_news.xlsx",
    "time_table.xlsx",
    "time_table_sumo.xlsx",
}
CALIBRATION_ROOT_FILES = {
    "config.json",
    "observation_speed_profiles_v7_v1.npz",
    "segment_runtime_calibration_v7_v4.json",
    "simulator_demand_calibration.json",
    "simulator_journeys_v1.json",
}


def _inside_root(path: Path) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError(f"CPU fidelity source escapes H2Oplus: {resolved}")
    return resolved


def _python_sources() -> Iterable[Path]:
    excluded_roots = {
        (ROOT / CALIBRATED_ENV).resolve(),
        (ROOT / ZERO_EVIDENCE).resolve(),
        (ROOT / BEHAVIOR_EVIDENCE).resolve(),
    }
    for source_root in (
        ROOT / "SimpleSAC",
        ROOT / "bus_h2o",
        ROOT / "collect_policy",
    ):
        for path in source_root.rglob("*.py"):
            resolved = path.resolve()
            if (
                "__pycache__" not in path.parts
                and not any(part in LEGACY_DATA_PARTS for part in path.parts)
                and not any(root == resolved or root in resolved.parents for root in excluded_roots)
            ):
                yield resolved
    for path in (ROOT / "reproducibility").glob("*.py"):
        yield path.resolve()
    for path in (ROOT / "requirements.txt", ROOT / "pytest.ini", ROOT / NETWORK_XML):
        if path.is_file():
            yield path.resolve()


def _calibrated_environment_sources() -> set[Path]:
    root = _inside_root(ROOT / CALIBRATED_ENV)
    sources = {_inside_root(root / name) for name in CALIBRATION_ROOT_FILES}
    data = root / "data"
    line_dirs = sorted(path for path in data.iterdir() if path.is_dir())
    for line_dir in line_dirs:
        available = {path.name for path in line_dir.iterdir() if path.is_file()}
        if REQUIRED_LINE_FILES <= available:
            sources.update(_inside_root(line_dir / name) for name in REQUIRED_LINE_FILES)
    identity = simulator_input_hashes(
        root,
        network_path=ROOT / NETWORK_XML,
        runtime_calibration_version="v7_v4",
    )
    discovered = set(identity["line_ids"])
    packaged = {path.parent.name for path in sources if path.parent.parent == data}
    if packaged != discovered or len(packaged) != 12:
        raise ValueError(
            "calibrated v7 line inventory drifted: "
            f"packaged={sorted(packaged)}, runtime={sorted(discovered)}"
        )
    return sources


def _tree_sources(relative_root: Path) -> set[Path]:
    root = _inside_root(ROOT / relative_root)
    return {
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
    }


def _sources() -> set[Path]:
    sources = set(_python_sources())
    sources.update(_calibrated_environment_sources())
    sources.update(_tree_sources(ZERO_EVIDENCE))
    sources.update(_tree_sources(BEHAVIOR_EVIDENCE))
    sources.add(_inside_root(ROOT / COMBINED_MANIFEST))
    missing = sorted(str(path) for path in sources if not path.is_file())
    if missing:
        raise FileNotFoundError(f"CPU fidelity bundle sources are missing: {missing[:20]}")
    return sources


def _copy(source: Path, destination: Path, *, link_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    if link_files:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _is_mutable_output(relative: str) -> bool:
    allowed = {path.as_posix() for path in OUTPUT_PATHS.values()}
    return relative in allowed or (
        relative.endswith(".partial") and relative[: -len(".partial")] in allowed
    )


def _actual_immutable_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    for directory, names, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in list(names) + list(files):
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if (
                relative == MANIFEST_NAME
                or "__pycache__" in path.parts
                or path.suffix == ".pyc"
                or _is_mutable_output(relative)
            ):
                continue
            if path.is_file() or path.is_symlink():
                paths.add(relative)
    return paths


def _data_policy(entries: list[dict[str, Any]]) -> dict[str, Any]:
    legacy = []
    forbidden_models = []
    hdf5 = []
    npz = []
    allowed_npz_roots = (ZERO_EVIDENCE, BEHAVIOR_EVIDENCE)
    calibrated_profile = CALIBRATED_ENV / "observation_speed_profiles_v7_v1.npz"
    unexpected_npz = []
    for entry in entries:
        path = Path(entry["path"])
        if any(part in LEGACY_DATA_PARTS for part in path.parts):
            legacy.append(entry["path"])
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_MODEL_SUFFIXES:
            forbidden_models.append(entry["path"])
        if suffix in {".h5", ".hdf5"}:
            hdf5.append(entry["path"])
        if suffix == ".npz":
            npz.append(entry["path"])
            if path != calibrated_profile and not any(
                root == path or root in path.parents for root in allowed_npz_roots
            ):
                unexpected_npz.append(entry["path"])
    if legacy or forbidden_models or hdf5 or unexpected_npz or not npz:
        raise ValueError(
            "CPU fidelity data policy failed: "
            f"legacy={legacy}, forbidden_models={forbidden_models}, "
            f"hdf5={hdf5}, unexpected_npz={unexpected_npz}, npz={len(npz)}"
        )
    return {
        "legacy_or_snapshot_data_files": 0,
        "training_hdf5_files": 0,
        "model_checkpoint_files": 0,
        "immutable_sumo_evidence_npz_files": len(npz) - 1,
        "calibrated_speed_profile_npz_files": 1,
        "statement": (
            "the bundle contains only immutable formal SUMO evidence and the "
            "minimal calibrated simulator inputs; no legacy snapshot corpus, "
            "training HDF5, or model checkpoint is present"
        ),
    }


def _subset_descriptor(entries: list[dict[str, Any]], prefix: Path) -> dict[str, Any]:
    selected = [
        entry
        for entry in entries
        if prefix == Path(entry["path"]) or prefix in Path(entry["path"]).parents
    ]
    if not selected:
        raise ValueError(f"CPU fidelity bundle subset is empty: {prefix}")
    return {
        "path": prefix.as_posix(),
        "entry_count": len(selected),
        "total_file_bytes": sum(int(entry.get("bytes", 0)) for entry in selected),
        "aggregate_sha256": _aggregate(selected),
    }


def _run_command(mode: str, *, workers: int, server_label: str) -> list[str]:
    if mode not in OUTPUT_PATHS:
        raise ValueError(f"unsupported CPU fidelity mode: {mode}")
    if server_label not in CPU_NODES:
        raise ValueError(f"unsupported CPU fidelity node: {server_label}")
    if workers <= 0 or workers > 96:
        raise ValueError("CPU fidelity workers must be in [1, 96]")
    common = [
        str(DEFAULT_SERVER_PYTHON),
        "-u",
        "reproducibility/build_cpu_fidelity_bundle_v7.py",
        "run",
        "--mode",
        mode,
        "--workers",
        str(workers),
        "--server-label",
        server_label,
    ]
    return common


def build(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to replace CPU fidelity bundle {output}")
    sources = _sources()
    output.mkdir(parents=True)
    try:
        for source in sorted(sources):
            _copy(
                source,
                output / source.relative_to(ROOT),
                link_files=args.link_files,
            )
        entries = _walk_entries(output)
        data_policy = _data_policy(entries)
        simulator_identity = simulator_input_hashes(
            output / CALIBRATED_ENV,
            network_path=output / NETWORK_XML,
            runtime_calibration_version="v7_v4",
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(ROOT),
            "bundle_label": args.label,
            "entry_count": len(entries),
            "total_file_bytes": sum(int(entry.get("bytes", 0)) for entry in entries),
            "aggregate_sha256": _aggregate(entries),
            "execution_policy": {
                "local_experiment_execution_allowed": False,
                "allowed_cpu_nodes": list(CPU_NODES),
                "python": str(DEFAULT_SERVER_PYTHON),
                "purpose": "CPU-only v7 simulator fidelity reruns",
            },
            "data_policy": data_policy,
            "inputs": {
                "zero_sumo_evidence": _subset_descriptor(entries, ZERO_EVIDENCE),
                "behavior_sumo_evidence": _subset_descriptor(
                    entries, BEHAVIOR_EVIDENCE
                ),
                "calibrated_simulator": _subset_descriptor(entries, CALIBRATED_ENV),
                "combined_matrix_manifest": {
                    "path": COMBINED_MANIFEST.as_posix(),
                    "bytes": (output / COMBINED_MANIFEST).stat().st_size,
                    "sha256": sha256(output / COMBINED_MANIFEST),
                },
                "simulator_input_aggregate_sha256": simulator_identity[
                    "aggregate_sha256"
                ],
            },
            "mutable_outputs": {
                mode: path.as_posix() for mode, path in OUTPUT_PATHS.items()
            },
            "registered_commands": {
                "zero_node001": _run_command(
                    "zero", workers=30, server_label="node001"
                ),
                "intervention_node002": _run_command(
                    "intervention", workers=64, server_label="node002"
                ),
            },
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
                    "manifest_sha256": sha256(manifest_path),
                    "aggregate_sha256": manifest["aggregate_sha256"],
                    "entry_count": manifest["entry_count"],
                    "total_file_bytes": manifest["total_file_bytes"],
                },
                sort_keys=True,
            )
        )
    except Exception:
        shutil.rmtree(output, ignore_errors=True)
        raise
    return 0


def _verify_bundle(root: Path, *, verify_loaders: bool) -> dict[str, Any]:
    root = root.expanduser().resolve(strict=True)
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    errors = []
    entries = manifest.get("entries", [])
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("manifest schema mismatch")
    if _aggregate(entries) != manifest.get("aggregate_sha256"):
        errors.append("manifest aggregate mismatch")
    expected_paths = {entry["path"] for entry in entries}
    actual_paths = _actual_immutable_paths(root)
    if expected_paths != actual_paths:
        errors.append(
            "immutable artifact set mismatch: "
            f"missing={sorted(expected_paths - actual_paths)[:20]}, "
            f"extra={sorted(actual_paths - expected_paths)[:20]}"
        )
    for entry in entries:
        path = root / entry["path"]
        if entry.get("type") == "symlink":
            if not path.is_symlink() or os.readlink(path) != entry.get("target"):
                errors.append(f"missing or stale symlink: {entry['path']}")
        elif (
            not path.is_file()
            or path.stat().st_size != entry.get("bytes")
            or sha256(path) != entry.get("sha256")
        ):
            errors.append(f"missing or stale file: {entry['path']}")
    try:
        _data_policy(entries)
        identity = simulator_input_hashes(
            root / CALIBRATED_ENV,
            network_path=root / NETWORK_XML,
            runtime_calibration_version="v7_v4",
        )
        if identity["aggregate_sha256"] != manifest.get("inputs", {}).get(
            "simulator_input_aggregate_sha256"
        ):
            raise ValueError("calibrated simulator input identity drifted")
        if verify_loaders:
            from reproducibility.audit_simulator_fidelity_v7 import (
                load_reserved_sumo_summaries,
            )
            from reproducibility.audit_simulator_intervention_fidelity_v7 import (
                load_sumo_rows,
            )
            from reproducibility.relocatable_paths_v7 import ArtifactPathRemap

            remap = ArtifactPathRemap.build(
                Path(manifest["source_root"]), root
            )
            _, _, zero_rows = load_reserved_sumo_summaries(
                root / ZERO_EVIDENCE / "zero_matrix_manifest.json",
                root / CALIBRATED_ENV / "segment_runtime_calibration_v7_v4.json",
                path_remap=remap,
            )
            _, intervention_rows = load_sumo_rows(
                root / COMBINED_MANIFEST,
                path_remap=remap,
            )
            if len(zero_rows) != 15 or len(intervention_rows) != 240:
                raise ValueError(
                    "formal fidelity evidence cell count drifted: "
                    f"zero={len(zero_rows)}, intervention={len(intervention_rows)}"
                )
    except Exception as error:
        errors.append(str(error))
    return {
        "schema_version": AUDIT_SCHEMA,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "manifest_sha256": sha256(manifest_path),
        "loaders_verified": bool(verify_loaders and not errors),
        "passed": not errors,
        "errors": errors,
    }


def verify(args: argparse.Namespace) -> int:
    report = _verify_bundle(args.root, verify_loaders=args.verify_loaders)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


def _audit_command(mode: str, workers: int, source_root: Path) -> list[str]:
    if mode == "zero":
        return [
            sys.executable,
            "-u",
            "reproducibility/audit_simulator_fidelity_v7.py",
            "--matrix-manifest",
            (ROOT / ZERO_EVIDENCE / "zero_matrix_manifest.json").as_posix(),
            "--runtime-calibration",
            (ROOT / CALIBRATED_ENV / "segment_runtime_calibration_v7_v4.json").as_posix(),
            "--sim-env",
            (ROOT / CALIBRATED_ENV).as_posix(),
            "--workers",
            str(workers),
            "--path-remap-from",
            source_root.as_posix(),
            "--path-remap-to",
            ROOT.as_posix(),
            "--output",
            (ROOT / OUTPUT_PATHS[mode]).as_posix(),
        ]
    return [
        sys.executable,
        "-u",
        "reproducibility/audit_simulator_intervention_fidelity_v7.py",
        "--matrix-manifest",
        (ROOT / COMBINED_MANIFEST).as_posix(),
        "--sim-env",
        (ROOT / CALIBRATED_ENV).as_posix(),
        "--workers",
        str(workers),
        "--path-remap-from",
        source_root.as_posix(),
        "--path-remap-to",
        ROOT.as_posix(),
        "--output",
        (ROOT / OUTPUT_PATHS[mode]).as_posix(),
    ]


def run(args: argparse.Namespace) -> int:
    if args.server_label not in CPU_NODES:
        raise ValueError(f"unsupported CPU fidelity node: {args.server_label}")
    hostname = socket.gethostname().split(".", 1)[0]
    if hostname != args.server_label:
        raise RuntimeError(
            "CPU fidelity experiments are server-only: "
            f"requested={args.server_label}, active_host={hostname}"
        )
    if args.workers <= 0 or args.workers > 96:
        raise ValueError("CPU fidelity workers must be in [1, 96]")
    report = _verify_bundle(ROOT, verify_loaders=True)
    if not report["passed"]:
        raise RuntimeError(f"CPU fidelity bundle verification failed: {report['errors']}")
    manifest = json.loads((ROOT / MANIFEST_NAME).read_text())
    source_root = Path(manifest["source_root"])
    output = ROOT / OUTPUT_PATHS[args.mode]
    if output.exists():
        raise FileExistsError(f"refusing to overwrite CPU fidelity output {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = _audit_command(args.mode, args.workers, source_root)
    print(json.dumps({"command": command}, sort_keys=True), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if not output.is_file():
        raise RuntimeError(f"CPU fidelity audit produced no output: {output}")
    payload = json.loads(output.read_text())
    result = {
        "mode": args.mode,
        "server_label": args.server_label,
        "hostname": hostname,
        "workers": args.workers,
        "returncode": completed.returncode,
        "output": str(output),
        "output_sha256": sha256(output),
        "passed": payload.get("passed") is True,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if completed.returncode == 0 and result["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--label", default="v7-cpu-fidelity-formal-v1")
    build_parser.add_argument("--link-files", action="store_true")
    build_parser.set_defaults(function=build)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--verify-loaders", action="store_true")
    verify_parser.set_defaults(function=verify)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--mode", choices=sorted(OUTPUT_PATHS), required=True)
    run_parser.add_argument("--workers", type=int, required=True)
    run_parser.add_argument("--server-label", choices=CPU_NODES, required=True)
    run_parser.set_defaults(function=run)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
