#!/usr/bin/env python3
"""Build and verify the immutable input bundle for v7 GPU trainer gates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bus_h2o.sumo_provenance import sha256
from reproducibility.deployment_bundle_v6 import (
    DEFAULT_COMPILER,
    MANIFEST_NAME,
    _aggregate,
    _copy_file,
    _manifest_runtime_environment,
    _resolve_manifest_runtime_environment,
    _runtime_environment,
    _runtime_fingerprint,
    _walk_entries,
)
from reproducibility.deployment_bundle_v7 import _payload_policy, _python_sources
from reproducibility.run_local_trainer_gates_v7 import (
    SCHEMA_VERSION as GATE_CHAIN_SCHEMA,
)


SCHEMA_VERSION = "transit-v7-server-trainer-gate-bundle-v1"
AUDIT_SCHEMA = "transit-v7-server-trainer-gate-bundle-audit-v1"
SERVER_LABELS = ("jtl110gpu", "jtl110gpu2", "node007")
DEFAULT_PYTHON = Path("/home/erzhu419/anaconda3/envs/LSTM-RL/bin/python")

INPUT_PATHS = {
    "pilot_dataset": Path(
        "bus_h2o/datasets_v7/formal_v2_pilot_r2/"
        "transit_smdp_v7_pilot_r2.h5"
    ),
    "pilot_dataset_audit": Path(
        "bus_h2o/datasets_v7/formal_v2_pilot_r2/"
        "transit_smdp_v7_pilot_r2.audit.json"
    ),
    "reward_audit": Path(
        "bus_h2o/datasets_v7/formal_v2_pilot_r2/"
        "reward_conditioning_v7_pilot_r2.json"
    ),
    "formal_dataset": Path(
        "bus_h2o/datasets_v7/formal_v2_full/"
        "transit_smdp_v7_formal_v2.h5"
    ),
}
OUTPUT_PATHS = {
    "reward_loss": Path(
        "bus_h2o/datasets_v7/formal_v2_pilot_r2/"
        "reward_loss_pilot_v7_formal_gate_v2.json"
    ),
    "hybrid_entropy": Path(
        "bus_h2o/datasets_v7/formal_v2_pilot_r2/"
        "hybrid_entropy_pilot_v7_formal_gate_v2.json"
    ),
    "device_precision": Path(
        "bus_h2o/datasets_v7/formal_v2_full/"
        "device_cache_precision_v7_formal_v2.json"
    ),
    "gate_chain": Path(
        "bus_h2o/datasets_v7/formal_v2_full/"
        "server_trainer_gate_chain_v7_formal_v2.json"
    ),
}


def _descriptor(path: Path, base: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    return {
        "path": path.relative_to(base.resolve()).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def _copy_inputs(output: Path, *, link_large_files: bool) -> None:
    sources = list(_python_sources()) + [ROOT / path for path in INPUT_PATHS.values()]
    seen: set[Path] = set()
    for source in sources:
        source = source.resolve(strict=True)
        if source in seen:
            continue
        seen.add(source)
        _copy_file(
            source,
            output / source.relative_to(ROOT),
            link_large_files=link_large_files,
        )


def _is_mutable_output(relative: str) -> bool:
    allowed = {path.as_posix() for path in OUTPUT_PATHS.values()}
    return relative in allowed or (
        relative.endswith(".partial") and relative[: -len(".partial")] in allowed
    )


def _actual_immutable_paths(root: Path) -> set[str]:
    actual: set[str] = set()
    for directory, names, files in os.walk(str(root), followlinks=False):
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
            if path.is_symlink() or path.is_file():
                actual.add(relative)
    return actual


def _gate_command(
    python: Path, server_label: str, cpu_threads: int
) -> list[str]:
    return [
        str(python),
        "-u",
        "reproducibility/run_server_trainer_gates_v7.py",
        "--server-label",
        server_label,
        "--device",
        "cuda:0",
        "--cpu-threads",
        str(cpu_threads),
        "--reward-loss-output",
        OUTPUT_PATHS["reward_loss"].as_posix(),
        "--entropy-output",
        OUTPUT_PATHS["hybrid_entropy"].as_posix(),
        "--precision-output",
        OUTPUT_PATHS["device_precision"].as_posix(),
        "--manifest-output",
        OUTPUT_PATHS["gate_chain"].as_posix(),
    ]


def _require_build_interpreter(python: Path) -> None:
    current = Path(sys.executable).resolve(strict=True)
    expected = python.resolve(strict=True)
    if current != expected:
        raise RuntimeError(
            "gate bundle must be built by --python so its runtime lock matches "
            f"the registered gate command: current={current}, expected={expected}"
        )


def build(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    compiler = args.compiler.expanduser().resolve(strict=True)
    python = args.python.expanduser().resolve(strict=True)
    _require_build_interpreter(python)
    if output.exists():
        raise FileExistsError(f"refusing to replace gate workspace {output}")
    if args.server_label not in SERVER_LABELS:
        raise ValueError(f"unsupported GPU server label: {args.server_label}")
    output.mkdir(parents=True)
    try:
        _copy_inputs(output, link_large_files=args.link_large_files)
        entries = _walk_entries(output)
        data_policy = _payload_policy(entries)
        expected_hdf5 = sorted(
            path.as_posix()
            for name, path in INPUT_PATHS.items()
            if name.endswith("dataset")
        )
        if data_policy["v7_hdf5_files"] != expected_hdf5:
            raise ValueError("gate bundle contains an unexpected HDF5 inventory")
        runtime_environment = _runtime_environment(output, compiler)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(ROOT),
            "bundle_label": args.label,
            "target_server": args.server_label,
            "entry_count": len(entries),
            "total_file_bytes": sum(
                int(item.get("bytes", 0)) for item in entries
            ),
            "aggregate_sha256": _aggregate(entries),
            "runtime_environment": _manifest_runtime_environment(compiler),
            "runtime_lock": _runtime_fingerprint(runtime_environment)["required"],
            "data_policy": data_policy,
            "inputs": {
                name: _descriptor(output / path, output)
                for name, path in INPUT_PATHS.items()
            },
            "mutable_outputs": {
                name: path.as_posix() for name, path in OUTPUT_PATHS.items()
            },
            "gate_command": _gate_command(
                python, args.server_label, args.cpu_threads
            ),
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
                    "gate_command": manifest["gate_command"],
                },
                sort_keys=True,
            )
        )
    except Exception:
        shutil.rmtree(str(output), ignore_errors=True)
        raise
    return 0


def _verify_descriptor(root: Path, descriptor: dict[str, Any], label: str) -> Path:
    relative = Path(str(descriptor.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} descriptor path is unsafe")
    path = root / relative
    if (
        not path.is_file()
        or path.stat().st_size != descriptor.get("bytes")
        or sha256(path) != descriptor.get("sha256")
    ):
        raise ValueError(f"{label} descriptor is missing or stale")
    return path


def _verify_gate_outputs(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    output_paths = {
        name: root / Path(relative)
        for name, relative in manifest.get("mutable_outputs", {}).items()
    }
    if set(output_paths) != set(OUTPUT_PATHS):
        raise ValueError("gate output allowlist drifted")
    for name, expected in OUTPUT_PATHS.items():
        if output_paths[name] != root / expected or not output_paths[name].is_file():
            raise ValueError(f"gate output is missing: {name}")
    chain = json.loads(output_paths["gate_chain"].read_text())
    if (
        chain.get("schema_version") != GATE_CHAIN_SCHEMA
        or chain.get("execution_boundary")
        != "registered GPU server; H2O+ trainer admission only"
        or chain.get("execution_host", {}).get("scheduler_label")
        != manifest.get("target_server")
        or chain.get("dependency_order")
        != ["reward_loss", "hybrid_entropy", "device_precision"]
    ):
        raise ValueError("server trainer gate-chain identity is invalid")
    for name in ("reward_loss", "hybrid_entropy", "device_precision"):
        descriptor = chain.get("outputs", {}).get(name, {})
        path = output_paths[name].resolve()
        if (
            Path(str(descriptor.get("path", ""))).resolve() != path
            or descriptor.get("bytes") != path.stat().st_size
            or descriptor.get("sha256") != sha256(path)
        ):
            raise ValueError(f"gate chain does not authenticate output {name}")
    for name, descriptor in manifest.get("inputs", {}).items():
        chain_descriptor = chain.get("inputs", {}).get(name, {})
        if chain_descriptor.get("sha256") != descriptor.get("sha256"):
            raise ValueError(f"gate chain uses another input: {name}")
    return {
        "chain_sha256": sha256(output_paths["gate_chain"]),
        "selected_arguments": chain.get("selected_arguments"),
        "execution_host": chain.get("execution_host"),
        "outputs": {
            name: {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in output_paths.items()
        },
    }


def verify(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve(strict=True)
    manifest_path = (
        args.manifest.expanduser().resolve(strict=True)
        if args.manifest is not None
        else root / MANIFEST_NAME
    )
    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    entries = manifest.get("entries")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("gate bundle manifest schema mismatch")
    if not isinstance(entries, list):
        raise ValueError("gate bundle entries must be a list")
    if _aggregate(entries) != manifest.get("aggregate_sha256"):
        errors.append("gate bundle aggregate hash mismatch")
    expected_paths = {str(item.get("path")) for item in entries}
    for item in entries:
        relative = str(item.get("path"))
        path = root / relative
        if item.get("type") == "symlink":
            if not path.is_symlink() or os.readlink(path) != item.get("target"):
                errors.append(f"symlink mismatch: {relative}")
        elif item.get("type") == "file":
            if (
                not path.is_file()
                or path.is_symlink()
                or path.stat().st_size != int(item.get("bytes", -1))
                or sha256(path) != item.get("sha256")
            ):
                errors.append(f"file mismatch: {relative}")
        else:
            errors.append(f"unsupported entry type: {relative}")
    unexpected = sorted(_actual_immutable_paths(root) - expected_paths)
    if unexpected:
        errors.append("unexpected immutable files: " + ", ".join(unexpected[:20]))

    runtime_fingerprint: dict[str, Any] | None = None
    gate_outputs: dict[str, Any] | None = None
    try:
        _payload_policy(entries)
        for name, descriptor in manifest.get("inputs", {}).items():
            path = _verify_descriptor(root, descriptor, f"gate input {name}")
            if path != root / INPUT_PATHS[name]:
                raise ValueError(f"gate input path drifted: {name}")
        runtime_environment = _resolve_manifest_runtime_environment(root, manifest)
        runtime_fingerprint = _runtime_fingerprint(runtime_environment)
        if runtime_fingerprint["required"] != manifest.get("runtime_lock"):
            raise ValueError("gate bundle runtime lock mismatch")
        if args.require_gates:
            gate_outputs = _verify_gate_outputs(root, manifest)
    except Exception as error:
        errors.append(str(error))
    report = {
        "schema_version": AUDIT_SCHEMA,
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "manifest": str(manifest_path),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "target_server": manifest.get("target_server"),
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
        "gate_outputs": gate_outputs,
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
    build_parser.add_argument("--server-label", choices=SERVER_LABELS, required=True)
    build_parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    build_parser.add_argument("--compiler", type=Path, default=DEFAULT_COMPILER)
    build_parser.add_argument("--cpu-threads", type=int, default=8)
    build_parser.add_argument("--link-large-files", action="store_true")
    build_parser.set_defaults(function=build)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--root", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path)
    verify_parser.add_argument("--require-gates", action="store_true")
    verify_parser.set_defaults(function=verify)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
