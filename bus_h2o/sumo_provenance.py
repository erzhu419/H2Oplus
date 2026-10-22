"""Deterministic provenance helpers for canonical SUMO experiments."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


SUMO_RUNTIME_FINGERPRINT_SCHEMA = "transit-sumo-runtime-fingerprint-v1"
SUMO_RUNTIME_LOCK_SCHEMA = "transit-sumo-runtime-lock-v7-v1"
SUMO_BACKENDS = frozenset({"libsumo", "traci"})
FORMAL_V7_SUMO_BACKEND = "libsumo"
FORMAL_V7_SUMO_VERSION = "1.25.0"
_VERSION_PATTERN = re.compile(r"\b(\d+\.\d+\.\d+)\b")


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest for one required file."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: Iterable[Path]) -> dict[str, str]:
    """Hash a deterministic, duplicate-free set of required files."""

    resolved = sorted({path.expanduser().resolve() for path in paths})
    return {str(path): sha256(path) for path in resolved}


def _json_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _semantic_version(value: str, *, source: str) -> str:
    match = _VERSION_PATTERN.search(value)
    if match is None:
        raise RuntimeError(f"unable to parse a SUMO version from {source}: {value!r}")
    return match.group(1)


def _sumo_executable() -> Path:
    value = shutil.which("sumo")
    if value is None:
        raise RuntimeError("unable to locate the active SUMO executable")
    resolved = Path(value).expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"active SUMO executable is not a file: {resolved}")
    return resolved


def _external_runtime_binary(launcher: Path) -> Path:
    """Resolve the binary behind the small eclipse-sumo Python entry point."""

    try:
        prefix = launcher.read_bytes()[:4096]
    except OSError:
        return launcher
    if b"from sumo import sumo" not in prefix:
        return launcher
    spec = importlib.util.find_spec("sumo")
    if spec is None or spec.origin is None:
        raise RuntimeError(f"SUMO launcher {launcher} imports an unavailable package")
    bundled = Path(spec.origin).resolve().parent / "bin" / "sumo"
    if not bundled.is_file():
        raise RuntimeError(f"SUMO launcher {launcher} has no bundled binary {bundled}")
    return bundled.resolve()


def sumo_version(executable: Path | None = None) -> str:
    """Return the first line of an external SUMO executable's version."""

    binary = _sumo_executable() if executable is None else executable.resolve()

    result = subprocess.run(
        [str(binary), "--version"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0 or not result.stdout.splitlines():
        raise RuntimeError(f"unable to identify SUMO executable {binary}")
    return result.stdout.splitlines()[0].strip()


def _distribution_fingerprint(name: str, package_roots: tuple[str, ...]) -> dict:
    """Hash installed package code without binding its host-specific path."""

    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(f"required SUMO distribution {name!r} is not installed") from error
    files = []
    native_extensions = {}
    for package_file in distribution.files or ():
        relative = Path(str(package_file))
        if not relative.parts or relative.parts[0] not in package_roots:
            continue
        if "__pycache__" in relative.parts or relative.suffix == ".pyc":
            continue
        resolved = Path(distribution.locate_file(package_file)).resolve()
        if not resolved.is_file():
            raise RuntimeError(
                f"installed {name!r} file declared by metadata is missing: {resolved}"
            )
        entry = {
            "relative_path": relative.as_posix(),
            "path": str(resolved),
            "size_bytes": int(resolved.stat().st_size),
            "sha256": sha256(resolved),
        }
        files.append(entry)
        if resolved.suffix in {".so", ".pyd", ".dll", ".dylib"}:
            native_extensions[relative.as_posix()] = entry["sha256"]
    if not files:
        raise RuntimeError(f"no installed package files found for {name!r}")
    files.sort(key=lambda item: item["relative_path"])
    portable_files = [
        {key: entry[key] for key in ("relative_path", "size_bytes", "sha256")}
        for entry in files
    ]
    content_signature = _json_sha256(
        {
            "distribution": name,
            "version": distribution.version,
            "files": portable_files,
        }
    )
    return {
        "distribution": name,
        "version": distribution.version,
        "content_signature": content_signature,
        "files": files,
        "native_extension_sha256": native_extensions,
    }


def portable_sumo_runtime_identity(fingerprint: dict) -> dict:
    """Strip installation paths while retaining every executable code digest."""

    if fingerprint.get("schema_version") != SUMO_RUNTIME_FINGERPRINT_SCHEMA:
        raise ValueError("unsupported SUMO runtime fingerprint schema")
    distributions = {}
    for name, values in sorted(fingerprint["distributions"].items()):
        distributions[name] = {
            "version": values["version"],
            "content_signature": values["content_signature"],
            "native_extension_sha256": values["native_extension_sha256"],
        }
    executable = fingerprint["external_executable"]
    return {
        "schema_version": SUMO_RUNTIME_FINGERPRINT_SCHEMA,
        "backend": fingerprint["backend"],
        "runtime_version": fingerprint["runtime_version"],
        "backend_api": fingerprint["backend_api"],
        "python": fingerprint["python"],
        "platform": fingerprint["platform"],
        "distributions": distributions,
        "external_executable": {
            "used_by_backend": executable["used_by_backend"],
            "version_line": executable["version_line"],
            "version": executable["version"],
            "size_bytes": executable["size_bytes"],
            "sha256": executable["sha256"],
            "runtime_binary_size_bytes": executable[
                "runtime_binary_size_bytes"
            ],
            "runtime_binary_sha256": executable["runtime_binary_sha256"],
        },
    }


def sumo_runtime_fingerprint(
    backend: str, *, expected_version: str | None = None
) -> dict:
    """Identify the actual SUMO backend and all executable runtime content."""

    if backend not in SUMO_BACKENDS:
        raise ValueError(f"SUMO backend must be one of {sorted(SUMO_BACKENDS)}")
    configure_sumo_backend_environment(backend)

    distribution_names = ("traci", "sumolib")
    if backend == "libsumo":
        distribution_names = ("libsumo", *distribution_names)
    distributions = {
        name: _distribution_fingerprint(name, (name,))
        for name in distribution_names
    }

    binary = _sumo_executable()
    runtime_binary = _external_runtime_binary(binary)
    executable_version_line = sumo_version(binary)
    executable_version = _semantic_version(
        executable_version_line, source=str(binary)
    )
    if backend == "libsumo":
        traci_module = importlib.import_module("traci")
        if not bool(getattr(traci_module, "isLibsumo", lambda: False)()):
            raise RuntimeError("requested libsumo backend did not bind traci to libsumo")
        module = importlib.import_module("libsumo")
        if not hasattr(module, "getVersion"):
            raise RuntimeError("installed libsumo lacks getVersion()")
        api_value = module.getVersion()
        if not (
            isinstance(api_value, (tuple, list))
            and len(api_value) == 2
            and isinstance(api_value[1], str)
        ):
            raise RuntimeError(f"unexpected libsumo getVersion() result: {api_value!r}")
        runtime_version = _semantic_version(
            api_value[1], source="libsumo.getVersion()"
        )
        backend_api = {
            "source": "libsumo.getVersion",
            "api_integer": int(api_value[0]),
            "version_line": api_value[1],
        }
        required_versions = {
            name: values["version"] for name, values in distributions.items()
        }
        mismatches = {
            name: value
            for name, value in required_versions.items()
            if value != runtime_version
        }
        if mismatches:
            raise RuntimeError(
                "libsumo/traci/sumolib package versions disagree with the loaded "
                f"libsumo API {runtime_version}: {mismatches}"
            )
    else:
        runtime_version = executable_version
        backend_api = {
            "source": "external-sumo-executable",
            "api_integer": None,
            "version_line": executable_version_line,
        }
        mismatches = {
            name: values["version"]
            for name, values in distributions.items()
            if values["version"] != runtime_version
        }
        if mismatches:
            raise RuntimeError(
                "traci/sumolib package versions disagree with the external SUMO "
                f"executable {runtime_version}: {mismatches}"
            )

    if expected_version is not None and runtime_version != expected_version:
        raise RuntimeError(
            f"SUMO runtime version mismatch: requested {expected_version}, "
            f"loaded {runtime_version}"
        )

    fingerprint = {
        "schema_version": SUMO_RUNTIME_FINGERPRINT_SCHEMA,
        "backend": backend,
        "runtime_version": runtime_version,
        "backend_api": backend_api,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "soabi": sysconfig.get_config_var("SOABI"),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
        },
        "distributions": distributions,
        "external_executable": {
            "path": str(binary),
            "used_by_backend": backend == "traci",
            "version_line": executable_version_line,
            "version": executable_version,
            "size_bytes": int(binary.stat().st_size),
            "sha256": sha256(binary),
            "runtime_binary_path": str(runtime_binary),
            "runtime_binary_size_bytes": int(runtime_binary.stat().st_size),
            "runtime_binary_sha256": sha256(runtime_binary),
        },
    }
    portable = portable_sumo_runtime_identity(fingerprint)
    fingerprint["portable_identity"] = portable
    fingerprint["portable_signature"] = _json_sha256(portable)
    return fingerprint


def configure_sumo_backend_environment(backend: str) -> None:
    """Bind traci's process-wide import mode before the bridge is imported."""

    if backend not in SUMO_BACKENDS:
        raise ValueError(f"SUMO backend must be one of {sorted(SUMO_BACKENDS)}")
    traci_module = sys.modules.get("traci")
    if traci_module is not None:
        is_libsumo = bool(
            getattr(traci_module, "isLibsumo", lambda: False)()
        )
        if is_libsumo != (backend == "libsumo"):
            raise RuntimeError(
                "traci was imported before the requested SUMO backend was bound"
            )
        return
    if backend == "libsumo":
        os.environ["LIBSUMO_AS_TRACI"] = "quiet"
    else:
        os.environ.pop("LIBSUMO_AS_TRACI", None)


def load_sumo_runtime_lock(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[dict, dict]:
    """Load a runtime lock and verify its self-contained portable signature."""

    resolved = path.expanduser().resolve()
    actual_sha256 = sha256(resolved)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError(
            f"SUMO runtime lock hash mismatch: expected {expected_sha256}, "
            f"got {actual_sha256}"
        )
    payload = json.loads(resolved.read_text())
    if payload.get("schema_version") != SUMO_RUNTIME_LOCK_SCHEMA:
        raise ValueError("unsupported SUMO runtime lock schema")
    fingerprint = payload.get("runtime_fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("SUMO runtime lock lacks a fingerprint")
    portable = portable_sumo_runtime_identity(fingerprint)
    signature = _json_sha256(portable)
    if (
        fingerprint.get("portable_identity") != portable
        or fingerprint.get("portable_signature") != signature
        or payload.get("runtime_signature") != signature
    ):
        raise ValueError("SUMO runtime lock fingerprint is internally inconsistent")
    return payload, {
        "path": str(resolved),
        "sha256": actual_sha256,
        "schema_version": SUMO_RUNTIME_LOCK_SCHEMA,
        "runtime_signature": signature,
    }


def sumo_input_paths(config_path: Path) -> tuple[Path, ...]:
    """Resolve every file declared in a SUMO configuration's input block."""

    config = config_path.expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    root = ET.parse(config).getroot()
    input_block = root.find("input")
    if input_block is None:
        raise ValueError(f"SUMO configuration has no input block: {config}")

    paths = {config}
    for element in input_block:
        raw_value = element.get("value", "")
        for value in raw_value.split(","):
            value = value.strip()
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = config.parent / path
            paths.add(path.resolve())
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"SUMO input referenced by {config} does not exist: {path}"
            )
    return tuple(sorted(paths))


def load_validated_scenario(path: Path) -> tuple[dict, dict]:
    """Load a demand manifest and verify both declared and runtime inputs."""

    manifest_path = path.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "transit-passenger-demand-v1":
        raise ValueError("scenario manifest has an unsupported schema")

    for raw_path, expected_hash in manifest.get("scenario_sha256", {}).items():
        source = Path(raw_path).expanduser()
        if not source.is_absolute():
            source = manifest_path.parent / source
        source = source.resolve()
        actual_hash = sha256(source)
        if actual_hash != expected_hash:
            raise ValueError(
                f"scenario hash mismatch for {source}: "
                f"expected {expected_hash}, got {actual_hash}"
            )

    sumo_config = Path(manifest["sumo_config"]).expanduser()
    passenger_objects = Path(manifest["passenger_objects"]).expanduser()
    if not sumo_config.is_absolute():
        sumo_config = manifest_path.parent / sumo_config
    if not passenger_objects.is_absolute():
        passenger_objects = manifest_path.parent / passenger_objects
    sumo_config = sumo_config.resolve()
    passenger_objects = passenger_objects.resolve()
    runtime_inputs = hash_files((*sumo_input_paths(sumo_config), passenger_objects))
    evidence = dict(manifest)
    evidence.update(
        {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "runtime_input_sha256": runtime_inputs,
        }
    )
    runtime = {
        "sumo_cfg": str(sumo_config),
        "passenger_obj_xml": str(passenger_objects),
    }
    return runtime, evidence


def collection_source_paths(
    root: Path,
    sumo_dir: Path,
    edge_xml: Path,
    contract_version: str = "v4",
) -> tuple[Path, ...]:
    """Return all Python/static inputs that define one collector process."""

    root = root.expanduser().resolve()
    sumo_dir = sumo_dir.expanduser().resolve()
    if contract_version not in {"v4", "v5", "v6", "v7"}:
        raise ValueError("contract_version must be 'v4', 'v5', 'v6', or 'v7'")
    paths = {
        root / "collect_policy" / f"collect_transit_{contract_version}.py",
        root / "bus_h2o" / "sumo_provenance.py",
        root / "bus_h2o" / f"mdp_contract_{contract_version}.py",
        root / "bus_h2o" / "common" / "data_utils.py",
        root / "bus_h2o" / "envs" / "all_lines_sumo_env.py",
        root / "bus_h2o" / "envs" / "sumo_gym_env.py",
        root / "bus_h2o" / "sumo_env" / "rl_bridge.py",
        root / "bus_h2o" / "sumo_env" / "rl_env.py",
        root / "SimpleSAC" / "transit_v3" / "evaluation.py",
        root / "SimpleSAC" / "transit_v4" / "policies.py",
        edge_xml.expanduser().resolve(),
    }
    if contract_version in {"v5", "v6", "v7"}:
        paths.update(
            {
                root / "bus_h2o" / "mdp_contract_v4.py",
                root / "bus_h2o" / "mdp_contract_v5.py",
                root / "bus_h2o" / "passenger_time.py",
                root / "SimpleSAC" / "transit_v5" / "evaluation.py",
            }
        )
    if contract_version == "v7":
        paths.add(root / "bus_h2o" / "full_service_trace_v7.py")
    for name in (
        "d_8_compute_running_time.py",
        "e_8_gurobi_test_considerbusnum_V3.py",
        "f_8_create_obj.py",
    ):
        paths.add(root / "bus_h2o" / "sumo_env" / "case" / name)
    paths.update((sumo_dir / "sim_obj").glob("*.py"))
    for name in (
        "save_obj_bus.add.xml",
        "save_obj_lane.add.xml",
        "save_obj_line.add.xml",
        "save_obj_signal.add.xml",
        "save_obj_stop.add.xml",
    ):
        paths.add(sumo_dir / "initialize_obj" / name)
    return tuple(sorted(path.resolve() for path in paths))


__all__ = [
    "FORMAL_V7_SUMO_BACKEND",
    "FORMAL_V7_SUMO_VERSION",
    "SUMO_BACKENDS",
    "SUMO_RUNTIME_FINGERPRINT_SCHEMA",
    "SUMO_RUNTIME_LOCK_SCHEMA",
    "collection_source_paths",
    "configure_sumo_backend_environment",
    "hash_files",
    "load_validated_scenario",
    "load_sumo_runtime_lock",
    "portable_sumo_runtime_identity",
    "sha256",
    "sumo_input_paths",
    "sumo_runtime_fingerprint",
    "sumo_version",
]
