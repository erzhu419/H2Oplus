#!/usr/bin/env python3
"""Build and verify a minimal immutable transit-smdp-v6 workspace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCHEMA_VERSION = "transit-v6-deployment-bundle-v2"
MANIFEST_NAME = "deployment_manifest.json"
FORMAL_DATASET_STEM = "transit_smdp_v6_formal_v1"
TOOLCHAIN_INCLUDE = Path(
    "reproducibility/toolchain/libxcrypt-4.4.36-hd590300_1/include"
)
DEFAULT_COMPILER = Path(
    "/home/erzhu419/anaconda3/envs/LSTM-RL/bin/"
    "x86_64-conda-linux-gnu-gcc"
)
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _temporary_environment(values: dict[str, str]):
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _runtime_fingerprint(environment: dict[str, str]) -> dict[str, Any]:
    import torch
    from SimpleSAC.train_transit_v3 import collect_runtime_fingerprint

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    with _temporary_environment(environment):
        return collect_runtime_fingerprint(torch, device)


def _runtime_environment(root: Path, compiler: Path) -> dict[str, str]:
    resolved_root = root.expanduser().resolve(strict=True)
    resolved_compiler = compiler.expanduser().resolve(strict=True)
    include = (resolved_root / TOOLCHAIN_INCLUDE).resolve(strict=True)
    try:
        include.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("runtime include directory escapes deployment root") from error
    if not resolved_compiler.is_file() or not os.access(
        str(resolved_compiler), os.X_OK
    ):
        raise FileNotFoundError(
            f"deployment compiler is not executable: {resolved_compiler}"
        )
    if not (include / "crypt.h").is_file():
        raise FileNotFoundError(
            f"deployment compatibility header is missing: {include / 'crypt.h'}"
        )
    return {"CC": str(resolved_compiler), "CPATH": str(include)}


def _manifest_runtime_environment(compiler: Path) -> dict[str, str]:
    return {
        "CC": str(compiler.expanduser().resolve(strict=True)),
        "CPATH": TOOLCHAIN_INCLUDE.as_posix(),
    }


def _resolve_manifest_runtime_environment(
    root: Path, manifest: dict[str, Any]
) -> dict[str, str]:
    configured = manifest.get("runtime_environment")
    if not isinstance(configured, dict) or set(configured) != {"CC", "CPATH"}:
        raise ValueError("deployment runtime_environment is invalid")
    cpath = Path(configured["CPATH"])
    if cpath.is_absolute() or ".." in cpath.parts:
        raise ValueError("deployment CPATH must be package-relative")
    return _runtime_environment(root, Path(configured["CC"]))


def _copy_file(source: Path, target: Path, *, link_large_files: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if link_large_files and source.stat().st_size >= 100 * 1024 * 1024:
        os.link(str(source), str(target))
    else:
        shutil.copy2(str(source), str(target))


def _python_sources() -> Iterable[Path]:
    for source_root in (
        ROOT / "SimpleSAC",
        ROOT / "bus_h2o",
        ROOT / "collect_policy",
    ):
        for path in source_root.rglob("*.py"):
            if "__pycache__" not in path.parts:
                yield path
    for path in (ROOT / "reproducibility").iterdir():
        if path.is_file() and path.suffix in {".py", ".json", ".md"}:
            yield path
    for path in (ROOT / "tests").glob("*.py"):
        yield path
    toolchain_root = ROOT / "reproducibility" / "toolchain"
    if toolchain_root.is_dir():
        for path in toolchain_root.rglob("*"):
            if path.is_file():
                yield path
    yield ROOT / "requirements.txt"
    yield ROOT / "pytest.ini"
    yield ROOT / "bus_h2o" / "network_data" / "a_sorted_busline_edge.xml"


def _formal_dataset_files() -> Iterable[Path]:
    dataset_dir = ROOT / "bus_h2o" / "datasets_v6"
    for path in sorted(dataset_dir.glob(FORMAL_DATASET_STEM + ".*")):
        if path.is_file():
            yield path


def _copy_calibrated_environment(destination_root: Path) -> None:
    source = ROOT / "bus_h2o" / "calibrated_env"
    target = destination_root / source.relative_to(ROOT)
    shutil.copytree(str(source), str(target), symlinks=True)

    # Source links encode the developer's absolute workspace.  Convert them to
    # package-internal relative links so the bundle is relocatable.
    for directory, names, files in os.walk(str(target), followlinks=False):
        parent = Path(directory)
        for name in list(names) + list(files):
            link = parent / name
            if not link.is_symlink():
                continue
            relative = link.relative_to(target)
            source_link = source / relative
            resolved_source = source_link.resolve(strict=True)
            try:
                resolved_relative = resolved_source.relative_to(source)
            except ValueError as error:
                raise ValueError(
                    f"calibrated-environment link escapes source tree: {source_link}"
                ) from error
            resolved_target = target / resolved_relative
            relative_target = os.path.relpath(str(resolved_target), str(link.parent))
            is_directory = resolved_source.is_dir()
            link.unlink()
            link.symlink_to(relative_target, target_is_directory=is_directory)


def _walk_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for directory, names, files in os.walk(str(root), followlinks=False):
        parent = Path(directory)
        for name in sorted(list(names) + list(files)):
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if relative == MANIFEST_NAME or path.is_dir() and not path.is_symlink():
                continue
            if path.is_symlink():
                entries.append(
                    {
                        "path": relative,
                        "type": "symlink",
                        "target": os.readlink(str(path)),
                    }
                )
            elif path.is_file():
                entries.append(
                    {
                        "path": relative,
                        "type": "file",
                        "bytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    entries.sort(key=lambda item: item["path"])
    return entries


def _aggregate(entries: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    compiler = args.compiler.expanduser().resolve(strict=True)
    if output.exists():
        raise FileExistsError(f"refusing to replace deployment workspace {output}")
    output.mkdir(parents=True)
    try:
        seen: set[Path] = set()
        calibrated_root = ROOT / "bus_h2o" / "calibrated_env"
        for source in list(_python_sources()) + list(_formal_dataset_files()):
            source = source.resolve()
            if source in seen or calibrated_root in source.parents:
                continue
            seen.add(source)
            _copy_file(
                source,
                output / source.relative_to(ROOT),
                link_large_files=args.link_large_files,
            )
        _copy_calibrated_environment(output)
        entries = _walk_entries(output)
        runtime_environment = _runtime_environment(output, compiler)
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
            "runtime_lock": _runtime_fingerprint(runtime_environment)["required"],
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
                    "entry_count": len(entries),
                    "total_file_bytes": manifest["total_file_bytes"],
                },
                sort_keys=True,
            )
        )
    except Exception:
        shutil.rmtree(str(output), ignore_errors=True)
        raise
    return 0


def _actual_paths(root: Path) -> set[str]:
    actual: set[str] = set()
    for directory, names, files in os.walk(str(root), followlinks=False):
        parent = Path(directory)
        for name in list(names) + list(files):
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if relative == MANIFEST_NAME or "__pycache__" in path.parts:
                continue
            if path.is_symlink() or path.is_file():
                if path.suffix != ".pyc":
                    actual.add(relative)
    return actual


def verify(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else root / MANIFEST_NAME
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    runtime: dict[str, Any] | None = None
    try:
        runtime_environment = _resolve_manifest_runtime_environment(root, manifest)
        runtime_fingerprint = _runtime_fingerprint(runtime_environment)
        runtime = runtime_fingerprint["required"]
        if runtime != manifest.get("runtime_lock"):
            errors.append(
                "runtime lock mismatch: expected="
                + json.dumps(manifest.get("runtime_lock"), sort_keys=True)
                + " actual="
                + json.dumps(runtime, sort_keys=True)
            )
    except Exception as error:
        errors.append(f"runtime environment audit failed: {error}")
    report = {
        "schema_version": "transit-v6-deployment-audit-v2",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "manifest": str(manifest_path),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "runtime": runtime,
        "runtime_environment": runtime_environment,
        "host_observation": (
            runtime_fingerprint["host"]
            if runtime_fingerprint is not None
            else None
        ),
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
    build_parser.add_argument("--link-large-files", action="store_true")
    build_parser.add_argument(
        "--compiler", type=Path, default=DEFAULT_COMPILER
    )
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
