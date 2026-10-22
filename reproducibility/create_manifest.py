#!/usr/bin/env python3
"""Create a machine-readable provenance manifest without mutating inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CHUNK_BYTES = 8 * 1024 * 1024


def _run(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            args, cwd=ROOT, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_values(dataset: Any, chunk_rows: int = 1_000_000) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for start in range(0, int(dataset.shape[0]), chunk_rows):
        values = dataset[start : start + chunk_rows]
        unique, frequencies = __import__("numpy").unique(values, return_counts=True)
        counts.update({str(item.item()): int(n) for item, n in zip(unique, frequencies)})
    return dict(sorted(counts.items()))


def _jsonable(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _hdf5_summary(path: Path) -> dict[str, Any] | None:
    if path.suffix.lower() not in {".h5", ".hdf5"} and ".h5." not in path.name:
        return None
    try:
        import h5py
    except ImportError:
        return {"error": "h5py is not installed"}

    try:
        with h5py.File(path, "r") as handle:
            datasets = {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "compression": value.compression,
                }
                for key, value in handle.items()
                if isinstance(value, h5py.Dataset)
            }
            summary: dict[str, Any] = {
                "attrs": {key: _jsonable(value) for key, value in handle.attrs.items()},
                "datasets": datasets,
            }
            for key in ("policy_id", "seed", "terminals", "timeouts", "snap_valid"):
                if key in handle and len(handle[key].shape) == 1:
                    summary[f"{key}_counts"] = _count_values(handle[key])
            return summary
    except OSError as exc:
        return {"error": str(exc)}


def _file_record(path_text: str, hash_content: bool) -> dict[str, Any]:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return record
    stat = path.stat()
    record.update(
        {
            "size_bytes": stat.st_size,
            "mtime_utc": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        }
    )
    if path.is_file() and hash_content:
        record["sha256"] = _sha256(path)
    hdf5 = _hdf5_summary(path) if path.is_file() else None
    if hdf5 is not None:
        record["hdf5"] = hdf5
    return record


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # runtime provenance must survive optional packages
            versions[name] = f"unavailable: {type(exc).__name__}"
    return versions


def build_manifest(label: str, inputs: list[str], hash_content: bool) -> dict[str, Any]:
    git_diff = _run("git", "diff", "--binary") or ""
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "label": label,
        "created_utc": datetime.now(tz=timezone.utc).isoformat(),
        "status": "legacy-invalid" if "legacy" in label else "candidate",
        "repository": {
            "root": str(ROOT),
            "commit": _run("git", "rev-parse", "HEAD"),
            "branch": _run("git", "branch", "--show-current"),
            "status_porcelain": (_run("git", "status", "--porcelain=v1") or "").splitlines(),
            "tracked_diff_sha256": hashlib.sha256(git_diff.encode()).hexdigest(),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": _package_versions(("numpy", "h5py", "torch", "scipy")),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "sumo_home": os.environ.get("SUMO_HOME"),
            "sumo_version": _run("sumo", "--version"),
            "nvidia_smi": _run(
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader",
            ),
        },
        "inputs": [_file_record(path, hash_content) for path in inputs],
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Destination JSON path")
    parser.add_argument("--label", required=True, help="Stable human-readable run label")
    parser.add_argument("--input", action="append", default=[], help="Input file to record")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Skip file hashing; not allowed for submission evidence",
    )
    args = parser.parse_args()

    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.label, args.input, not args.metadata_only)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(output)


if __name__ == "__main__":
    main()
