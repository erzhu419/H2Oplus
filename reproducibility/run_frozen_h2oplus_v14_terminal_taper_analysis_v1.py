#!/usr/bin/env python3
"""Run the frozen v14 analyzer with a fail-closed block-label adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ANALYZER = REPO / "reproducibility/analyze_h2oplus_v14_terminal_taper.py"
EXPECTED_ANALYZER_SHA256 = (
    "0df400805ac90e22be18a016bad42ce884cd946d576bdb95ad1d07afdbe0b297"
)
EXPECTED_METHODS_SHA256 = (
    "560a91eaca3e279cb986daedf6978c8275ce4c937859d60cd2eed37e27e52c3b"
)
REGISTERED_TO_MATRIX_BLOCK = {
    "v12-staged-release-development-01": "sumo-seed-5101",
    "v12-staged-release-development-02": "sumo-seed-5102",
    "v12-staged-release-development-03": "sumo-seed-5103",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_roots(argv: list[str]) -> list[Path]:
    roots: list[Path] = []
    index = 1
    while index < len(argv):
        item = argv[index]
        if item == "--matrix-root":
            if index + 1 >= len(argv):
                raise ValueError("--matrix-root lacks a value")
            roots.append(Path(argv[index + 1]).expanduser().resolve(strict=True))
            index += 2
            continue
        if item.startswith("--matrix-root="):
            roots.append(
                Path(item.split("=", 1)[1]).expanduser().resolve(strict=True)
            )
        index += 1
    return roots


def validate_block_labels(roots: list[Path]) -> tuple[str, ...]:
    if len(roots) != 3 or len(set(roots)) != 3:
        raise ValueError("v14 analysis requires three unique merged matrix roots")
    observed: dict[str, str] = {}
    for root in roots:
        registered = root.name
        actual = REGISTERED_TO_MATRIX_BLOCK.get(registered)
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        blocks = {str(job.get("environment_block")) for job in jobs}
        seeds = {int(job.get("sumo_seed", -1)) for job in jobs}
        expected_seed = 5100 + int(registered.rsplit("-", 1)[-1]) if actual else -1
        if (
            actual is None
            or manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != EXPECTED_METHODS_SHA256
            or len(manifest.get("source_sha256", {})) != 47
            or len(jobs) != 210
            or blocks != {actual}
            or seeds != {expected_seed}
            or manifest.get("matrix_merge", {}).get("all_outputs_verified") is not True
            or manifest.get("matrix_merge", {}).get("source_manifests") is None
            or len(manifest["matrix_merge"]["source_manifests"]) != 2
        ):
            raise ValueError(f"v14 registered/matrix block binding failed: {root}")
        observed[registered] = actual
    if observed != REGISTERED_TO_MATRIX_BLOCK:
        raise ValueError("v14 registered block coverage is incomplete")
    return tuple(REGISTERED_TO_MATRIX_BLOCK.values())


def load_analyzer():
    spec = importlib.util.spec_from_file_location("_frozen_h2oplus_v14_analysis", ANALYZER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v14 analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    actual_hash = sha256(ANALYZER)
    if actual_hash != EXPECTED_ANALYZER_SHA256:
        raise RuntimeError(f"frozen v14 analyzer identity drifted: {actual_hash}")
    actual_blocks = validate_block_labels(matrix_roots(sys.argv))
    module = load_analyzer()
    registered_blocks = tuple(module.BLOCKS)
    if registered_blocks != tuple(REGISTERED_TO_MATRIX_BLOCK):
        raise RuntimeError("frozen v14 analyzer registered block contract drifted")

    original_default = json.JSONEncoder.default

    def numpy_scalar_default(encoder: json.JSONEncoder, value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        return original_default(encoder, value)

    module.BLOCKS = actual_blocks
    json.JSONEncoder.default = numpy_scalar_default
    try:
        result = int(module.main())
    finally:
        json.JSONEncoder.default = original_default
        if sha256(ANALYZER) != EXPECTED_ANALYZER_SHA256:
            raise RuntimeError("frozen v14 analyzer changed during execution")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
