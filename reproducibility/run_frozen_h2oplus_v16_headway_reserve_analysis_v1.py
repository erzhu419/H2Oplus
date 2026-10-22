#!/usr/bin/env python3
"""Run the frozen v16 analyzer with a fail-closed block-label adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
ANALYZER = REPO / "reproducibility/analyze_h2oplus_v16_headway_reserve.py"
EVIDENCE = (
    REPO
    / "reproducibility/h2oplus_v16_headway_reserve_results/"
    "analysis_block_label_adapter_evidence_v1.json"
)
EXPECTED_ANALYZER_SHA256 = (
    "e46c04ffcab861ca045dafdacd8f7257da026213a82edf82bd8761fe15636834"
)
EXPECTED_METHODS_SHA256 = (
    "35047fd2ce616acbccd4592ae333a7ba84a1e4231c1a2505c7768d96b91de06a"
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
        if item == "--v16-matrix-root":
            if index + 1 >= len(argv):
                raise ValueError("--v16-matrix-root lacks a value")
            roots.append(Path(argv[index + 1]).expanduser().resolve(strict=True))
            index += 2
            continue
        if item.startswith("--v16-matrix-root="):
            roots.append(
                Path(item.split("=", 1)[1]).expanduser().resolve(strict=True)
            )
        index += 1
    return roots


def output_path(argv: list[str]) -> Path:
    for index, item in enumerate(argv[1:], start=1):
        if item == "--output" and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser().resolve()
        if item.startswith("--output="):
            return Path(item.split("=", 1)[1]).expanduser().resolve()
    raise ValueError("--output is required")


def validate_block_labels(roots: list[Path]) -> tuple[tuple[str, ...], list[dict]]:
    if len(roots) != 3 or len(set(roots)) != 3:
        raise ValueError("v16 analysis requires three unique matrix roots")
    observed: dict[str, str] = {}
    evidence: list[dict] = []
    for root in roots:
        registered = root.name
        actual = REGISTERED_TO_MATRIX_BLOCK.get(registered)
        manifest_path = root / "matrix_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        jobs = manifest.get("jobs", [])
        blocks = {str(job.get("environment_block")) for job in jobs}
        seeds = {int(job.get("sumo_seed", -1)) for job in jobs}
        expected_seed = 5100 + int(registered.rsplit("-", 1)[-1]) if actual else -1
        partition = manifest.get("matrix_partition", {})
        if (
            actual is None
            or manifest.get("schema_version") != "transit-evaluation-matrix-v7"
            or manifest.get("methods_manifest_sha256") != EXPECTED_METHODS_SHA256
            or len(manifest.get("source_sha256", {})) != 51
            or len(jobs) != 45
            or blocks != {actual}
            or seeds != {expected_seed}
            or partition.get("full_job_count") != 45
            or partition.get("selected_job_count") != 45
            or partition.get("shard_count") != 1
            or partition.get("shard_index") != 0
        ):
            raise ValueError(f"v16 registered/matrix block binding failed: {root}")
        observed[registered] = actual
        evidence.append(
            {
                "registered_block": registered,
                "matrix_block": actual,
                "manifest": str(manifest_path.relative_to(REPO)),
                "manifest_sha256": sha256(manifest_path),
                "jobs": 45,
                "source_file_count": 51,
                "sumo_seed": expected_seed,
            }
        )
    if observed != REGISTERED_TO_MATRIX_BLOCK:
        raise ValueError("v16 registered block coverage is incomplete")
    return tuple(REGISTERED_TO_MATRIX_BLOCK.values()), evidence


def load_analyzer():
    spec = importlib.util.spec_from_file_location(
        "_frozen_h2oplus_v16_analysis", ANALYZER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v16 analyzer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    actual_hash = sha256(ANALYZER)
    if actual_hash != EXPECTED_ANALYZER_SHA256:
        raise RuntimeError(f"frozen v16 analyzer identity drifted: {actual_hash}")
    if EVIDENCE.exists():
        raise FileExistsError(f"refusing to overwrite adapter evidence: {EVIDENCE}")
    actual_blocks, matrices = validate_block_labels(matrix_roots(sys.argv))
    output = output_path(sys.argv)
    module = load_analyzer()
    registered_blocks = tuple(module.BLOCKS)
    if registered_blocks != tuple(REGISTERED_TO_MATRIX_BLOCK):
        raise RuntimeError("frozen v16 analyzer registered block contract drifted")
    parent_registered_blocks = tuple(module.v14.BLOCKS)
    if parent_registered_blocks != tuple(REGISTERED_TO_MATRIX_BLOCK):
        raise RuntimeError("frozen v14 helper registered block contract drifted")

    original_default = json.JSONEncoder.default

    def numpy_scalar_default(encoder: json.JSONEncoder, value: object) -> object:
        if isinstance(value, np.generic):
            return value.item()
        return original_default(encoder, value)

    module.BLOCKS = actual_blocks
    module.v14.BLOCKS = actual_blocks
    json.JSONEncoder.default = numpy_scalar_default
    try:
        result = int(module.main())
    finally:
        json.JSONEncoder.default = original_default
        if sha256(ANALYZER) != EXPECTED_ANALYZER_SHA256:
            raise RuntimeError("frozen v16 analyzer changed during execution")
    if not output.is_file():
        raise FileNotFoundError(f"frozen analyzer did not create output: {output}")
    atomic_json(
        EVIDENCE,
        {
            "schema_version": (
                "transit-h2oplus-v16-headway-reserve-analysis-block-label-"
                "adapter-evidence-v1"
            ),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_analyzer": {
                "path": str(ANALYZER.relative_to(REPO)),
                "sha256": actual_hash,
            },
            "frozen_analyzer_exit_code": result,
            "analysis_output": {
                "path": str(output.relative_to(REPO)),
                "sha256": sha256(output),
                "size_bytes": output.stat().st_size,
            },
            "registered_to_matrix_block_mapping": REGISTERED_TO_MATRIX_BLOCK,
            "matrices": matrices,
            "adapter_scope": (
                "in-memory v16 analyzer and imported frozen v14 helper "
                "block-label constants only"
            ),
            "result_files_or_manifests_modified": False,
            "performance_values_or_preregistered_gate_formulas_changed": False,
            "formal_held_out_outcomes_read": False,
            "formal_confirmation_stage_authorized_or_submitted": False,
            "passed": True,
        },
    )
    return result


if __name__ == "__main__":
    raise SystemExit(main())
